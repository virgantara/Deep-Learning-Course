import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader


# =========================================================
# IoU, Encode/Decode, NMS
# =========================================================
def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4] in xyxy
    boxes2: [M, 4] in xyxy
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def xyxy_to_cxcywh(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


def cxcywh_to_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def encode_boxes(gt_boxes, anchors, variances=(0.1, 0.2)):
    """
    gt_boxes: [N, 4] xyxy
    anchors: [N, 4] cxcywh
    return offsets [N,4]
    """
    gt_c = xyxy_to_cxcywh(gt_boxes)
    gcxcy = (gt_c[:, :2] - anchors[:, :2]) / (variances[0] * anchors[:, 2:])
    gwh = torch.log(gt_c[:, 2:] / anchors[:, 2:].clamp(min=1e-6)) / variances[1]
    return torch.cat([gcxcy, gwh], dim=1)


def decode_boxes(pred_locs, anchors, variances=(0.1, 0.2)):
    """
    pred_locs: [N,4]
    anchors: [N,4] cxcywh
    return boxes [N,4] xyxy
    """
    cxcy = pred_locs[:, :2] * variances[0] * anchors[:, 2:] + anchors[:, :2]
    wh = torch.exp(pred_locs[:, 2:] * variances[1]) * anchors[:, 2:]
    boxes = torch.cat([cxcy, wh], dim=1)
    return cxcywh_to_xyxy(boxes)


def iou_single(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: Tensor [N,4] xyxy
    scores: Tensor [N]
    return kept indices
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    scores_sorted, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        current_box = boxes[i]
        remaining = order[1:]

        ious = []
        for idx in remaining:
            iou_val = iou_single(current_box.tolist(), boxes[idx].tolist())
            ious.append(iou_val)

        ious = torch.tensor(ious, device=boxes.device)
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)


# =========================================================
# Anchor Generator
# =========================================================
class SSDAnchorGenerator:
    def __init__(self, image_size=300):
        self.image_size = image_size

        # Feature map sizes for 300x300 input with this educational SSD
        self.feature_map_sizes = [10, 5, 3, 2, 2, 2]

        # Number of boxes per location
        self.num_anchors = [4, 6, 6, 6, 4, 4]

        # Scales (normalized)
        self.scales = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]

        # Aspect ratios
        self.aspect_ratios = [
            [1.0, 2.0, 0.5, 1.5],
            [1.0, 2.0, 0.5, 3.0, 1/3, 1.5],
            [1.0, 2.0, 0.5, 3.0, 1/3, 1.5],
            [1.0, 2.0, 0.5, 3.0, 1/3, 1.5],
            [1.0, 2.0, 0.5, 1.5],
            [1.0, 2.0, 0.5, 1.5],
        ]

    def generate(self, device):
        anchors = []

        for k, fm_size in enumerate(self.feature_map_sizes):
            scale = self.scales[k]
            ars = self.aspect_ratios[k]

            for i in range(fm_size):
                for j in range(fm_size):
                    cx = (j + 0.5) / fm_size
                    cy = (i + 0.5) / fm_size

                    for ar in ars:
                        w = scale * math.sqrt(ar)
                        h = scale / math.sqrt(ar)
                        anchors.append([cx, cy, w, h])

        anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
        anchors[:, 2:] = anchors[:, 2:].clamp(max=1.0)
        return anchors


# =========================================================
# Matching
# =========================================================
def match_anchors_to_targets(anchors, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    anchors: [A,4] cxcywh
    gt_boxes: [G,4] xyxy normalized [0,1]
    gt_labels: [G]
    """
    device = anchors.device
    num_anchors = anchors.size(0)

    if gt_boxes.numel() == 0:
        matched_boxes = torch.zeros((num_anchors, 4), device=device)
        matched_labels = torch.zeros((num_anchors,), dtype=torch.long, device=device)
        return matched_boxes, matched_labels

    anchor_xyxy = cxcywh_to_xyxy(anchors)
    ious = box_iou(anchor_xyxy, gt_boxes)  # [A, G]

    best_gt_iou, best_gt_idx = ious.max(dim=1)
    best_anchor_iou, best_anchor_idx = ious.max(dim=0)

    # force each gt to match at least one anchor
    for gt_idx in range(gt_boxes.size(0)):
        best_anchor = best_anchor_idx[gt_idx]
        best_gt_idx[best_anchor] = gt_idx
        best_gt_iou[best_anchor] = 1.0

    matched_boxes = gt_boxes[best_gt_idx]
    matched_labels = gt_labels[best_gt_idx]

    matched_labels[best_gt_iou < iou_threshold] = 0  # background
    return matched_boxes, matched_labels


# =========================================================
# Multibox Loss
# =========================================================
class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, iou_threshold=0.5):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.iou_threshold = iou_threshold

    def forward(self, pred_locs, pred_confs, anchors, targets):
        """
        pred_locs: [B, A, 4]
        pred_confs: [B, A, C]
        anchors: [A,4] cxcywh
        targets: list of dicts with keys boxes [N,4], labels [N]
        """
        device = pred_locs.device
        batch_size = pred_locs.size(0)
        num_anchors = anchors.size(0)
        num_classes = pred_confs.size(2)

        all_gt_locs = []
        all_gt_labels = []

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)

            matched_boxes, matched_labels = match_anchors_to_targets(
                anchors, gt_boxes, gt_labels, iou_threshold=self.iou_threshold
            )
            encoded_boxes = encode_boxes(matched_boxes, anchors)

            all_gt_locs.append(encoded_boxes)
            all_gt_labels.append(matched_labels)

        gt_locs = torch.stack(all_gt_locs, dim=0)       # [B,A,4]
        gt_labels = torch.stack(all_gt_labels, dim=0)   # [B,A]

        pos_mask = gt_labels > 0
        num_pos = pos_mask.sum(dim=1, keepdim=True)

        # Localization loss
        loc_loss = F.smooth_l1_loss(
            pred_locs[pos_mask], gt_locs[pos_mask], reduction="sum"
        ) if pos_mask.any() else torch.tensor(0.0, device=device)

        # Confidence loss
        conf_loss_all = F.cross_entropy(
            pred_confs.view(-1, num_classes),
            gt_labels.view(-1),
            reduction="none"
        ).view(batch_size, num_anchors)

        conf_loss_pos = conf_loss_all.clone()
        conf_loss_pos[pos_mask] = 0.0

        # Hard negative mining
        _, loss_idx = conf_loss_pos.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(dim=1)

        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_anchors - 1)
        neg_mask = idx_rank < num_neg

        conf_mask = pos_mask | neg_mask
        conf_loss = F.cross_entropy(
            pred_confs[conf_mask],
            gt_labels[conf_mask],
            reduction="sum"
        ) if conf_mask.any() else torch.tensor(0.0, device=device)

        N = num_pos.sum().clamp(min=1).float()
        total_loss = (loc_loss + conf_loss) / N
        return total_loss, loc_loss / N, conf_loss / N


# =========================================================
# SSD Model with ResNet50 Backbone
# =========================================================
class SSD(nn.Module):
    def __init__(self, num_classes, debug_shapes=False):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.debug_shapes = debug_shapes

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B,2048,H,W]

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(2048, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(2048, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        locs = []
        confs = []

        x = self.features(x)
        if self.debug_shapes:
            print("Backbone output:", x.shape)

        locs.append(self.loc[0](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[0](x).permute(0, 2, 3, 1).contiguous())

        for i, layer in enumerate(self.extras):
            x = layer(x)
            if self.debug_shapes:
                print(f"Extra layer {i} output:", x.shape)
            locs.append(self.loc[i + 1](x).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf[i + 1](x).permute(0, 2, 3, 1).contiguous())

        locs = torch.cat([o.view(o.size(0), -1) for o in locs], dim=1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], dim=1)

        locs = locs.view(locs.size(0), -1, 4)
        confs = confs.view(confs.size(0), -1, self.num_classes)

        return locs, confs


# =========================================================
# Dummy Dataset for Demonstration
# Replace this with VOC/COCO/custom dataset
# =========================================================
class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples=100, image_size=300, num_classes=21):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_size, self.image_size)

        # 1-3 random boxes
        num_boxes = torch.randint(1, 4, (1,)).item()
        boxes = []
        labels = []

        for _ in range(num_boxes):
            x1 = torch.rand(1).item() * 0.7
            y1 = torch.rand(1).item() * 0.7
            w = 0.1 + torch.rand(1).item() * 0.2
            h = 0.1 + torch.rand(1).item() * 0.2
            x2 = min(x1 + w, 0.98)
            y2 = min(y1 + h, 0.98)
            boxes.append([x1, y1, x2, y2])
            labels.append(torch.randint(1, self.num_classes, (1,)).item())

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return image, target


def detection_collate(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images, dim=0), targets


# =========================================================
# Inference
# =========================================================
@torch.no_grad()
def detect(model, images, anchors, conf_threshold=0.4, nms_threshold=0.5, top_k=50):
    model.eval()
    pred_locs, pred_confs = model(images)

    pred_scores = F.softmax(pred_confs, dim=-1)
    batch_results = []

    for b in range(images.size(0)):
        boxes = decode_boxes(pred_locs[b], anchors).clamp(0.0, 1.0)
        scores = pred_scores[b]

        image_boxes = []
        image_labels = []
        image_scores = []

        for cls in range(1, scores.size(1)):  # skip background class 0
            cls_scores = scores[:, cls]
            keep = cls_scores > conf_threshold
            if keep.sum() == 0:
                continue

            cls_boxes = boxes[keep]
            cls_scores = cls_scores[keep]

            kept_idx = nms(cls_boxes, cls_scores, iou_threshold=nms_threshold)
            cls_boxes = cls_boxes[kept_idx]
            cls_scores = cls_scores[kept_idx]

            image_boxes.append(cls_boxes)
            image_scores.append(cls_scores)
            image_labels.append(torch.full((cls_boxes.size(0),), cls, dtype=torch.long, device=images.device))

        if len(image_boxes) == 0:
            batch_results.append({
                "boxes": torch.empty((0, 4), device=images.device),
                "scores": torch.empty((0,), device=images.device),
                "labels": torch.empty((0,), dtype=torch.long, device=images.device),
            })
            continue

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        if image_scores.numel() > top_k:
            vals, idxs = image_scores.topk(top_k)
            image_boxes = image_boxes[idxs]
            image_labels = image_labels[idxs]
            image_scores = vals

        batch_results.append({
            "boxes": image_boxes,
            "scores": image_scores,
            "labels": image_labels
        })

    return batch_results


# =========================================================
# Training Loop
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, anchors, device):
    model.train()
    total_loss = 0.0
    total_loc = 0.0
    total_conf = 0.0

    for images, targets in loader:
        images = images.to(device)

        optimizer.zero_grad()
        pred_locs, pred_confs = model(images)
        loss, loc_loss, conf_loss = criterion(pred_locs, pred_confs, anchors, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loc += loc_loss.item()
        total_conf += conf_loss.item()

    n = len(loader)
    return total_loss / n, total_loc / n, total_conf / n


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 21  # 20 classes + background
    image_size = 300
    batch_size = 4
    epochs = 3
    lr = 1e-4

    model = SSD(num_classes=num_classes, debug_shapes=False).to(device)

    anchor_generator = SSDAnchorGenerator(image_size=image_size)
    anchors = anchor_generator.generate(device=device)

    # quick shape check
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        locs, confs = model(dummy)
        print("Localization predictions:", locs.shape)
        print("Confidence predictions:", confs.shape)
        print("Generated anchors:", anchors.shape)

        if locs.size(1) != anchors.size(0):
            raise ValueError(
                f"Jumlah anchor tidak sama dengan jumlah prediksi. "
                f"pred={locs.size(1)}, anchors={anchors.size(0)}"
            )

    dataset = DummyDetectionDataset(num_samples=50, image_size=image_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)

    criterion = MultiBoxLoss(neg_pos_ratio=3, iou_threshold=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss, loc_loss, conf_loss = train_one_epoch(
            model, loader, optimizer, criterion, anchors, device
        )
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Total Loss: {loss:.4f} | "
            f"Loc Loss: {loc_loss:.4f} | "
            f"Conf Loss: {conf_loss:.4f}"
        )

    # inference demo
    images, targets = next(iter(loader))
    images = images.to(device)
    results = detect(model, images, anchors, conf_threshold=0.2, nms_threshold=0.5, top_k=10)

    print("\nInference result sample:")
    for i, result in enumerate(results[:2]):
        print(f"Image {i}:")
        print("Boxes:", result["boxes"][:5])
        print("Scores:", result["scores"][:5])
        print("Labels:", result["labels"][:5])