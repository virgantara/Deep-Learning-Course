import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()

        self.features = nn.Sequential(
            # Input: [B, 3, 32, 32]

            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)                               # [B, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def unnormalize(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    img = img.detach().cpu()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    return img

def visualize_predictions(model, loader, class_names, device, num_images=12):
    model.eval()

    images_shown = 0
    plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.show()
                    return

                img = unnormalize(images[i])
                img = img.permute(1, 2, 0).numpy()

                gt_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]

                plt.subplot(3, 4, images_shown + 1)
                plt.imshow(img)
                plt.axis("off")

                title = f"GT: {gt_label}\nPred: {pred_label}"

                if gt_label == pred_label:
                    plt.title(title, color="green")
                else:
                    plt.title(title, color="red")

                images_shown += 1

    plt.tight_layout()
    plt.show()


data_dir = "data/cifar10/cifar10"

test_dir = os.path.join(data_dir, "test")


test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFAR10CNN(num_classes=10).to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()

print("Model loaded successfully")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    ax=ax
)

ax.set_title("CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.show()

visualize_predictions(
    model=model,
    loader=test_loader,
    class_names=test_dataset.classes,
    device=device,
    num_images=12
)