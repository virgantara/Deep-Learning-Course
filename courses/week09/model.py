import os
import random
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class ContrastivePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.class_to_indices = self._group_by_class()

    def _group_by_class(self):
        class_to_idx = {}
        for idx, (path, label) in enumerate(self.dataset.samples):
            class_to_idx.setdefault(label, []).append(idx)
        return class_to_idx

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        if self.transform:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img1)
        else:
            pos_1 = img1
            pos_2 = img1

        neg_label = label1
        while neg_label == label1:
            neg_label = random.choice(list(self.class_to_indices.keys()))
        
        neg_index = random.choice(self.class_to_indices[neg_label])
        img2, label2 = self.dataset[neg_index]
        neg_img = self.transform(img2) if self.transform else img2

        return pos_1, pos_2, neg_img

    def __len__(self):
        return len(self.dataset)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomApply([
    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    # ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
])

root_dir = '../week04/data/catndog/train'
contrastive_dataset = ContrastivePairDataset(root_dir=root_dir, transform=transform)
pos1, pos2, neg = contrastive_dataset[0]

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(pos1.permute(1, 2, 0))
axs[0].set_title("Positive View 1")
axs[0].axis("off")

axs[1].imshow(pos2.permute(1, 2, 0))
axs[1].set_title("Positive View 2")
axs[1].axis("off")

axs[2].imshow(neg.permute(1, 2, 0))
axs[2].set_title("Negative Sample")
axs[2].axis("off")

plt.tight_layout()
plt.show()