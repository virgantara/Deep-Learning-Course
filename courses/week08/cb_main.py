import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
from tqdm import tqdm

# 1. Rotation Transform Function
def rotate_image(img, label):
    angles = [0, 90, 180, 270]
    angle = angles[label]
    return img.rotate(angle)

# 2. Custom Dataset for Rotation Prediction

class RotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for Self-Supervised Learning (SSL) using rotation prediction,
        applied to Animals image structure.
        """
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform
        self.rotations = [0, 90, 180, 270]  # label 0 → 0°, 1 → 90°, etc.

    def __len__(self):
        return len(self.dataset) * len(self.rotations)

    def __getitem__(self, idx):
        # Tentukan index gambar dan rotasi
        img_idx = idx // len(self.rotations)
        rotation_label = idx % len(self.rotations)

        # Ambil gambar dari ImageFolder
        image, _ = self.dataset[img_idx]

        # Rotasi gambar sesuai label
        rotated_image = TF.rotate(image, self.rotations[rotation_label])

        # Transformasi tambahan jika ada
        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image, rotation_label

# 3. CNN Rotation Classifier
class RotationCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# 4. Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = RotationDataset("./data/animals", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RotationCNN()
criterion = nn.CrossEntropyLoss()  # corresponds to Eq (3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Training Loop
for epoch in range(5):
    total_loss = 0
    for images, labels in tqdm(dataloader):
        outputs = model(images)
        loss = criterion(outputs, labels)  # this corresponds to ℒ(Xi, θ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
