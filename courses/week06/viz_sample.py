import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.stats import norm

from model_celeb import VAE
from dataset import CelebADataset

# Dataset & transform
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = CelebADataset(root_dir='./data/celeba/Img', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("model_celeba_VAE.pth", map_location=device,weights_only=True))
model.eval()

# Ambil latent
latents = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        x = batch.to(device)
        z, mu, logvar, _ = model.encoder(x)
        latents.append(z.cpu())
        if len(latents) * x.size(0) >= 1000:
            break

latents = torch.cat(latents, dim=0).numpy()  # [N, z_dim]

# Plot dimensi 0–49
num_dims = 50
cols = 10
rows = num_dims // cols

plt.figure(figsize=(20, 8))
for i in range(num_dims):
    ax = plt.subplot(rows, cols, i + 1)
    sns.histplot(latents[:, i], bins=30, stat='density', kde=False, color='navy', ax=ax)

    # Tambah PDF normal
    x_range = np.linspace(-4, 4, 200)
    y_pdf = norm.pdf(x_range, loc=0, scale=1)
    ax.plot(x_range, y_pdf, color='darkorange', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(i), fontsize=12)

plt.tight_layout()
plt.suptitle("Distributions of Latent Variables (Dim 0–49)", fontsize=14, y=1.02)
plt.show()
