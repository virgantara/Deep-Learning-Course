import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import Autoencoder
from dataset import FashionMNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device,weights_only=True))
model.eval()

# Load embeddings dan label asli
embeddings = np.load("embeddings.npy")  # shape: (N, 2)
labels = np.load("labels.npy")          # shape: (N,) — harus disiapkan saat visualisasi sebelumnya
x = np.linspace(-10, 7.5, 30)
y = np.linspace(-2.5, 8.5, 30)
grid_x, grid_y = np.meshgrid(x, y)
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (900, 2)

# Decode each point in the grid
with torch.no_grad():
    latent = torch.tensor(grid_points, dtype=torch.float32).to(device)
    reconstructions = model.decoder(latent).cpu().numpy()

# Begin plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Show decoded images on the grid
for i, (gx, gy) in enumerate(grid_points):
    img = reconstructions[i].squeeze()
    extent = [gx - 0.5, gx + 0.5, gy - 0.5, gy + 0.5]
    ax.imshow(img, cmap='gray', extent=extent, origin='lower', interpolation='bilinear')

# Select a subset of embeddings to overlay (e.g., 500 random points)
subset_idx = np.random.choice(len(embeddings), size=500, replace=False)
sc = ax.scatter(
    embeddings[subset_idx, 0],
    embeddings[subset_idx, 1],
    c=labels[subset_idx],
    cmap='tab10',
    s=10,
    alpha=0.6
)

plt.colorbar(sc, ax=ax, label='Label Pakaian (0–9)')

ax.set_title("Decoded Image Grid with Sparse Overlay")
ax.set_xlabel("Dimensi 1")
ax.set_ylabel("Dimensi 2")
ax.set_xlim(-10.5, 8)
ax.set_ylim(-3, 9)
plt.grid(False)
plt.tight_layout()
plt.show()