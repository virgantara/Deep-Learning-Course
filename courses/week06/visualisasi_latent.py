import torch
import matplotlib.pyplot as plt
from model import Autoencoder
from dataset import FashionMNISTDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("model.pth",weights_only=True))  # if saved model exists
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

test_dataset = FashionMNISTDataset('./data/fashion-mnist/fashion-mnist_test.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, _, labels in test_loader:
        imgs = imgs.to(device)
        z = model.encoder(imgs)  # output shape: [batch_size, 2]
        all_embeddings.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

embeddings = np.vstack(all_embeddings)
labels = np.concatenate(all_labels).squeeze()

np.save("embeddings.npy", embeddings)
np.save("labels.npy",labels)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='Label Kategori Pakaian (0-9)')
plt.title("Visualisasi Ruang Laten Autoencoder (2D)")
plt.xlabel("Dimensi 1")
plt.ylabel("Dimensi 2")
plt.grid(True)
plt.show()
