import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VAE
from dataset import FashionMNISTDataset
from tqdm import tqdm

train_dataset = FashionMNISTDataset('./data/fashion-mnist/fashion-mnist_train.csv')
test_dataset = FashionMNISTDataset('./data/fashion-mnist/fashion-mnist_test.csv')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss  + beta * kl_loss, recon_loss, kl_loss

epochs = 20
train_loss, recon_losses, kl_losses = [], [], []

for epoch in range(epochs):
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs = imgs.to(device)
        
        recon, mu, logvar = model(imgs)
        loss, recon_loss, kl_loss = vae_loss(recon, imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    train_loss.append(total_loss / len(train_loader))
    recon_losses.append(total_recon / len(train_loader))
    kl_losses.append(total_kl / len(train_loader))

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.2f} | Recon: {total_recon/len(train_loader):.2f} | KL: {total_kl/len(train_loader):.2f}")

plt.plot(train_loss, label='Total Loss')
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Loss Components")
plt.legend()
plt.show()


torch.save(model.state_dict(), 'model_vae.pth')

def show_reconstruction():
    model.eval()
    with torch.no_grad():
        test_imgs, _, _ = next(iter(train_loader))
        test_imgs = test_imgs.to(device)
        recon_imgs, _, _ = model(test_imgs)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in tqdm(range(n), desc="showing reconstruction"):
            # original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(test_imgs[i].cpu().squeeze(), cmap='gray')
            ax.axis("off")
            # reconstructed
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_imgs[i].cpu().squeeze(), cmap='gray')
            ax.axis("off")
        plt.show()

show_reconstruction()
