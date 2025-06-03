import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torchvision import transforms

from model_celeb import VAE
from dataset import CelebADataset

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((32, 32)),  # match your VAE input
    transforms.ToTensor(),        # values in [0, 1]
])

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, z_dim]
    kl_per_sample = kl_elementwise.sum(dim=1)  # [batch]
    kl_loss = kl_per_sample.sum()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def main(args):
    train_dataset = CelebADataset(root_dir='./data/celeba/Img',transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VAE().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = args.epochs

    train_loss, recon_losses, kl_losses = [], [], []
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Total Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {args.model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'train_loss_{args.model_name}.png')  # Save to file
    torch.save(model.state_dict(), 'model_celeba_'+args.model_name+'.pth')
    plt.show()

    

def show_reconstruction(args):
    train_dataset = CelebADataset(root_dir='./data/celeba/Img',transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    
    model.load_state_dict(torch.load("model_celeba_"+args.model_name+".pth", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_imgs = next(iter(train_loader))
        test_imgs = test_imgs.to(device)
        
        outputs, _, _ = model(test_imgs)

        # Show original and reconstructed
        n = 10
        plt.figure(figsize=(20, 4))
        for i in tqdm(range(n),desc="showing reconstruction"):
            # original
            ax = plt.subplot(2, n, i + 1)
            img = test_imgs[i].cpu().permute(1, 2, 0).numpy()  # CHW → HWC
            plt.imshow(img)
            ax.axis("off")
            
            # reconstructed
            ax = plt.subplot(2, n, i + 1 + n)
            recon = outputs[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(recon)
            ax.axis("off")
            
        plt.suptitle(f"Reconstruction Samples - {args.model_name}", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VAE', help='Model Name')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epoch')
    args = parser.parse_args()
    main(args)
    # show_reconstruction(args)