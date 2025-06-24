import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_generated_images(epoch, generator, latent_dim, examples=10, device='cpu'):
    generator.eval()
    z = torch.randn(examples, latent_dim).to(device)

    generated_images = generator(z).detach().cpu()

    generated_images = generated_images * 0.5 + 0.5
    
    fig, axes = plt.subplots(1, examples, figsize=(15, 2))
    for i in range(examples):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"{path_to_dir}/generated_images/generated_epoch_{epoch}.png")
    plt.close()

def loss_d(real_pred, fake_pred):
        
    real_labels = torch.ones_like(real_pred)
    fake_labels = torch.zeros_like(fake_pred)

    noisy_real_labels = real_labels - (0.1 * torch.rand(real_pred.shape)).to(device)
    noisy_fake_labels = fake_labels + (0.1 * torch.rand(fake_pred.shape)).to(device)

    loss_real_labels = F.binary_cross_entropy(real_pred, noisy_real_labels)
    loss_fake_labels = F.binary_cross_entropy(fake_pred, noisy_fake_labels)

    return (loss_real_labels + loss_fake_labels)

def loss_g(fake_pred):
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred).to(device))