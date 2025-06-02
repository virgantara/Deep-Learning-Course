import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import Autoencoder, VAE

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'AE':
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
        embeddings = np.load("embeddings.npy")
        labels = np.load("labels.npy")
    elif args.model_name == 'VAE':
        model = VAE().to(device)
        model.load_state_dict(torch.load("model_vae.pth", map_location=device, weights_only=True))
        embeddings = np.load("embeddings_vae.npy")
        labels = np.load("labels_vae.npy")


    model.eval()


    # Define grid points based on latent space
    x = np.linspace(np.min(embeddings[:, 0]), np.max(embeddings[:, 0]), 20)
    y = np.linspace(np.min(embeddings[:, 1]), np.max(embeddings[:, 1]), 20)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Decode grid
    with torch.no_grad():
        latent = torch.tensor(grid_points, dtype=torch.float32).to(device)
        reconstructions = model.decoder(latent).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    cell_w = (x[1] - x[0])
    cell_h = (y[1] - y[0])

    gap = 0.5  

    for i, (gx, gy) in enumerate(grid_points):
        img = reconstructions[i].squeeze()
        extent = [
            gx - (cell_w * gap) / 2,
            gx + (cell_w * gap) / 2,
            gy - (cell_h * gap) / 2,
            gy + (cell_h * gap) / 2,
        ]
        ax.imshow(img, cmap='gray', extent=extent, origin='lower', interpolation='bilinear', alpha=0.7)


    # Overlay colored latent embeddings
    subset_idx = np.random.choice(len(embeddings), size=500, replace=False)
    scatter = ax.scatter(
        embeddings[subset_idx, 0],
        embeddings[subset_idx, 1],
        c=labels[subset_idx],
        cmap='Spectral',
        s=128,
        alpha=1
    )

    # Custom inset colorbar, matching latent space height
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # cax = inset_axes(ax, width="3%", height="100%", loc='right', borderpad=2)
    # cbar = fig.colorbar(scatter, cax=cax)
    # cbar.set_label("FashionMNIST Class", fontsize=12)
    # cbar.ax.tick_params(labelsize=10)

    # Format axes
    ax.set_title("Latent Space Grid with Decoded Reconstructions", fontsize=14)
    ax.set_xlabel("Latent Dimension 1", fontsize=12)
    ax.set_ylabel("Latent Dimension 2", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_aspect("equal")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='AE', help='Model Name')
    args = parser.parse_args()
    main(args)