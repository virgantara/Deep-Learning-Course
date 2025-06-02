import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import Autoencoder, VAE
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'AE':
        model = Autoencoder().to(device)
    elif args.model_name == 'VAE':
        model = VAE().to(device)

    model.load_state_dict(torch.load("model_"+args.model_name+".pth", map_location=device, weights_only=True))
    model.eval()

    embeddings_np = np.load("embeddings_"+args.model_name+".npy")
    mins, maxs = np.min(embeddings_np, axis=0), np.max(embeddings_np, axis=0)
    samples = np.random.uniform(mins, maxs, size=(18, 2)).astype(np.float32)
    samples_tensor = torch.tensor(samples).to(device)

    with torch.no_grad():
        reconstructions = model.decoder(samples_tensor).cpu()

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(3, 9)  # 3 baris, 9 kolom

    # Scatter plot di posisi kiri
    ax_scatter = fig.add_subplot(gs[:, :3])  # seluruh baris, kolom 0–2
    ax_scatter.scatter(embeddings_np[:, 0], embeddings_np[:, 1], c='black', s=1, alpha=0.3)
    ax_scatter.scatter(samples[:, 0], samples[:, 1], c='deepskyblue', s=48)
    ax_scatter.set_title("Latent Space + Sampled Points", fontsize=16)
    ax_scatter.set_xlabel("Dimensi 1", fontsize=14)
    ax_scatter.set_ylabel("Dimensi 2", fontsize=14)
    ax_scatter.tick_params(axis='both', labelsize=14) 

    # Grid 3x6 hasil rekonstruksi di kanan (kolom 3–8)
    for i in range(18):
        row, col = divmod(i, 6)
        ax_img = fig.add_subplot(gs[row, 3 + col])
        ax_img.imshow(reconstructions[i].squeeze(), cmap='gray')
        ax_img.set_title(f"[{samples[i][0]:.1f}, {samples[i][1]:.1f}]", fontsize=14)
        ax_img.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='AE', help='Model Name')
    args = parser.parse_args()
    main(args)