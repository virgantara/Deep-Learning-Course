import torch
import numpy as np
import matplotlib.pyplot as plt
from model_celeb import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("model_celeba_VAE.pth", map_location=device,weights_only=True))
model.eval()

grid_width, grid_height = 10, 3
z_dim = 200 
num_samples = grid_width * grid_height
shape_before_flattening = (128, 2, 2)  # dari encoder

z_sample = torch.randn(num_samples, z_dim).to(device)

with torch.no_grad():
    generated = model.decoder(z_sample, shape_before_flattening)  

fig = plt.figure(figsize=(18, 5))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(num_samples):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    img = generated[i].cpu().permute(1, 2, 0).numpy()  
    ax.imshow(img)
    ax.axis("off")

plt.suptitle("Hasil Wajah Baru dari Ruang Laten", fontsize=16)
plt.tight_layout()
plt.show()
