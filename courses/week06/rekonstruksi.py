import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Autoencoder
from dataset import FashionMNISTDataset
from tqdm import tqdm

train_dataset = FashionMNISTDataset('./data/fashion-mnist/fashion-mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()
with torch.no_grad():
    test_imgs, _,_ = next(iter(train_loader))
    test_imgs = test_imgs.to(device)
    outputs = model(test_imgs)

    # Show original and reconstructed
    n = 10
    plt.figure(figsize=(20, 4))
    for i in tqdm(range(n),desc="Hasil rekonstruksi"):
        # original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_imgs[i].cpu().squeeze(), cmap='gray')
        ax.axis("off")
        
        # reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(outputs[i].cpu().squeeze(), cmap='gray')
        ax.axis("off")
    plt.show()