import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),  
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].values
        label = int(row[0])
        image = row[1:].astype(np.uint8).reshape(28, 28) 
        image = np.expand_dims(image, axis=2)  
        image = self.transform(image)

        return image, image, label 


if __name__ == '__main__':
    dataset = FashionMNISTDataset('./data/fashion-mnist/fashion-mnist_train.csv')
    x, y = dataset[0]
    print(x.shape)  # should be torch.Size([1, 32, 32])