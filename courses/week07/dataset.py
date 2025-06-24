from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

class DatasetImages(Dataset):
    def __init__(self, path_to_images, transform=None):
        self.path_to_images = path_to_images # путь к папке с изображениями
        self.transform = transform 

    def __getitem__(self, idx):
        image = Image.open(self.path_to_images[idx]) # получить путь к изображению по индексу и считать его

        if self.transform:
            image = self.transform(image) # трансформация изображения при необходимости
        
        return image
    
    def __len__(self):
        return len(self.path_to_images) # количество поданных изображений

