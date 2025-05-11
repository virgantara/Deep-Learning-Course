import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import glob
from PIL import Image
from tqdm import tqdm

class KucingAnjingDataset(Dataset):

	def __init__(self, root_dir, random_seed=42, image_size=224):

		self.root_dir = root_dir
		self.image_size = image_size

		if not os.path.exists(self.root_dir):
			raise RuntimeError(f"Dataset not found at {self.root_dir}.")


		self.transform = transforms.Compose([
        	transforms.Resize((self.image_size, self.image_size)),
        	transforms.ToTensor(),
        	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

		self.data = []
		self.labels = []

		for label, class_name in tqdm(enumerate(['cats','dogs'])):
			class_dir = os.path.join(self.root_dir, class_name)
			image_paths = glob.glob(os.path.join(class_dir,'*.jpg'))
			self.data.extend(image_paths)
			self.labels.extend([label] * len(image_paths))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image_path = self.data[idx]
		label = self.labels[idx]
		image = Image.open(image_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, label

