import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class JenisKelaminDataset(Dataset):

	def __init__(self, csv_file_path):
		self.data = pd.read_csv(csv_file_path)
		self.le = LabelEncoder()

		self.data['label'] = self.le.fit_transform(self.data['Jenis_Kelamin'])

		self.data['TB'] = (self.data['TB'] - self.data['TB'].min()) / (self.data['TB'].max() - self.data['TB'].min())
		self.data['BB'] = (self.data['BB'] - self.data['BB'].min()) / (self.data['BB'].max() - self.data['BB'].min())



	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		x = torch.tensor([self.data.iloc[idx]['TB'], self.data.iloc[idx]['BB']], dtype=torch.float32)
		y = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)

		return x, y