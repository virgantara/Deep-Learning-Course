import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class JenisKelaminClassifier(nn.Module):

	def __init__(self, hidden_size = 8, output_size=2):

		super(JenisKelaminClassifier,self).__init__()

		self.fc1 = nn.Linear(2, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x