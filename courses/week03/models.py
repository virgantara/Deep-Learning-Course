import torch
import torch.nn as nn
import torch.nn.functional as F

class KucingAnjingClassifier(nn.Module):

	def __init__(self, num_classes=2):
		super(KucingAnjingClassifier, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=1)
		self.pool = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=2, padding=1)

		self.fc1 = nn.Linear(64 * 28 * 28, 128)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.pool(x)

		x = self.conv2(x)
		x = F.relu(x)
		x = self.pool(x)

		x = self.conv3(x)
		x = F.relu(x)
		x = self.pool(x)

		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = F.relu(x)

		self.fc2(x)

		return x