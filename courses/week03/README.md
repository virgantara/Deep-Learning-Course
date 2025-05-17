# Konsep Convolutional Neural Network (CNN)
## Import library

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset


import os
import struct
import numpy as np

from tqdm import tqdm
```

## Download Dataset
Silakan unduh dataset dari [sini](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## fungsi untuk unpack dataset mentah
```python
def read_idx(filename):
    """Read IDX file format"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
```


## Arsitektur CNN: Convolution, Pooling, Fully Connected Layer
Tabel 3.1 apabila diterjemahkan ke PyTorch, jadinya seperti ini:

```python
class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size =5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

```

## Augmentasi Data
Merujuk pada Subbab 3.3.2.3 Augmentasi Data
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(15), # Contoh Augmentasi tipe rotasi
    transforms.RandomCrop(28, padding=4), # Contoh Augmentasi tipe random croop
    transforms.Normalize((0.5,), (0.5,)) # merujuk pada subbab 3.3.2.1 tentang Normalisasi data
])

test_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])
```

## load dataset
```python
data_dir = 'path/to/data'

train_images = read_idx(os.path.join(data_dir, 'train-images.idx3-ubyte'))
train_labels = read_idx(os.path.join(data_dir, 'train-labels.idx1-ubyte'))

test_images = read_idx(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
test_labels = read_idx(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))

print(f"Train images shape: {train_images.shape}")  # (60000, 28, 28)
print(f"Train labels shape: {train_labels.shape}")  # (60000,)

print(f"Test images shape: {test_images.shape}")  # (10000, 28, 28)
print(f"Test labels shape: {test_labels.shape}")  # (10000,)
```

## Custom Dataset for PyTorch
```python
class CustomMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32)
        image = image / 255.0
        image = image.unsqueeze(0)  # (1, 28, 28)

        if self.transform:
            image = self.transform(image)

        return image, label
```

## Preloader dataset dari PyTorch
```python
train_dataset = CustomMNISTDataset(train_images, train_labels, transform=train_transform)
test_dataset = CustomMNISTDataset(test_images, test_labels, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
```
## Device configuration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Hyperparameters
```python
batch_size = 64
learning_rate = 0.001
num_epochs = 5
```

## Persiapan training
```python
model = ModelCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

## Training
```python
for epoch in tqdm(range(num_epochs)):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

## Evaluasi model
```python
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report
)
import matplotlib.pyplot as plt

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4,5,6,7,8,9])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))


FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FPR = FP / (FP + TN)
TPR = TP / (TP + FN)

for i in range(len(TP)):
    print(f"Kelas {i}: FPR = {FPR[i]:.4f}, TPR (Recall) = {TPR[i]:.4f}")
```

## Simpan model hanya bobot saja
```python
torch.save(model.state_dict(), 'model_cnn_mnist.pth')
# model.load_state_dict(torch.load('cnn_mnist.pth'))
```

## Load model hanya bobot saja
```python
model.load_state_dict(torch.load('cnn_mnist.pth'))
```

## Simpan model semua
```python
torch.save(model, 'cnn_mnist_full.pth')
```

## Load model semua
```python
model = torch.load('cnn_mnist_full.pth')
```

## Contoh load model untuk prediksi

```python
from PIL import Image

model = ModelCNN()
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((28, 28)),                  
    transforms.ToTensor(),                        
    transforms.Normalize((0.5,), (0.5,))          
])

image_path = 'path/to/sample_input.jpg'  
image = Image.open(image_path)
image = transform(image)               
image = image.unsqueeze(0)             

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print(f'Predicted Class: {predicted.item()}')
```