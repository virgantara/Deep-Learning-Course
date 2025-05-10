# Struktur MLP

## Dataset Klasifikasi Jenis Kelamin

Dataset ini berisi informasi sederhana yang dapat digunakan untuk klasifikasi jenis kelamin berdasarkan **tinggi badan (TB)** dan **berat badan (BB)**. Dataset cocok untuk eksperimen machine learning dasar seperti regresi logistik, decision tree, atau KNN.

## Struktur Dataset

| Kolom         | Deskripsi                            |
|---------------|---------------------------------------|
| TB            | Tinggi badan dalam satuan centimeter |
| BB            | Berat badan dalam satuan kilogram     |
| Jenis_Kelamin | Target kelas: `Laki-laki` atau `Perempuan` |

## Contoh Data

| TB  | BB | Jenis_Kelamin |
|-----|----|----------------|
| 172 | 70 | Laki-laki      |
| 158 | 52 | Perempuan      |
| 180 | 82 | Laki-laki      |
| 165 | 55 | Perempuan      |
| 177 | 76 | Laki-laki      |

## Format File

Dataset tersedia dalam format CSV:

```csv
TB,BB,Jenis_Kelamin
172,70,Laki-laki
158,52,Perempuan
```

## Building an MLP with PyTorch

1. Import library yang diperlukan

Untuk memulai, impor modul PyTorch yang diperlukan:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

`torch` - Library PyTorch utama.
`torch.nn` - Berisi blok-blok bangunan untuk jaringan syaraf.
`torch.optim` - Menyediakan algoritma pengoptimalan seperti SGD dan Adam.
`torch.nn.functional` - Menyediakan berbagai fungsi aktivasi dan fungsi kerugian.

2. Mendefinisikan Model MLP
Di PyTorch, kita mendefinisikan MLP menggunakan kelas `nn.Module`:

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for output (assumes regression task)
        return x

```

`nn.Linear()` membuat fully-connected layer.
`F.relu()` menerapkan fungsi aktivasi ReLU untuk memperkenalkan non-linearitas.

3. Menginisialisasi Model, Fungsi Rugi, dan Pengoptimal
Sebelum melatih, tentukan model, fungsi loss, dan optimizer:
```python
input_size = 10  
hidden_size = 32  
output_size = 1  
model = MLP(input_size, hidden_size, output_size)

criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  
```

`MSELoss()` digunakan untuk masalah regresi.
`Adam()` adalah optimizer adaptif untuk memperbarui parameter model.

4. Melatih MLP
Untuk melatih MLP, feedforward dan backpropagation:
```python
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  
    inputs = torch.randn(64, input_size) 
    targets = torch.randn(64, output_size) 

    outputs = model(inputs)  
    loss = criterion(outputs, targets)  
    loss.backward()  
    optimizer.step() 

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

```

- Model ini memprediksi keluaran, menghitung kerugian, dan memperbarui bobot menggunakan backpropagation.
- Every 10 epochs, the loss is printed for monitoring.

