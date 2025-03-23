# Clustering dan Generative Models 
##  K-Means, DBSCAN, dan Hierarchical Clustering 

### K-Means
1. Import Library yang Diperlukan

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

2. Membuat Data Sederhana
Kita akan membuat data sederhana untuk klasterisasi.

```python
np.random.seed(0)
data = np.random.rand(100, 2)  # 100 on 2D Data
```

3. Inisialisasi Centroid
Kita perlu menginisialisasi centroid untuk kluster.

```python
k = 3  # Number of clusters
centroids = data[np.random.choice(data.shape[0], k, replace=False)]
```

4. Fungsi untuk Menghitung Jarak
Kita perlu menghitung jarak antara data dan centroid.

```python
# We need to calculate the distance between the data and the centroid.
def compute_distances(data, centroids):
    distances = torch.cdist(torch.tensor(data, dtype=torch.float32), 
                             torch.tensor(centroids, dtype=torch.float32))
    return distances
```

5. Algoritma K-Means
Implementasi algoritma K-Means.

```python
# K-Means algorithm
def k_means(data, k, max_iters=100):
    # Inisialisasi centroid
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Calculate the distance
        distances = compute_distances(data, centroids)
        
        # Define cluster
        labels = torch.argmin(distances, dim=1)
        
        # Update centroid
        for i in range(k):
            centroids[i] = data[labels == i].mean(axis=0)
    
    return centroids, labels
```
6. Menjalankan K-Means
Sekarang kita dapat menjalankan algoritma K-Means pada data kita.

```python
centroids, labels = k_means(data, k)
```

7. Visualisasi Hasil Klasterisasi
Kita dapat memvisualisasikan hasil klasterisasi.

```python
plt.scatter(data[:, 0], data[:, 1], c=labels.numpy(), cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

![K-Means Clustering](k-means-result.png)

##  Generative Adversarial Networks (GANs)
##  Implementasi Clustering dan GANs 