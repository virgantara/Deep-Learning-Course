import pandas as pd
import numpy as np

# Path to your CSV
csv_path = "data/fashion-mnist/fashion-mnist_train.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Extract image pixels (skip label in column 0)
pixel_values = df.iloc[:, 1:].values.astype(np.uint8)  # shape: (60000, 784)

# Reshape to (N, 28, 28)
images = pixel_values.reshape(-1, 28, 28)

# Save as images.npy
np.save("images.npy", images)
print("Saved images.npy with shape:", images.shape)
