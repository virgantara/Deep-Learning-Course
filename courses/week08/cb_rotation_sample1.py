from PIL import Image
import matplotlib.pyplot as plt

# Fungsi g(X|y): rotasi citra X dengan sudut y
def g(X, y):
    return X.rotate(y)

# Baca citra asli
X = Image.open("data/kambing.jpg")

# Daftar sudut rotasi
rotations = [0, 90, 180, 270]

# Terapkan transformasi rotasi
rotated_images = {y: g(X, y) for y in rotations}

# Tampilkan hasil
plt.figure(figsize=(16, 8))
for i, y in enumerate(rotations, 1):
    plt.subplot(1, 4, i)
    plt.imshow(rotated_images[y])
    plt.title(f"Rotasi {y}°",fontsize=24)
    plt.axis("off")
plt.show()
