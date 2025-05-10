import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from dataset import JenisKelaminDataset
from model import JenisKelaminClassifier
from torch.utils.data import DataLoader

dataset = JenisKelaminDataset('jenis_kelamin.csv')
train_set, test_set = random_split(dataset, [8, 2])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2)

model = JenisKelaminClassifier(8, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), 'jenis_kelamin_model.pth')
print("Model saved")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Accuracy on test set: {100 * correct / total:.2f}%")