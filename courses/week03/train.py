import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MNISTDataset
from models import MnistClassifier
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def main(args):

	train_dataset = MNISTDataset(
	    images_path='data/mnist/train-images.idx3-ubyte',
	    labels_path='data/mnist/train-labels.idx1-ubyte'
	)

	test_dataset = MNISTDataset(
	    images_path='data/mnist/t10k-images.idx3-ubyte',
	    labels_path='data/mnist/t10k-labels.idx1-ubyte'
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_classes = args.num_classes
	model = MnistClassifier(num_classes)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Total parameters: {total_params / 1e6:.2f}M")
	print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

	train_losses = []
	test_losses = []

	for epoch in range(args.epochs):
		model.train()
		train_loss = 0

		for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
			data, labels = data.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(data)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
		
		train_loss += loss.item() * data.size(0)

		model.eval()
		correct = 0
		total = 0
		test_loss = 0

		with torch.no_grad():
			for data, labels in test_loader:
				data, labels = data.to(device), labels.to(device)
				outputs = model(data)
				loss = criterion(outputs, labels)
				test_loss += loss.item() * data.size(0)
				_, preds = torch.max(outputs, 1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)

		acc = correct / total
		avg_train_loss = train_loss / len(train_dataset)
		avg_test_loss = test_loss / len(test_dataset)

		train_losses.append(avg_train_loss)
		test_losses.append(avg_test_loss)

		print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Test Loss {avg_test_loss:.4f}, Test Acc {acc:.4f}")

	torch.save({
	    'epoch': args.epochs,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	    'train_loss': train_losses,
	    'test_loss': test_losses,
	}, args.checkpoint)

	epochs_range = range(1, args.epochs + 1)
	plt.figure(figsize=(10, 5))
	plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
	plt.plot(epochs_range, test_losses, label='Test Loss', marker='x')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Train vs Test Loss per Epoch')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir', type=str, default='data/catndog/train', help='Path to training data')
	parser.add_argument('--test_dir', type=str, default='data/catndog/test', help='Path to test data')
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
	parser.add_argument('--checkpoint', type=str, default='cat_dog_checkpoint.pth', help='Path to save model checkpoint')
	args = parser.parse_args()
	main(args)
