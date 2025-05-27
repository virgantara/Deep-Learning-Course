import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from collections import Counter
from dataset import IMDBCSVDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import SentimentClassifier

# ========= Hyperparams ==========
VOCAB_SIZE = 10000
SEQ_LEN = 20
EMBED_DIM = 8
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Preprocessing ==========
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def build_vocab(texts, max_vocab=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    most_common = counter.most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})
    return vocab





# ========= Training Loop ==========
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((out > 0.5).long() == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            out = model(x).squeeze()
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += ((out > 0.5).long() == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ========= Main ==========
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("data/imdb.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()

    print("Building vocab...")
    vocab = build_vocab(texts)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = IMDBCSVDataset(X_train, y_train, vocab, SEQ_LEN)
    test_dataset = IMDBCSVDataset(X_test, y_test, vocab, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = SentimentClassifier(len(vocab), EMBED_DIM, SEQ_LEN).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2%}")
