import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import NewsGroupDataset
from model import RNNClassifier

# -------------------------------
# 1. Set random seed for reproducibility
# -------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
# 2. Load tokenized dataset
# -------------------------------
dataset = NewsGroupDataset("data/20_newsgroups", tokenizer=str.split)

# -------------------------------
# 3. Build vocabulary manually
# -------------------------------
def build_vocab(dataset, min_freq=1, specials=["<pad>", "<unk>"]):
    counter = Counter()
    for tokens, _ in dataset:
        counter.update(tokens)

    vocab_tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    vocab = specials + sorted(vocab_tokens)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    return token_to_idx

vocab = build_vocab(dataset)
vocab["<unk>"] = vocab.get("<unk>", 1)  # Ensure <unk> exists
vocab["<pad>"] = vocab.get("<pad>", 0)  # Ensure <pad> exists

# -------------------------------
# 4. Collate function
# -------------------------------
def collate_fn(batch):
    texts, labels = zip(*batch)
    token_ids = [torch.tensor([vocab.get(token, vocab["<unk>"]) for token in tokens], dtype=torch.long) for tokens in texts]
    padded = pad_sequence(token_ids, batch_first=True, padding_value=vocab["<pad>"])
    return padded, torch.tensor(labels)

# -------------------------------
# 5. DataLoaders (train/test split)
# -------------------------------
train_len = int(len(dataset) * 0.8)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# 6. Model, Loss, Optimizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNClassifier(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=256,
    num_classes=20  # assuming 20 classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# 7. Training Loop
# -------------------------------
for epoch in range(5):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
