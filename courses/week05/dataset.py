import torch
from pathlib import Path
from torch.utils.data import Dataset
import re


def pad_or_truncate(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def encode(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]



class IMDBCSVDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len):
        self.data = [(pad_or_truncate(encode(text, vocab), seq_len), label) for text, label in zip(texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


class NewsGroupDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.tokenizer = tokenizer

        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            class_folder = self.root_dir / cls_name
            for file in class_folder.iterdir():
                if file.is_file():
                    self.samples.append((file, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        with open(filepath, 'r', encoding='latin1') as f:
            text = f.read()

        if self.transform:
            text = self.transform(text)
        if self.tokenizer:
            text = self.tokenizer(text)

        return text, label