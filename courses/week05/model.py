import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        return self.fc(hidden[-1])  # use last hidden state


# ========= Model ==========
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embed_dim * seq_len, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        return torch.sigmoid(self.fc(x))