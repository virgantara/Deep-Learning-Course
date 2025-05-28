import torch
import torch.nn as nn


class VideoRNNClassifier(nn.Module):
    def __init__(self, input_size=112*112*3, hidden_size=256, num_layers=1, num_classes=5):
        super(VideoRNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B, T, -1)  # Flatten frame: (B, T, C*H*W)
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        return out


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