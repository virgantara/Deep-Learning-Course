import torch
from pathlib import Path
from torch.utils.data import Dataset
import re
import os
import cv2

def pad_or_truncate(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def encode(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]

class HMDBDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, frames_per_video=16):
        self.samples = []
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.frames_per_video = frames_per_video

        for cls in classes:
            class_path = os.path.join(root_dir, cls)
            for file in os.listdir(class_path):
                if file.endswith(".avi"):
                    self.samples.append((os.path.join(class_path, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_video_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        video_tensor = torch.stack(frames)  # shape: (T, C, H, W)
        return video_tensor, label

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        cap.release()
        # Padding or truncating
        if len(frames) < self.frames_per_video:
            frames += [frames[-1]] * (self.frames_per_video - len(frames))
        else:
            frames = frames[:self.frames_per_video]
        return frames



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