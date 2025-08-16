import re
import random
import unicodedata
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from util import *

random.seed(7)





SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)

def build_vocab(token_lists, min_freq=1, max_size=None):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    most_common = counter.most_common()
    if max_size:
        most_common = most_common[: max(0, max_size - len(SPECIALS))]
    vocab = {w: i + len(SPECIALS) for i, (w, c) in enumerate(most_common) if c >= min_freq}
    for i, sp in enumerate(SPECIALS):
        vocab[sp] = i
    itos = {i: w for w, i in vocab.items()}
    return vocab, itos

def to_ids(tokens, vocab):
    return [BOS] + [vocab.get(t, UNK) for t in tokens] + [EOS]

def pad_batch(batch, pad_id=PAD):
    max_len = max(len(x) for x in batch)
    return [seq + [pad_id] * (max_len - len(seq)) for seq in batch]

def split_pairs(pairs, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return train, val, test

def plot_length_histograms(pairs, title_suffix=""):
    en_lengths = [len(src) for src, _ in pairs]
    fr_lengths = [len(tgt) for _, tgt in pairs]

    # Choose nice bins up to, say, 20 tokens
    max_bin = max(20, max(en_lengths + fr_lengths))
    bins = [i + 0.5 for i in range(1, max_bin + 1)]  # center bars on integers

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].hist(en_lengths, bins=bins)
    axes[0].set_title("Panjang Kalimat Inggris" + title_suffix)
    axes[0].set_ylabel("# Kalimat")

    axes[1].hist(fr_lengths, bins=bins)
    axes[1].set_title("Panjang Kalimat Indonesia" + title_suffix)
    axes[1].set_xlabel("# Token pada Kalimat")
    axes[1].set_ylabel("# Kalimat")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to your file, e.g., data/eng-fra.txt from ManyThings
    data_file = Path("data/ind-eng/ind.txt")

    # 1) Load + preprocess + filter (e.g., <= 20 tokens)
    pairs = load_pairs(data_file, max_len=20, max_pairs=None)
    print(f"Total usable pairs after filtering: {len(pairs):,}")

    # 2) Visualize length distributions (before adding <bos>/<eos>)
    plot_length_histograms(pairs)

    # 3) Split 80/10/10
    train, val, test = split_pairs(pairs, 0.8, 0.1)
    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    # 4) Build separate vocabs (you can also build joint if you prefer)
    en_vocab, en_itos = build_vocab([src for src, _ in train])
    fr_vocab, fr_itos = build_vocab([tgt for _, tgt in train])
    print(f"EN vocab size: {len(en_vocab):,} | ID vocab size: {len(fr_vocab):,}")

    # 5) Numericalize with <bos>/<eos>
    train_en_ids = [to_ids(src, en_vocab) for src, _ in train]
    train_fr_ids = [to_ids(tgt, fr_vocab) for _, tgt in train]

    # Example: how you'd pad a batch before feeding a model
    example_batch_en = pad_batch(train_en_ids[:32], pad_id=PAD)
    example_batch_fr = pad_batch(train_fr_ids[:32], pad_id=PAD)
    print(f"Example batch shapes: EN {len(example_batch_en)} x {len(example_batch_en[0])}, "
          f"ID {len(example_batch_fr)} x {len(example_batch_fr[0])}")