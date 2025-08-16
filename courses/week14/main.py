import unicodedata
from collections import Counter
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import csv

from util import *
from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='data/catndog/train', help='Path to training data')
parser.add_argument('--test_dir', type=str, default='data/catndog/test', help='Path to test data')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--checkpoint', type=str, default='cat_dog_checkpoint.pth', help='Path to save model checkpoint')
args = parser.parse_args()


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

class NMTDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def collate_batch(batch):
    """
    batch: list of (src_ids[T1], trg_ids[T2]).
    Returns:
      src_pad: [Tsrc, B]
      trg_pad: [Ttrg, B]
      src_lens, trg_lens (optional if you need)
    """
    src_seqs, trg_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    trg_lens = [len(t) for t in trg_seqs]

    max_src = max(src_lens)
    max_trg = max(trg_lens)

    padded_src = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    padded_trg = torch.full((len(batch), max_trg), PAD, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, trg_seqs)):
        padded_src[i, : len(s)] = s
        padded_trg[i, : len(t)] = t

    # transpose to [T, B] for your encoder/decoder
    return padded_src.t().contiguous(), padded_trg.t().contiguous()

data_file = Path("data/ind-eng/ind.txt")

# 1) Load + preprocess + filter (e.g., <= 20 tokens)
pairs = load_pairs(data_file, max_len=20, max_pairs=None)
print(f"Total usable pairs after filtering: {len(pairs):,}")


# 3) Split 80/10/10

train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)
print(f"Train: {len(train_pairs):,}, Val: {len(val_pairs):,}, Test: {len(test_pairs):,}")

# 4) Build separate vocabs (you can also build joint if you prefer)
en_vocab, en_itos = build_vocab([src for src, _ in train_pairs], max_size=10000)
id_vocab, id_itos = build_vocab([tgt for _, tgt in train_pairs], max_size=10000)
print(f"EN vocab size: {len(en_vocab):,} | ID vocab size: {len(id_vocab):,}")

train_ds = NMTDataset(train_pairs, en_vocab, id_vocab)
val_ds   = NMTDataset(val_pairs,   en_vocab, id_vocab)
test_ds  = NMTDataset(test_pairs,  en_vocab, id_vocab)

en_unk_tokens = []
id_unk_tokens = []

train_ds.data = [(to_ids(src, en_vocab, en_unk_tokens), 
                  to_ids(trg, id_vocab, id_unk_tokens)) for src, trg in train_pairs]


print("\nTop unknown EN tokens:")
for tok, c in Counter(en_unk_tokens).most_common(20):
    print(f"  {tok:15}  {c}x")

print("\nTop unknown ID tokens:")
for tok, c in Counter(id_unk_tokens).most_common(20):
    print(f"  {tok:15}  {c}x")

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


ENCODER_HIDDEN_SIZE = 512
DECODER_HIDDEN_SIZE = 256

ENCODER_EMBEDDING_DIM, DECODER_HIDDEN_SIZE  = 256, 256

encoder = BahdanauEncoder(input_dim=len(en_vocab),
						embedding_dim = ENCODER_EMBEDDING_DIM,
						encoder_hidden_dim=ENCODER_HIDDEN_SIZE,
						decoder_hidden_dim = DECODER_HIDDEN_SIZE,
						dropout_p = 0.15)

attn = BahdanauAttentionQKV(
            hidden_size=DECODER_HIDDEN_SIZE,
            query_size=DECODER_HIDDEN_SIZE,
            key_size= 2 * ENCODER_HIDDEN_SIZE,
            dropout_p = 0.0)

decoder = BahdanauDecoder(output_dim=len(id_vocab), 
						embedding_dim=DECODER_HIDDEN_SIZE,
						encoder_hidden_dim = ENCODER_HIDDEN_SIZE,
						decoder_hidden_dim=DECODER_HIDDEN_SIZE,
						attention = attn,
						dropout_p = 0.15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq2seq = BahdanauSeq2Seq(encoder, decoder, device, pad_id=0, bos_id=1, eos_id=2).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=3e-4)
CLIP = 1.0



def epoch_run(model, loader, train=True, teacher_forcing=0.5):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, trg in tqdm(loader):
            src = src.to(device)  # [Tsrc, B]
            trg = trg.to(device)  # [Ttrg, B]

            # Forward: outputs = [Ttrg, B, V]
            outputs, _att = model(src, trg, teacher_forcing_ratio=teacher_forcing if train else 0.0)

            # Shift for CE: predict trg[1:] from inputs up to trg[:-1]
            logits = outputs[1:].reshape(-1, outputs.size(-1))      # [(Ttrg-1)*B, V]
            target = trg[1:].reshape(-1)                            # [(Ttrg-1)*B]

            loss = criterion(logits, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()

            n_tokens = (target != PAD).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl

def decode_ids(ids, itos, src=None, src_itos=None):
    tokens = []
    for i, tok_id in enumerate(ids):
        tok = tok_id.item()
        if tok == EOS:
            break
        if tok == PAD or tok == BOS:
            continue
        if tok == UNK and src is not None and src_itos is not None:
            # try to copy aligned source token
            if i < len(src):
                tokens.append(src_itos.get(src[i].item(), "<src-unk>"))
            else:
                tokens.append("<unk>")
        else:
            tokens.append(itos.get(tok, "<unk>"))
    return " ".join(tokens)

def plot_curves(history, save_prefix="bahdanau", fontsize=14):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"],   marker="o", label="Val")
    plt.title("Cross-Entropy Loss per Epoch", fontsize=fontsize+2)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=180)
    plt.show()

    # Perplexity
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_ppl"], marker="o", label="Train")
    plt.plot(epochs, history["val_ppl"],   marker="o", label="Val")
    plt.title("Perplexity (PPL) per Epoch", fontsize=fontsize+2)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("PPL", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_ppl.png", dpi=180)
    plt.show()

# -----------------------
# Train loop
# -----------------------


history = {
    "train_loss": [], "val_loss": [],
    "train_ppl":  [], "val_ppl":  []
}

EPOCHS = args.epochs
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    # optionally decay teacher forcing a bit
    tf = max(0.3, 0.7 - 0.04 * (epoch - 1))
    train_loss, train_ppl = epoch_run(seq2seq, train_loader, train=True,  teacher_forcing=tf)
    val_loss,   val_ppl   = epoch_run(seq2seq, val_loader,   train=False, teacher_forcing=0.0)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_ppl"].append(train_ppl)
    history["val_ppl"].append(val_ppl)

    print(f"Epoch {epoch:02d} | TF={tf:.2f} | "
          f"Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | "
          f"Val Loss {val_loss:.4f} PPL {val_ppl:.2f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(seq2seq.state_dict(), "bahdanau_best.pt")
        print("  -> saved best to bahdanau_best.pt")

with open("train_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch","train_loss","val_loss","train_ppl","val_ppl"])
    for i in range(EPOCHS):
        w.writerow([i+1, history["train_loss"][i], history["val_loss"][i],
                    history["train_ppl"][i], history["val_ppl"][i]])

# plot_curves(history, save_prefix="bahdanau", fontsize=14)
# -----------------------
# Test + Sample decode
# -----------------------
seq2seq.load_state_dict(torch.load("bahdanau_best.pt", weights_only=True, map_location=device))
test_loss, test_ppl = epoch_run(seq2seq, test_loader, train=False, teacher_forcing=0.0)
print(f"TEST  | Loss {test_loss:.4f} | PPL {test_ppl:.2f}")

# Show a few greedy decodes
seq2seq.eval()
with torch.no_grad():
    n_show = 5
    shown = 0
    for src, trg in test_loader:
        src = src.to(device)
        trg = trg.to(device)
        ys, _atts = seq2seq.greedy_decode(src, max_len=40)  # [Tout, B]
        B = src.size(1)
        for b in range(min(B, n_show - shown)):
            src_txt  = decode_ids(src[:, b], en_itos)
            trg_txt  = decode_ids(trg[:, b], id_itos)
            pred_txt = decode_ids(ys[:, b], id_itos, src[:, b], en_itos)
            pred_txt_ids = ys[:, b]
            unk_pos = (pred_txt_ids == UNK).nonzero(as_tuple=True)[0]
            
            print("-" * 60)
            print("SRC :", src_txt)
            print("TRG :", trg_txt)
            print("PRED:", pred_txt)
            print("UNK predicted at positions:", unk_pos.tolist())
        shown += B
        if shown >= n_show:
            break
