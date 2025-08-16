import unicodedata
from collections import Counter
from pathlib import Path
from util import *
import torch

from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq




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

data_file = Path("data/ind-eng/ind.txt")

# 1) Load + preprocess + filter (e.g., <= 20 tokens)
pairs = load_pairs(data_file, max_len=20, max_pairs=None)
print(f"Total usable pairs after filtering: {len(pairs):,}")


# 3) Split 80/10/10
train, val, test = split_pairs(pairs, 0.8, 0.1)
print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

# 4) Build separate vocabs (you can also build joint if you prefer)
en_vocab, en_itos = build_vocab([src for src, _ in train])
id_vocab, id_itos = build_vocab([tgt for _, tgt in train])
print(f"EN vocab size: {len(en_vocab):,} | ID vocab size: {len(id_vocab):,}")

ENCODER_HIDDEN_SIZE = 512
DECODER_HIDDEN_SIZE = 2 * ENCODER_HIDDEN_SIZE

ENCODER_EMBEDDING_DIM, DECODER_HIDDEN_SIZE  = 256, 256

encoder = BahdanauEncoder(input_dim=len(en_vocab),
						embedding_dim = ENCODER_EMBEDDING_DIM,
						encoder_hidden_dim=ENCODER_HIDDEN_SIZE,
						decoder_hidden_dim = DECODER_HIDDEN_SIZE,
						dropout_p = 0.15)

attn = BahdanauAttentionQKV(DECODER_HIDDEN_SIZE)

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
        for src, trg in loader:
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

def decode_ids(ids, itos):
    # ids: [T]
    toks = []
    for i in ids:
        if i.item() == EOS:
            break
        if i.item() in (BOS, PAD):
            continue
        toks.append(itos.get(i.item(), "<unk>"))
    return " ".join(toks)

# -----------------------
# Train loop
# -----------------------
EPOCHS = 10
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    # optionally decay teacher forcing a bit
    tf = max(0.3, 0.7 - 0.04 * (epoch - 1))
    train_loss, train_ppl = epoch_run(seq2seq, train_loader, train=True,  teacher_forcing=tf)
    val_loss,   val_ppl   = epoch_run(seq2seq, val_loader,   train=False, teacher_forcing=0.0)

    print(f"Epoch {epoch:02d} | TF={tf:.2f} | "
          f"Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | "
          f"Val Loss {val_loss:.4f} PPL {val_ppl:.2f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(seq2seq.state_dict(), "bahdanau_best.pt")
        print("  -> saved best to bahdanau_best.pt")

# -----------------------
# Test + Sample decode
# -----------------------
seq2seq.load_state_dict(torch.load("bahdanau_best.pt", map_location=device))
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
            src_txt = decode_ids(src[:, b], en_itos)
            trg_txt = decode_ids(trg[:, b], id_itos)
            pred_txt = decode_ids(ys[:, b], id_itos)
            print("-" * 60)
            print("SRC :", src_txt)
            print("TRG :", trg_txt)
            print("PRED:", pred_txt)
        shown += B
        if shown >= n_show:
            break
