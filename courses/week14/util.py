import unicodedata
import re
import numpy as np
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)


def decode_ids(ids, itos, src=None, src_itos=None, return_tokens=False):
    tokens = []
    for i, tok_id in enumerate(ids):
        tok = tok_id.item()
        if tok == EOS:
            break
        if tok == PAD or tok == BOS:
            continue
        if tok == UNK and src is not None and src_itos is not None:
            if i < len(src):
                tokens.append(src_itos.get(src[i].item(), "<src-unk>"))
            else:
                tokens.append("<unk>")
        else:
            tokens.append(itos.get(tok, "<unk>"))
    return tokens if return_tokens else " ".join(tokens)

def evaluate_bleu(model, loader, src_itos, trg_itos, src_vocab=None, max_len=40):
    """
    Menghitung BLEU score untuk data validasi/test.
    
    Args:
        model      : Trained seq2seq model (BahdanauSeq2Seq)
        loader     : DataLoader untuk validasi/test
        src_itos   : index-to-token untuk bahasa sumber
        trg_itos   : index-to-token untuk bahasa target
        src_vocab  : (opsional) vocab sumber, hanya digunakan jika ada copy <unk>
        max_len    : panjang maksimal prediksi

    Returns:
        bleu_score : BLEU score dalam skala 0–100
    """
    model.eval()
    references = []
    hypotheses = []

    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for src, trg in loader:
            src = src.to(model.device)
            trg = trg.to(model.device)
            pred_ids, _ = model.greedy_decode(src, max_len=max_len)  # [Tpred, B]

            B = src.size(1)
            for b in range(B):
                ref = decode_ids(trg[:, b], trg_itos, return_tokens=True)
                hyp = decode_ids(pred_ids[:, b], trg_itos, 
                                 src[:, b] if src_vocab else None, 
                                 src_itos if src_vocab else None, 
                                 return_tokens=True)
                
                ref = [w for w in ref if w not in {'<pad>', '<bos>', '<eos>'}]
                hyp = [w for w in hyp if w not in {'<pad>', '<bos>', '<eos>'}]
                
                references.append([ref])
                hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100
    return bleu


def save_vocab(vocab, path):
    import json
    with open(path, "w") as f:
        json.dump(vocab, f)

def load_vocab(path):
    import json
    with open(path) as f:
        vocab = json.load(f)
    vocab = {k: int(v) for k, v in vocab.items()}
    itos = {v: k for k, v in vocab.items()}
    return vocab, itos

def load_pairs(path, max_len=20, max_pairs=None):
	pairs = []

	with open(path, encoding="utf-8") as f:
		for i, line in enumerate(f):
			cols = line.rstrip("\n").split("\t")

			if len(cols) < 2:
				continue

			src, tgt = cols[0], cols[1]
			src_token = normalize_and_tokenize(src)
			tgt_token = normalize_and_tokenize(tgt)

			if 1 <= len(src_token) <= max_len and 1 <= len(tgt_token) <= max_len:
				pairs.append((src_token, tgt_token))

			if max_pairs and len(pairs) >= max_pairs:
				break

	return pairs

def to_ascii(s: str) -> str:
	return (
		unicodedata.normalize("NFKD",s)
		.encode("ascii","ignore")
		.decode("ascii")
	)



def normalize_and_tokenize(s: str):
	s = to_ascii(s).lower().strip()

	s = re.sub(r"\s+"," ",s)
	_token_re = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
	return _token_re.findall(s)

def to_ids(tokens, vocab, unk_log=None):
    ids = [BOS]
    for tok in tokens:
        tok_id = vocab.get(tok, UNK)
        if tok_id == UNK and unk_log is not None:
            unk_log.append(tok)
        ids.append(tok_id)
    ids.append(EOS)
    return ids

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