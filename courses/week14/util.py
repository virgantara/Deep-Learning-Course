import unicodedata
import re
import numpy as np
import random


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)

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