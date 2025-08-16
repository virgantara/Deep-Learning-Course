import unicodedata
import re

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