import matplotlib.pyplot as plt
from collections import Counter
from util import *
import string

def plot_top_words_vertical(
    pairs,
    lang1_name="English",
    lang2_name="Indonesian",
    top_n=20,
    remove_punct=True,
    fontsize=20
):
    """
    pairs: list of (lang1_tokens, lang2_tokens)
    """

    # Flatten token list per language
    lang1_tokens = [tok for src, _ in pairs for tok in src]
    lang2_tokens = [tok for _, tgt in pairs for tok in tgt]

    if remove_punct:
        # buang token yang hanya tanda baca/angka (opsional)
        punct = set(string.punctuation)
        lang1_tokens = [t for t in lang1_tokens if any(ch.isalpha() for ch in t) and t not in punct]
        lang2_tokens = [t for t in lang2_tokens if any(ch.isalpha() for ch in t) and t not in punct]

    # Hitung frekuensi
    c1 = Counter(lang1_tokens).most_common(top_n)
    c2 = Counter(lang2_tokens).most_common(top_n)

    words1, counts1 = zip(*c1) if c1 else ([], [])
    words2, counts2 = zip(*c2) if c2 else ([], [])

    # Plot 2 bar chart horizontal secara VERTIKAL (2x1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharex=False)

    # --- Language 1 (atas) ---
    axes[0].barh(words1, counts1)
    axes[0].invert_yaxis()  # frekuensi tertinggi di paling atas
    axes[0].set_title(f"Top {top_n} kata-kata bahasa {lang1_name}", fontsize=fontsize+2)
    axes[0].set_xlabel("Jumlah",fontsize=fontsize)
    axes[0].tick_params(axis='both', labelsize=fontsize)

    # --- Language 2 (bawah) ---
    axes[1].barh(words2, counts2)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top {top_n} kata-kata bahasa {lang2_name}", fontsize=fontsize+2)
    axes[1].set_xlabel("Jumlah",fontsize=fontsize)
    axes[1].tick_params(axis='both', labelsize=fontsize)

    plt.tight_layout()
    plt.show()

# Example usage
pairs = load_pairs("data/ind-eng/ind.txt", max_len=20)
plot_top_words_vertical(pairs, lang1_name="Inggris", lang2_name="Indonesia", top_n=20, fontsize=20)
