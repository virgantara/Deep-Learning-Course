import numpy as np
import matplotlib.pyplot as plt
from util import *

def plot_joint_length_heatmap(pairs, max_len=20, title="Joint Distribution of Sentence Lengths"):
    """
    pairs: list of (en_tokens, id_tokens)
    max_len: clip lengths to [1..max_len] for cleaner axes (bins at integers)
    """
    en_lengths = np.array([len(src) for src, _ in pairs], dtype=np.int32)
    id_lengths = np.array([len(tgt) for _, tgt in pairs], dtype=np.int32)

    # clip to [1..max_len] so the plot matches the book-style figure (4..20 etc. is common)
    en_lengths = np.clip(en_lengths, 1, max_len)
    id_lengths = np.clip(id_lengths, 1, max_len)

    # Build integer-centered bins: [0.5, 1.5, 2.5, ...]
    bins = np.arange(0.5, max_len + 1.5, 1.0)

    # 2D histogram counts
    H, xedges, yedges = np.histogram2d(en_lengths, id_lengths, bins=[bins, bins])

    # Plot (transpose so y increases upward with imshow’s default origin='upper' handling)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        H.T,
        origin="lower",          # so (1,1) starts at bottom-left
        interpolation="nearest",
        cmap="gray_r",           # like the book: darker = higher count
        extent=[1, max_len, 1, max_len]
    )
    plt.colorbar(im, label="Count")
    plt.title(title)
    plt.xlabel("# English Tokens")
    plt.ylabel("# Indonesian Tokens")
    plt.tight_layout()
    plt.show()

# ---- Example usage (assuming you already have 'pairs' from the previous loader) ----
pairs = load_pairs("data/ind-eng/ind.txt", max_len=20)  # English–Indonesian
plot_joint_length_heatmap(pairs, max_len=20,
                          title="Joint Distribution of Sentence Lengths (EN–ID)")
