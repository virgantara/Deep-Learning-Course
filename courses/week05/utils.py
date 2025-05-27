
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Convert tokens to indices
    token_ids = [torch.tensor([vocab[token] for token in tokens], dtype=torch.long) for tokens in texts]
    padded = pad_sequence(token_ids, batch_first=True, padding_value=vocab['<pad>'])
    return padded, torch.tensor(labels)