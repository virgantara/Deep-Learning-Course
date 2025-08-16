import torch
import torch.nn as nn
import random

class BahdanauSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    @torch.no_grad()
    def make_src_mask(self, src):
        """
        src: [src_len, batch]
        returns mask: [src_len, batch] with 1 for valid tokens, 0 for pad
        """
        return (src != self.pad_id).long()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Training forward pass with teacher forcing.

        src: [src_len, batch]
        trg: [trg_len, batch] -- includes <bos> at trg[0], and typically ends with <eos> (padded)
        returns:
            outputs: [trg_len, batch, vocab_out] (logits)
            attentions: [trg_len, batch, src_len]
        """
        src_len, batch_size = src.size()
        trg_len, _ = trg.size()

        # Encode
        encoder_outputs, hidden = self.encoder(src)  # enc_out: [src_len, batch, 2*enc_h], hidden: [batch, dec_h]
        src_mask = self.make_src_mask(src)           # [src_len, batch]

        vocab_out = self.decoder.out.out_features
        outputs = torch.zeros(trg_len, batch_size, vocab_out, device=self.device)
        attentions = torch.zeros(trg_len, batch_size, src_len, device=self.device)

        # First input to decoder is <bos>
        input_t = trg[0, :]  # [batch], expected to be <bos>

        for t in range(1, trg_len):
            logits, hidden, attn = self.decoder(input_t, hidden, encoder_outputs, src_mask)
            outputs[t] = logits
            attentions[t] = attn  # [batch, src_len]

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)  # [batch]

            input_t = trg[t] if teacher_force else top1

        return outputs, attentions

    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        """
        Inference (no teacher forcing), returns predicted token ids and attentions.

        src: [src_len, batch]
        returns:
            ys: [out_len, batch] (predicted tokens, starts with <bos>, ends at <eos> or max_len)
            attentions: [out_len, batch, src_len]
        """
        src_len, batch_size = src.size()
        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        # Start with <bos>
        input_t = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=self.device)

        ys = [input_t]  # list of [batch]
        attn_list = []

        for _ in range(1, max_len):
            logits, hidden, attn = self.decoder(input_t, hidden, encoder_outputs, src_mask)  # logits: [batch, vocab]
            pred = logits.argmax(1)  # [batch]
            ys.append(pred)
            attn_list.append(attn)   # [batch, src_len]

            input_t = pred
            # Early stop if all predicted EOS
            if (pred == self.eos_id).all():
                break

        ys = torch.stack(ys, dim=0)  # [out_len, batch]
        if attn_list:
            attentions = torch.stack(attn_list, dim=0)  # [out_len-1, batch, src_len]
            # pad one step at the front to align lengths with ys
            pad_attn = torch.zeros(1, ys.size(1), src_len, device=ys.device)
            attentions = torch.cat([pad_attn, attentions], dim=0)  # [out_len, batch, src_len]
        else:
            attentions = torch.zeros(1, ys.size(1), src_len, device=ys.device)

        return ys, attentions
