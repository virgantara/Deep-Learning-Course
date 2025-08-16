from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV

ENCODER_EMBEDDING_DIM, DECODER_HIDDEN_SIZE  = 256, 256

enc = BahdanauEncoder(input_dim=len(en_vocab),
						embedding_dim = ENCODER_EMBEDDING_DIM,
						encoder_hidden_dim=ENCODER_HIDDEN_SIZE,
						decoder_hidden_dim = DECODER_HIDDEN_SIZE,
						dropout_p = 0.15)

attn = BahdanauAttentionQKV(DECODER_HIDDEN_SIZE)

dec = BahdanauDecoder(output_dim=len(id_vocab), 
						embedding_dim=DECODER_HIDDEN_SIZE,
						encoder_hidden_dim = ENCODER_HIDDEN_SIZE,
						decoder_hidden_dim=DECODER_HIDDEN_SIZE,
						attention = attn,
						dropout_p = 0.15)

