import torch
import torch.nn as nn

class BahdanauDecoder(nn.Module):

	def __init__(self, output_dim, embedding_dim, 
		encoder_hidden_dim, decoder_hidden_dim, attention, dropout_p):

		super().__init__()

		self.embedding_dim = embedding_dim
		self.output_dim = output_dim
		self.encoder_hidden_dim = encoder_hidden_dim
		self.decoder_hidden_dim = decoder_hidden_dim
		self.dropout_p = dropout_p

		self.embedding = nn.Embedding(output_dim, embedding_dim)

		self.attention = attention
		self.gru = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim)

		self.out = nn.Linear((encoder_hidden_dim * 2) + embedding_dim + decoder_hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout_p)

	def forward(self, x, hidden, encoder_outputs, src_mask=None):
		x = x.unsqueeze(0)
		x = self.embedding(x)
		embedded = self.dropout(x)

		attentions = self.attention(hidden, encoder_outputs, src_mask)
		a = attentions.unsqueeze(1)
		encoder_outputs = encoder_outputs.transpose(0, 1)

		weighted = torch.bmm(a, encoder_outputs)
		weighted = weighted.transpose(0, 1)

		rnn_input = torch.cat((embedded, weighted), dim=2)

		output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

		assert (output == hidden).all()

		embedded = embedded.squeeze(0)
		output = output.squeeze(0)
		weighted = weighted.squeeze(0)

		linear_input = torch.cat((output, weighted, embedded), dim=1)

		output = self.out(linear_input)
		return output, hidden.squeeze(0), attentions

