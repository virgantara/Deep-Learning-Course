import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttentionQKV(nn.Module):

	def __init__(self, hidden_size, query_size=None, key_size=None, dropout_p=0.15):
		super().__init__()

		self.hidden_size = hidden_size
		self.query_size = hidden_size if query_size is None else query_size

		self.key_size = 2 * hidden_size if key_size is None else key_size

		self.query_layer = nn.Linear(self.query_size, hidden_size)
		self.key_layer = nn.Linear(self.key_size, hidden_size)
		self.energy_layer = nn.Linear(hidden_size, 1)
		self.dropout = nn.Dropout(dropout_p)

	def forward(self, hidden, encoder_outputs, src_mask=None):
		query_out = self.query_layer(hidden)

		key_out = self.key_layer(encoder_outputs)

		query_exp = query_out.unsqueeze(0) 
		energy_input = torch.tanh(query_exp + key_out)
		energies = self.energy_layer(energy_input).squeeze(2)

		if src_mask is not None:
			energies.data.masked_fill_(src_mask == 0, float("-inf"))

		weights = F.softmax(energies, dim=0)
		weights = self.dropout(weights)

		return weights.transpose(0,1)
