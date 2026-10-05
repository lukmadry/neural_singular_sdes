import torch
from torch import nn

class MLP(nn.Module):
	def __init__(self, input_size, hidden_layers, output_size, num_bands = 16):
		super(MLP, self).__init__()
		# self.encoder = FourierEncoding(num_bands=num_bands, max_freq=15.0)
		sequences = []
		layers = list(zip([input_size] + hidden_layers, hidden_layers + [output_size]))
		for l0, l1 in layers[:-1]:
			sequences.append(nn.Linear(l0, l1))
			sequences.append(nn.SiLU())
		sequences.append(nn.Linear(layers[-1][0], layers[-1][1]))
		self.net = nn.Sequential(*sequences)

	def forward(self, x):
		# enc_pos = self.encoder(x)
		# enc_neg = self.encoder(-x)
		pos = self.net(x)
		neg = self.net(-x)
		return 0.5*(pos-neg)


def get_model(input_size, hidden_layers, learning_rate):
	mlp = MLP(input_size, hidden_layers, 1)
	optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
	return mlp, optimizer
