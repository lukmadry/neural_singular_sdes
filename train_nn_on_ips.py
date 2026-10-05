import h5py
import numpy as np

import torch
from torch import nn

from helpers import inverse_power_law, train_epoch, validation_epoch
from nets import get_model

def compute_drift_vec(arr, positions):
	diffs = arr[:, None] - positions[None, :]
	return inverse_power_law(diffs).mean(axis=1)

# I guess it'd be better if it'd be a generator, TODO to rethink usage, maybe supply DataLoader from PyTorch
# TODO there are two options : either we supply time argument as it is, or we split it via frequencies
def load_and_prepare_data(fname, lower_bd, upper_bd, grid_size):
	with h5py.File(fname, "r") as f:
		positions = f['ips_steps'][:]
	xgrid = np.linspace(lower_bd, upper_bd, grid_size)
	return xgrid, positions


def build_data(x_grid, time_indexes, pos_arr, shuffle=True):
	"""

	:param x_grid: np.ndarray, evenly spaced grid
	:param time_indexes: iterable of ints
	:param pos_arr: 2D np.array containing the positions of particles
	:param shuffle:
	:return:
	"""
	all_vals = []
	time_max = pos_arr.shape[0]
	for t in time_indexes:
		vals = compute_drift_vec(x_grid, pos_arr[t, :])
		all_vals.append(vals)

	ngrid = x_grid.shape[0]
	xs = [np.stack([x_grid, np.ones(ngrid) * (t/time_max)], axis=1) for t in time_indexes]
	xs = np.concatenate(xs, axis=0)
	if shuffle:
		idx = np.random.permutation(xs.shape[0])
		return xs[idx], np.concatenate(all_vals, axis=0)[idx]
	return xs, np.concatenate(all_vals, axis=0)

# TODO to be reworked to not mix time, and force us to check interpolation
def train_val_splitter(X, y, train_size=0.75, ret_torch=True):
	idx = np.random.permutation(X.shape[0])
	N = int(X.shape[0] * train_size)
	y = y.reshape(-1, 1)
	if ret_torch:
		X = torch.as_tensor(X, dtype=torch.float32)
		y = torch.as_tensor(y, dtype=torch.float32)
		return X[idx[:N]], y[idx[:N]], X[idx[N:]], y[idx[N:]]
	return X[idx[:N]], y[idx[:N]], X[idx[N:]], y[idx[N:]]

xgrid, positions = load_and_prepare_data("ips_300_0.5.hdf5", -10, 10, 1000)
t_k = np.arange(0, positions.shape[0], 10000)
t_k_norm = t_k / positions.shape[0]

batch_size = 64
X_s, Y_s = build_data(x_grid=xgrid,
					  time_indexes=t_k,
					  pos_arr=positions)

x_train_s, y_train_s, x_val_s, y_val_s = train_val_splitter(X_s, Y_s)
nb_batches_train = x_train_s.shape[0] // batch_size
nb_batches_val = x_val_s.shape[0] // batch_size

def train_loop(nb_epochs, net, optim,
               loss_fn, x_train, y_train, x_val, y_val):

	for i in range(nb_epochs):
		mean_loss = train_epoch(net, optim, x_train, y_train, batch_size, nb_batches_train, loss_fn)
		print(f"Epoch: {i}. Mean train loss: {mean_loss:.4f}")
		mean_loss = validation_epoch(net, x_val, y_val, batch_size, nb_batches_val, loss_fn)
		print(f"Epoch: {i}. Mean validation loss: {mean_loss:.4f}")

mse_loss = nn.MSELoss()
model, opt = get_model(2, [256, 256], 1e-2)
first_preds = model.forward(x_train_s)
L = mse_loss(first_preds, y_train_s).item()
print(f"Initial loss: {L}")

try:
	train_loop(100, model, opt, mse_loss, x_train_s, y_train_s, x_val_s, y_val_s)
except KeyboardInterrupt:
	pass

# saving model