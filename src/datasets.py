import itertools
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from src.helpers import function_factory, build_progress_bar


def compute_drift_vec(arr, positions, fn):
	diffs = arr[:, None] - positions[None, :]
	return fn(diffs).mean(axis=1)


def build_part_of_dataset(xgrid, positions_arr,
                          time_indices, fn_parameters):
	fn = function_factory(*fn_parameters)
	fval = fn(xgrid)
	ft = np.fft.fft(fval)
	f_real = np.real(ft)
	f_imag = np.imag(ft)
	# TODO check the frequencies and how to cut them
	total_x = []
	for i, ti in enumerate(time_indices):
		vals = compute_drift_vec(xgrid, positions_arr[i, :], fn)
		total_x.append((f_real, f_imag, ti, vals))
	return total_x


def build_part_of_dataset_parametrised(xgrid, fn_parameters,
                                       time_indices, positions_arr,
                                       with_init_dist=False, bins_for_init_dist = 30):
	"""
	Creates a part of a dataset corresponding to a fixed set of parameters. It evaluates drift parametrised with
	fn_parameters at point x, in other words it computes

	x -> 1/N \sum_{j=1}^N f(x - X^j_t)

	for all t in time_indices

	Parameters
	--------
	xgrid : an evenly space grid
	fn_parameters : a tuple of paramaters (a, m, coefs, eps), where a, m are floats, coefs is a list or None,
		eps is a very small float for numerical stability
	time_indices : an array of time indices at which we are going to evaluate.
	positions_arr: an array loaded from an experiment

	Returns
	------
	(parameters of function, time, list of values of f(x - X^j) at this time evaluated at x=xgrid, prop)

	where prop is either starting values or a histogram of a starting distribution
	"""
	fn = function_factory(*fn_parameters)
	total_x = []
	if with_init_dist: # could be optimised by precomputing the edges, but the most expensive part of the computation is generating McK so
		# maybe we can just listen to Donald Knuth
		bins_edges = np.linspace(-3, 3, num=bins_for_init_dist)
		prop, _ = np.histogram(positions_arr[0, :], bins=bins_edges, density=True) # 0 index is initial distribution
		prop /= prop.sum() # normalise
	else:
		prop = positions_arr[0, :]
	for i, ti in enumerate(time_indices[1:], 1):
		vals = compute_drift_vec(xgrid, positions_arr[i, :], fn)
		total_x.append((fn_parameters, ti, vals, prop))

	return total_x


def build_whole_dataset_parametrised(xgrid, params_dict,
                                     Ninter=150, stop_at_k=np.inf, return_rarified_pos_arr=False,
                                     with_init_dist=False):
	"""
	Reads the files and creates a dataset that can be fed into a neural network

	Parameters
	---------
	xgrid : an evenly space grid. The assumption that it is evenly spaced is NOT tested against
	params_dict : a dict with keys : alphas, multipliers, coefs, storage_names
	:param Ninter:
	:param stop_at_k:

	Returns
	-----
	:return: (a list of tuples arranged as in build_part_..., time_indices)
	"""
	alphas, multips, coefs, data = params_dict['alphas'], params_dict['multipliers'], params_dict['coefs'], params_dict['storage_names']
	eps = 1/np.power(Ninter, 1-1e-4)
	dt = 0.1 * eps * eps # reproducing the computation of dt in the .cpp file
	M = int(1/dt) + 1
	time_skeleton = np.arange(0, M, 1000) # we can start at zero because later on we start with [1:] anyway
	time_indices = time_skeleton * dt

	if stop_at_k < np.inf:
		time_indices = time_indices[:2*stop_at_k]

	all_datasets = []
	print(f"Number of time steps: {int(M/1000)}")
	print("Progress at loading storage: ")
	pos_arrays = []

	coefs_iterator = coefs if coefs is not None else itertools.repeat(None)

	for nb, (a, m, c, storage_fname) in enumerate(zip(alphas, multips, coefs_iterator, data)):
		# we will first check if Ninter matches the file

		if storage_fname.startswith("--f"):
			storage_fname = storage_fname.replace("--f ", "") # quick fix to a small bug in a previous iteration

		fname = Path(storage_fname).name
		ninter_idx = fname.index("N_")
		ninter = int(fname[ninter_idx:ninter_idx+7].split("_")[1])
		if ninter != Ninter:
			raise ValueError("number of particles that you passed does not match the number in the dataset")
		with h5py.File(storage_fname, "r") as f:
			positions_arr = f['drift_results'][:]

		positions_arr = positions_arr.reshape(-1, Ninter).astype(np.float32)
		part_dataset = build_part_of_dataset_parametrised(xgrid, (a, m, c, eps),
		                                                  time_indices, positions_arr,
		                                                  with_init_dist=with_init_dist)
		all_datasets.extend(part_dataset)
		if return_rarified_pos_arr:
			pos_arrays.append(positions_arr) # no need to do anything, they are already rarified in c++

		if nb > stop_at_k: # mostly for testing
			break

		print(build_progress_bar(nb, len(data)), end="\r")

	print(build_progress_bar(len(data), len(data)), end="\n")
	if return_rarified_pos_arr:
		return all_datasets, time_indices, pos_arrays
	return all_datasets, time_indices


def split_param_dict_train_val(param_dict,
                               train_size=0.75, output_max = (None, None)):
	idx = np.random.permutation(len(param_dict['alphas']))
	Ntrain = int(len(param_dict['alphas']) * train_size )

	train_dict, val_dict = {}, {}

	for tag in param_dict.keys():
		train_v = [param_dict[tag][i] for i in idx[:Ntrain]]
		if output_max[0] is not None:
			train_v = train_v[:output_max[0]]
		val_v = [param_dict[tag][i] for i in idx[Ntrain:]]
		if output_max[1] is not None:
			val_v = val_v[:output_max[1]]
		train_dict[tag] = train_v
		val_dict[tag] = val_v

	return train_dict, val_dict


class DatasetParametrised(Dataset):
	def __init__(self, xgrid, all_datasets, max_polynomial_degree=5,
	             with_histogram=False, bins=30):
		"""
		Dataset that takes the

		Parameters
		-------
		xgrid , np.ndarray of evenly spaced grid
		all_datasets: a list of tuples in the format
		    (parameters of a function, time, values of a function at fgrid)

		"""

		self.xgrid = xgrid
		self.all_datasets = all_datasets

		self.length_of_one_fdataset = len(self.all_datasets[0][2]) # we train network on the grid
		# above could have also been xgrid.shape[0]
		# we will not check this every time __getitem__ is called so we check it now
		for dataset in self.all_datasets:
			if len(dataset[2]) != self.length_of_one_fdataset:
				raise ValueError("all lengths of a last array in dataset should be equal")

		self.k = len(self.all_datasets)
		self.no_param = 2 + max_polynomial_degree # a, multiplier and polynomial coefficients
		self.input_size = self.no_param + 2 # functions parameters time and space

		if with_histogram:
			self.input_size += bins-1 # if number of edges is 30 then there are 29 cells
			self.bins = bins

	def __getitem__(self, index):

		nr_dataset = index // self.length_of_one_fdataset
		nr_point = index % self.length_of_one_fdataset
		u = np.zeros(self.input_size, dtype=np.float32) # plus time and space argument
		u[0] = self.all_datasets[nr_dataset][0][0] # alpha parameter
		u[1] = self.all_datasets[nr_dataset][0][1] # multiplier
		taylor = self.all_datasets[nr_dataset][0][2]
		u[2:2+len(taylor)] = taylor
		u[self.no_param] = self.all_datasets[nr_dataset][1] # time
		u[self.no_param+1] = self.xgrid[nr_point] # x
		return u, self.all_datasets[nr_dataset][2][nr_point] #( (fn_params, time, x), y)

	def __len__(self):
		return self.k * self.length_of_one_fdataset


class DatasetParametrisedHistogramNoPolynomial(DatasetParametrised):
	def __init__(self, xgrid, all_datasets, bins=30):
		super().__init__(xgrid, all_datasets, max_polynomial_degree=0, with_histogram=True, bins=bins)

	def __getitem__(self, index):
		nr_dataset = index // self.length_of_one_fdataset
		nr_point = index % self.length_of_one_fdataset
		u = np.zeros(self.input_size, dtype=np.float32)  # plus time and space argument
		u[0] = self.all_datasets[nr_dataset][0][0]  # alpha parameter
		u[1] = self.all_datasets[nr_dataset][0][1]  # multiplier
		u[self.no_param] = self.all_datasets[nr_dataset][1]  # time
		u[self.no_param + 1] = self.xgrid[nr_point]  # x
		u[self.no_param + 2:self.no_param + 2 + self.bins-1] = self.all_datasets[nr_dataset][3]
		return u, self.all_datasets[nr_dataset][2][nr_point]  # ( (fn_params, time, x), y)
