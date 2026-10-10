import numpy as np
import os
import json

def compare_dimensions(x, y):
	if isinstance(x, list) and isinstance(y, np.ndarray):
		x = np.array(x)
		return x.shape == y.shape
	elif isinstance(x, np.ndarray) and isinstance(x, list):
		return compare_dimensions(y, x)
	elif isinstance(x, list) and isinstance(y, list):
		return np.array(x).shape == np.array(y).shape
	else:
		return x.shape == y.shape


def none_sentinel(arg1, arg2):
	if arg1 is not None and arg2 is not None:
		raise ValueError
	elif arg1 is None and arg2 is None:
		raise ValueError

def build_progress_bar(i, M, length=50, p_sign='@'):
	prop = int(float(i)/float(M)* length)
	s = "[" + p_sign * prop + '_' * (length - prop) + "]" + f" {100*float(i/M):.2f}%"
	return s

def update_json(filename, json_dict):
	if os.path.isfile(filename):
		with open(filename, "r") as f:
			new_json = json.load(f)
		keys = set(json_dict.keys()) & set(new_json.keys())

		for key in keys:
			new_json[key].extend(json_dict[key])

		with open(filename, "w") as f:
			json.dump(new_json, f)
	else:
		with open(filename, "w") as f:
			json.dump(json_dict, f)


def function_factory(power, multiplier, taylor_coefs, eps):
	def fn(x):
		polynomial = 0
		for j, t in enumerate(taylor_coefs):
			polynomial += np.power(x, j+1) * t
		return np.sign(x) * multiplier/(eps + np.power(np.abs(x), power)) * (1 + np.maximum(-1, np.minimum(polynomial, 1)))
	return fn

# this is just a convenience structure for input data, not a Dataset in a PyTorch sense which needs to provide outputs as well
class OneFunctionOnAGrid:
	def __init__(self, xgrid, alpha, multip, coefs, eps, times=None,
	             max_degree=5, with_histogram=False, position_arr=None):
		self.xgrid = xgrid
		self.alpha = alpha
		self.multip = multip
		self.times = times
		self.coefs = np.zeros(max_degree)
		self.max_degree = max_degree
		self.coefs[:len(coefs)] = coefs
		self.no_param = 2 + max_degree
		self.input_size = self.no_param + 2
		self.fn = function_factory(alpha, multip, coefs, eps)

		if times is not None:
			self.inputs = np.zeros((len(times) * len(xgrid), self.input_size))
			for i, t in enumerate(self.times):
				arr = self.create_input_with_time(t)
				self.inputs[i * len(xgrid):(i + 1) * len(xgrid)] = arr

		if position_arr is not None:
			self._input_pos_arr(position_arr)

	def create_input_with_time(self, time):
		arr = np.zeros((self.xgrid.shape[0], self.input_size))
		arr[:, 0] = self.alpha
		arr[:, 1] = self.multip
		arr[:, 2:2 + self.max_degree] = self.coefs.reshape(1, -1)
		arr[:, self.no_param] = time
		arr[:, self.no_param + 1] = self.xgrid
		return arr

	def _input_pos_arr(self, pos_arr):
		pass

	def __getitem__(self, index):
		return self.inputs[index]

	def __len__(self):
		return self.inputs.shape[0]
