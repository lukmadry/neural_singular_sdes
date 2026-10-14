import numpy as np
import os
import json

from collections import Counter

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

# we ignore coefs
def find_closest_path(fn_params, storage_filelist,
                      alph_range=(0, 0.49),
                      mult_range=(0.5, 3)):

	closest_path_idx = 0
	min_path_dist = 100
	fn_alph_norm = (fn_params[0] - alph_range[0]) / (alph_range[1] - alph_range[0])
	fn_mult_norm = (fn_params[1] - mult_range[0]) / (mult_range[1] - mult_range[0])
	for i, path in enumerate(storage_filelist):
		parts = path.split("_")
		alph = float(parts[3])
		multip = float(parts[5])
		alph_norm = (alph - alph_range[0])/(alph_range[1] - alph_range[0])
		multip_norm = (multip - mult_range[0])/(mult_range[1] - mult_range[0])
		dist = (fn_alph_norm - alph_norm) ** 2 + (fn_mult_norm - multip_norm) ** 2
		if dist < min_path_dist and dist > 1e-6: # i.e. it's not the same path, there are surely better way to do this
			closest_path_idx = i
			min_path_dist = dist
	return storage_filelist[closest_path_idx]

def find_closest_path_cluster(fn_param_dict, storage_filelist,
                            alph_range=(0, 0.49), mult_range=(0.5, 3),
                              assume_empty_intersection=False):

	if assume_empty_intersection: # we will just compare the mean
		alph = np.mean(fn_param_dict['alphas'])
		mult = np.mean(fn_param_dict['multipliers'])
		return find_closest_path((alph, mult), storage_filelist, alph_range=alph_range, mult_range=mult_range)
	else:
		path_counter = Counter()
		# we will make sure that the intersection is empty
		filtered_storage_filelist = []
		for path in storage_filelist:
			parts = path.split("_")
			alph = float(parts[3])
			multip = float(parts[5])
			for a, m in zip(fn_param_dict['alphas'], fn_param_dict['multipliers']):
				if not np.allclose([a, m], [alph, multip]):
					filtered_storage_filelist.append(path)

		for a, m in zip(fn_param_dict['alphas'], fn_param_dict['multipliers']):
			path = find_closest_path((a, m), filtered_storage_filelist)
			path_counter[path] += 1

		return path_counter.most_common(1)[0][0]

def path_to_params(path, return_dict=False, return_as_list=False):
	parts = path.split("_")
	alph = float(parts[3])
	multip = float(parts[5])
	coefs = [float(x.replace(".hdf5", "")) for x in parts[7:]]
	if return_dict and not return_as_list:
		return {"alphas": alph, "multipliers": multip, "coefs": coefs}
	elif return_dict and return_as_list:
		return {"alphas": [alph], "multipliers": [multip], "coefs": [coefs]}
	return alph, multip, coefs

def build_a_cluster(params, storage_filelist, sample_k_paths,
                    storage_dir='storage_dir', idx=0):
	trunc_params = {}
	for key, items in params.items():
		trunc_params[key] = [items[idx]]  # this can be randomised as well

	for k in range(sample_k_paths):
		new_path = find_closest_path_cluster(fn_param_dict=trunc_params,
		                                     storage_filelist=storage_filelist,
		                                     assume_empty_intersection=True)
		storage_filelist = list(set(storage_filelist).difference([new_path]))
		new_params = path_to_params(new_path, return_dict=True)
		new_params['storage_names'] = os.path.join(storage_dir, new_path)
		for key in trunc_params.keys():
			trunc_params[key].append(new_params[key])

	return trunc_params

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
