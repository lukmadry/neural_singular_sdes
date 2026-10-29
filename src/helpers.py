import numpy as np
import os
import json
import h5py

from collections import Counter

from scipy.integrate import trapezoid
from scipy.stats import norm, uniform

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

def update_json(filename, json_dict, replace_old_json=False):
	if os.path.isfile(filename) and not replace_old_json:
		with open(filename, "r") as f:
			new_json = json.load(f)
		keys = set(json_dict.keys()) & set(new_json.keys())

		for key in keys:
			if new_json[key] is not None: # it's either None or a list
				new_json[key].extend(json_dict[key])

		with open(filename, "w") as f:
			json.dump(new_json, f)
	elif not os.path.isfile(filename) or replace_old_json:
		with open(filename, "w") as f:
			json.dump(json_dict, f)
	else:
		raise ValueError


def function_factory(power, multiplier, taylor_coefs, eps):
	if taylor_coefs is not None:
		def fn(x):
			polynomial = 0
			for j, t in enumerate(taylor_coefs):
				polynomial += np.power(x, j+1) * t
			return np.sign(x) * multiplier/(eps + np.power(np.abs(x), power)) * (1 + np.maximum(-1, np.minimum(polynomial, 1)))
	else:
		def fn(x):
			return np.sign(x) * multiplier/(eps + np.power(np.abs(x), power))
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

# essentially a join
def update_param_file_with_dist_data(params, distribution_logfile_path):
	with open(distribution_logfile_path, "r") as f:
		logfile = f.readlines()

	mixt_dict = {}
	for line in logfile:
		parts = line.strip("\n").split("|||")
		mean, scale = float(parts[-2].lstrip("loc: ")), float(parts[-1].lstrip("scale: "))
		fname = parts[0].split("/")[-1].strip(".hdf5")
		mixt_dict[fname] = {"loc": mean, "scale": scale}

	loc_arr = []
	scale_arr = []
	for fname in params['storage_names']:
		distname_clear = "mixture_" + fname.split("_")[-1].split(".")[0] # this works in a very specific conditions and is not very robust
		loc_arr.append(mixt_dict[distname_clear]['loc'])
		scale_arr.append(mixt_dict[distname_clear]['scale'])
	params['loc'] = loc_arr
	params['scale'] = scale_arr
	return params

# TODO test it
def filter_params(params, param_ranges, return_indices=True):
	params_indices_in = []
	params_indices_partially_out = []
	params_indices_tot_out = []

	keys = list(param_ranges.keys())
	all_keys = list(params.keys())
	keys = [key for key in keys if key in all_keys]
	K = len(keys)
	N = len(params[keys[0]])
	for i in range(N):
		nr_in = 0
		for key in keys:
			if param_ranges[key][1] > params[key][i] > param_ranges[key][0]:
				nr_in += 1
		if nr_in == K:
			params_indices_in.append(i)
		elif nr_in == 0:
			params_indices_tot_out.append(i)
		else:
			params_indices_partially_out.append(i)

	params_in = {}
	params_part_out = {}
	params_tot_out = {}

	keys_params = list(params.keys())

	for idx_list, output_dict in zip([params_indices_in, params_indices_partially_out, params_indices_tot_out],
	                                 [params_in, params_part_out, params_tot_out]):
		for key in keys_params:
			if params[key] is not None:
				output_dict[key] = [params[key][i] for i in idx_list]
			else:
				output_dict[key] = None

	if return_indices:
		return params_in, params_part_out, params_tot_out, params_indices_in, params_indices_partially_out, params_indices_tot_out
	return params_in, params_part_out, params_tot_out

# this is just a convenience structure for input data, not a Dataset in a PyTorch sense which needs to provide outputs as well
class OneFunctionOnAGrid:
	def __init__(self, xgrid, alpha, multip, coefs, eps, times=None,
	             max_degree=5, with_histogram=False, histogram=None, position_arr=None, bins=30):
		"""
		Class holding data on a drift function and array of positions

		:param xgrid: np.ndarray, evenly spaced grid
		:param alpha: float, single value, not checked
		:param multip: float, single value, not checked
		:param coefs: iterable of floats
		:param eps: float, single value, not checked
		:param times:
		:param max_degree:
		:param with_histogram: bool, if True then the input contains histogram of an initial distribution
		:param histogram: np.ndarray, histogram from the previous parameter
		:param position_arr: np.ndarray, array of positions of particles.
		:param bins: either array of bin edges or a number of bins for histogram. Only relevant if histogram not passed

		# TODO compute histogram from position_arr
		"""
		self.xgrid = xgrid
		self.alpha = alpha
		self.multip = multip
		self.times = times
		self.coefs = np.zeros(max_degree)
		if coefs is not None:
			self.coefs[:len(coefs)] = coefs
			self.max_degree = max_degree
		else:
			self.coefs = None
			self.max_degree = 0
		self.no_param = 2 + self.max_degree
		self.input_size = self.no_param + 2
		self.fn = function_factory(alpha, multip, coefs, eps)
		self.bins_cnt = bins.shape[0] if isinstance(bins, np.ndarray) else bins
		self.bins = np.linspace(-3, 3, num=self.bins_cnt)
		if with_histogram:
			self.input_size += self.bins_cnt-1

		if position_arr is not None:
			self._input_pos_arr(position_arr)

		if with_histogram and histogram is None:
			raise ValueError("if with_histogram is True, then you need to pass a histogram")
		elif with_histogram and histogram is not None:
			self.hist = histogram # because in this case we will just pass the histogram

		if with_histogram:
			self.create_input_with_time = self._create_input_with_time_hist
			self.create_input_with_time_and_pos = self._create_input_with_time_and_pos_hist
		else:
			self.create_input_with_time = self._create_input_with_time_no_hist
			self.create_input_with_time_and_pos = self._create_input_with_time_and_pos_no_hist

		if times is not None:
			self.inputs = np.zeros((len(times) * len(xgrid), self.input_size))
			for i, t in enumerate(self.times):
				arr = self.create_input_with_time(t)
				self.inputs[i * len(xgrid):(i + 1) * len(xgrid)] = arr

		self.with_hist = with_histogram

	def _create_input_with_time_no_hist(self, time):
		arr = np.zeros((self.xgrid.shape[0], self.input_size))
		arr[:, 0] = self.alpha
		arr[:, 1] = self.multip
		arr[:, 2:2 + self.max_degree] = self.coefs.reshape(1, -1)
		arr[:, self.no_param] = time
		arr[:, self.no_param + 1] = self.xgrid
		return arr

	def _create_input_with_time_hist(self, time):
		# we could precreate a building block
		arr = np.zeros((self.xgrid.shape[0], self.input_size))
		arr[:, 0] = self.alpha
		arr[:, 1] = self.multip
		arr[:, 2] = time
		arr[:, 3] = self.xgrid
		arr[:, 4:4+self.bins_cnt-1] = self.hist
		return arr

	def _create_input_with_time_and_pos_no_hist(self, time, pos):
		arr = np.zeros((1, self.input_size))
		arr[:, 0] = self.alpha
		arr[:, 1] = self.multip
		arr[:, 2:2 + self.max_degree] = self.coefs.reshape(1, -1)
		arr[:, self.no_param] = time
		arr[:, self.no_param + 1] = pos
		return arr

	def _create_input_with_time_and_pos_hist(self, time, pos):

		arr = np.zeros((self.xgrid.shape[0], self.input_size))
		arr[:, 0] = self.alpha
		arr[:, 1] = self.multip
		arr[:, 2] = time
		arr[:, 3] = pos
		arr[:, 4:4+self.bins_cnt-1] = self.hist
		return arr

	def _input_pos_arr(self, pos_arr):
		self._pos_arr = pos_arr

	def apply_fn_pos_arr(self, x, tidx, pos_arr):
		diffs = x - pos_arr[tidx, :]
		return self.fn(diffs).mean()

	def apply_fn(self, x, tidx):
		diffs = x - self._pos_arr[tidx, :]
		return self.fn(diffs).mean()

	def __getitem__(self, index):
		return self.inputs[index]

	def __len__(self):
		return self.inputs.shape[0]

# TODO to be tested separately
def redistribute_points_inside_bin(counts, grid, density, bin_idx_left=None,
                                   bin_idx_right=None, subbin_count=10, return_rounded_val=False):
	if counts == 0:
		return np.zeros(subbin_count)
	if bin_idx_left is None:
		bin_idx_left = 0
	if bin_idx_right is None:
		bin_idx_right = grid.shape[0]
	dx = grid[1] - grid[0]
	probas = []
	idx_l = bin_idx_left
	jump = (bin_idx_right - bin_idx_left) // subbin_count
	idx_h = idx_l + jump
	for i in range(subbin_count):
		probas.append(trapezoid(density[idx_l:idx_h], grid[idx_l:idx_h], dx))
		idx_l = idx_h
		idx_h += jump

	probas = np.array(probas)
	probas /= probas.sum()
	if return_rounded_val:
		weighted_counts = np.array(counts * probas, dtype=int)
		if sum(weighted_counts) < counts:
			added_counts = np.random.choice(a=subbin_count, size=counts - sum(weighted_counts), p=probas)
			for a in added_counts:
				weighted_counts[a] += 1
		return weighted_counts
	else:
		return probas


def interpolate_pdfs(pdf_one, pdf_two, no_dists):
	if pdf_one.shape != pdf_two.shape:
		raise ValueError("pdf_one and pdf_two should have identical shapes")

	if pdf_one.ndim > 1:
		raise ValueError("pdf_one must have one dimension")

	new_pdfs = np.zeros((no_dists, pdf_one.shape[0]))
	weights = np.linspace(0, 1, num=no_dists, endpoint=True)
	for i, w in enumerate(weights):
		pdf = pdf_one * (1-w) + pdf_two * w
		new_pdfs[i, :] = pdf

	return new_pdfs



def read_dist_from_text(text):
	if isinstance(text, list):
		text = text[0]
	dist_type, mean, var = text.split()
	if dist_type == 'gauss':
		return float(mean), float(var), dist_type
	elif dist_type == 'uniform':
		return float(mean-var/2), float(var), dist_type
	else:
		raise NotImplementedError("dist_type should be 'gauss' or 'uniform'")


def generate_and_dump_distributions(textone, texttwo, n_dist, samples_dir, n_samples = 400):

	locone, scaleone, typeone = read_dist_from_text(textone)
	loctwo, scaletwo, typetwo = read_dist_from_text(texttwo)

	if not typeone == typetwo:
		raise NotImplementedError("for now two distributions should be from the same family")

	dist = {"gauss": norm, "uniform": uniform}

	os.makedirs(samples_dir, exist_ok=True)
	filenames = []
	dist_sizes = {}
	weights = np.linspace(0, 1, num=n_dist, endpoint=True)
	dist_datas = []
	for i in range(n_dist):
		loc = locone * (1-weights[i]) + loctwo * weights[i]
		scale = scaleone * (1-weights[i]) + scaletwo * weights[i]

		new_dist = dist[typeone](loc=loc, scale=scale)
		dist_datas.append((loc, scale))
		samples = new_dist.rvs(n_samples)
		# notes - stratified sampling did not seem to have better L2 wasserstein distance
		# not sure if that's an expected outcome (probably not), but I can't spend whole days on fixing this
		#samples = sample_full_dist(n_samples, dense_grid=grid, very_dense_density=pdf, second_level_refinement=True)
		fname = os.path.join(samples_dir, "mixture_" + str(i) + ".hdf5")
		with h5py.File(fname, "w") as f:
			f.create_dataset("init_distribution", data=samples)
		filenames.append(fname)
		dist_sizes[fname] = {"size": len(samples), "loc": loc, "scale": scale}

	with open(os.path.join(samples_dir, "distribution_logfile.txt"), "w") as f:
		info_to_dump = ""
		for fname, s in dist_sizes.items():
			info_to_dump += f"{fname}|||" + str(s["size"]) + "|||loc: " + str(s["loc"]) + "|||scale: " + str(s["scale"]) + "\n"
		f.write(info_to_dump)
	return filenames, dist_sizes


def parse_dist_config_file(dist_config_file):
	with open(dist_config_file, "r") as f:
		config = f.readlines()
	if config[0] != 'DIST_ONE\n':
		raise ValueError(f"incorrect config file format, instead of DIST_ONE we got {config[0]}")
	curr_dist_nr = 1
	distributions_texts = []
	curr_dist = []
	for line in config[1:]:
		if not line.startswith("DIST"):
			curr_dist.append(line.strip("\n"))
		else: # new distribution
			curr_dist_nr += 1
			distributions_texts.append(curr_dist)
			curr_dist = []

	distributions_texts.append(curr_dist)
	if curr_dist_nr != len(distributions_texts): # essentially a sanity check
		raise ValueError(f"incorrect config file format, we have {len(distributions_texts)} distributions, but"
		                 f" the number of distributions counted is {curr_dist_nr}")
	return distributions_texts

# this is not idiot-proof but doesn't have to be
def parse_range_config_file(filepath):
	with open(filepath, "r") as f:
		config = f.readlines()

	alphas_range = []
	multips_range = []
	means_range = []
	scales_range = []
	final_dict = {}

	for line in config:
		param, lower, upper = line.strip("\n").split()
		lower = float(lower)
		upper = float(upper)
		if param.upper() == "ALPHA":
			final_dict["alphas"] = [lower, upper]
		elif param.upper() == "MULTIPLE":
			final_dict["multipliers"] = [lower, upper]
		elif param.upper() == "MEAN":
			final_dict["loc"] = [lower, upper]
		elif param.upper() == "SCALE":
			final_dict["scale"] = [lower, upper]

	return final_dict
