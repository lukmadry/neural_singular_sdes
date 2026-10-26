import os

import h5py
import numpy as np
import argparse
import time
import itertools

import json
from os import makedirs
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from nets import get_model
from helpers import update_json, build_progress_bar, function_factory, path_to_params, find_closest_path, \
	find_closest_path_cluster, build_a_cluster, parse_range_config_file, filter_params
from helpers import parse_dist_config_file, generate_and_dump_distributions, update_param_file_with_dist_data
from evals import Evaluator

# TODO refactor so that this is a main short file which provides access to a backend
# TODO add CI/CD run using this loop

# think about how to best parametrise distributions
# in the directory you should have some parameters of the distribution. Let us maybe first try moving gaussian
# so that it is easy to parametrise and do interpolation/extrapolation thing later
# or just some shape that's moving from left to right so that we can parametrise it with a single number
def sample_interaction_parameters(N, alph_range, multip_range, coefs_range=None,
                                  init_dist_dir=None, with_polynomial=False, max_n_coefs=None, ret_full_paths=True):
	"""

	TODO
	"""
	alphas = np.random.uniform(alph_range[0], alph_range[1], N)
	multips = np.random.uniform(multip_range[0], multip_range[1], N)

	if with_polynomial:
		if coefs_range is None and max_n_coefs is None:
			raise ValueError("if with_polynomial == True, then you need to pass range of coefs and their max nr")
		n_coefs = np.random.randint(low=2, high=max_n_coefs,
		                            size=N)  # low=2 is there because of Horner method in cpp file
		coefs = [[np.random.uniform(low=coefs_range[0], high=coefs_range[1]) for _ in range(k)] for k in n_coefs]
	else:
		coefs = None

	if init_dist_dir is not None:
		all_files = [x for x in os.listdir(init_dist_dir) if x.endswith(".hdf5")]
		dist_idx = np.random.choice(len(all_files), size=N, replace=True)
		dist_files = [all_files[i] for i in dist_idx]
		if ret_full_paths:
			dist_files = [os.path.join(init_dist_dir, x) for x in dist_files]
	else:
		dist_files = None
	return alphas, multips, coefs, dist_files



def create_runner_sh_script(compilation_fpath, cpp_file_fpath, alphas, multips, coefs=None,
                            return_storage_fnames=True, add_compilation=True, Ninter=None,
                            compilation_template=None, storage_dir="storage", init_distributions=None):
	"""

	:param compilation_fpath:
	:param cpp_file_fpath:
	:param alphas:
	:param multips:
	:param coefs:
	:param return_storage_fnames:
	:param add_compilation:
	:param Ninter: int or a dict str -> int, where str is a filepath to an initial_distribution and int is
	    a corresponding Ninter
	:param compilation_template:
	:param storage_dir:
	:param init_distributions:
	:return:
	"""
	big_script = ""
	if add_compilation:
		if compilation_template is None or compilation_fpath is None:
			raise ValueError("if you want to add a compilation script, need to add template (sh script)")
		with open(compilation_template, "r") as f:
			temp = f.read()
		big_script += temp.replace("{FILENAME}", cpp_file_fpath).replace("{OUTPUT_NAME}", compilation_fpath)
		big_script += '\nprintf "finished compilation"\n'
	if return_storage_fnames:
		storages = []

	if Ninter is None:
		raise ValueError("Ninter has to be either int or a dict of ints")
	elif not isinstance(Ninter, (int, dict)):
		raise TypeError("Ninter has to be either int or a dict of ints")

	dist_name_iterator = init_distributions if init_distributions is not None else itertools.repeat(None)
	if init_distributions is None:
		ninter_iterator = itertools.repeat(Ninter)
	else:
		ninter_iterator = [Ninter[x]['size'] for x in dist_name_iterator]
	if coefs is not None:
		for loc in zip(alphas, multips, coefs, dist_name_iterator, ninter_iterator):
			local_alph, local_multip, local_coefs, local_distname, local_ninter = loc
			s = f"./{compilation_fpath} --a {local_alph} --m {local_multip} --distname {local_distname} "
			storage_filename = (f"{storage_dir}/N_{local_ninter}_alph_{local_alph}_mult_{local_multip}_coefs_" +
			                    "_".join([str(x) for x in local_coefs]) + ".hdf5")
			s += "--f " + storage_filename
			s +=  f" --p {len(local_coefs)}"
			if local_distname is None:
				s += f"--N {Ninter}"
			for coef in local_coefs:
				s += f" {coef}"
			big_script += s + "\n"
			if return_storage_fnames:
				storages.append(storage_filename)
	else:
		for loc in zip(alphas, multips, dist_name_iterator, ninter_iterator):
			local_alph, local_multip, local_distname, local_ninter = loc
			s = f"./{compilation_fpath} --a {local_alph} --m {local_multip} --distname {local_distname} "
			if local_distname.endswith(".hdf5"):
				local_distname = local_distname.strip(".hdf5")
				local_distname = local_distname.split("/")[-1]
			storage_filename = (f"{storage_dir}/N_{local_ninter}_alph_{local_alph}_mult_{local_multip}_" +
			                    f"init_dist_{local_distname}.hdf5")
			s += "--f " + storage_filename
			if return_storage_fnames:
				storages.append(storage_filename)
			big_script += s + "\n"
	if return_storage_fnames:
		return big_script, storages
	return big_script

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
                                       ):
	"""

	:param xgrid:
	:param fn_parameters:
	:param time_indices:
	:param positions_arr:
	:return: a list of tuples arranged as :
	(parameters of function, time, list of values of f(x - X^j) at this time evaluated at x=xgrid, starting values of position array)
	"""
	fn = function_factory(*fn_parameters)
	total_x = []
	for i, ti in enumerate(time_indices[1:], 1):
		vals = compute_drift_vec(xgrid, positions_arr[i, :], fn)
		total_x.append((fn_parameters, ti, vals, positions_arr[0, :]))

	return total_x

def build_whole_dataset_parametrised(xgrid, params_dict,
                                     Ninter=150, stop_at_k=np.inf, return_rarified_pos_arr=False):
	"""

	:param xgrid:
	:param params_dict:
	:param Ninter:
	:param stop_at_k:

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

	for nb, (a, m, c, storage_fname) in enumerate(zip(alphas, multips, coefs, data)):
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
		                                                  time_indices, positions_arr)
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

# TODO we may need to do some lazy loading
class DatasetParametrised(Dataset):
	def __init__(self, xgrid, all_datasets, max_polynomial_degree=5, with_histogram=False):
		"""

		:param xgrid:
		:param all_datasets: a list of tuples in the format
		    (parameters of a function, time, values of a function at fgrid)

		"""
		if with_histogram:
			raise NotImplementedError

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
		# TODO I think you should also include the initial distribution, i.e. its histogram


	def __getitem__(self, index):
		nr_dataset = index // self.length_of_one_fdataset
		nr_point = index % self.length_of_one_fdataset
		u = np.zeros(self.no_param + 2, dtype=np.float32) # plus time and space argument
		u[0] = self.all_datasets[nr_dataset][0][0] # alpha parameter
		u[1] = self.all_datasets[nr_dataset][0][1] # multiplier
		taylor = self.all_datasets[nr_dataset][0][2]
		u[2:2+len(taylor)] = taylor
		u[self.no_param] = self.all_datasets[nr_dataset][1] # time
		u[self.no_param+1] = self.xgrid[nr_point] # x
		return u, self.all_datasets[nr_dataset][2][nr_point] #( (fn_params, time, x), y)

	def __len__(self):
		return self.k * self.length_of_one_fdataset

# TODO add parallel training on multiple cores
def train_loop(loader, net, opt,
               loss_fn, max_batches = np.inf, penalty_param = 1):

	t0 = time.perf_counter()
	total_loss = 0.0
	nb_batches = 0
	max_iter = min(max_batches, len(loader))
	for data, labels in loader:
		output = net(data).flatten()
		loss = loss_fn(output, labels, data)

		opt.zero_grad()
		loss.backward()
		opt.step()

		total_loss += loss.item()
		nb_batches += 1
		if nb_batches > max_batches:
			break

		print(build_progress_bar(nb_batches, max_iter), end="\r")

	t1 = time.perf_counter()
	return total_loss / nb_batches, t1 - t0


def test_loop(loader, net, loss_fn, max_batches):

	total_loss = 0.0
	nb_batches = 0
	for data, labels in loader:
		output = net(data).flatten()
		loss = loss_fn(output, labels, data)
		total_loss += loss.item()
		nb_batches += 1
		if nb_batches > max_batches:
			break

	return total_loss / nb_batches


def train_over_epochs(nb_epochs, training_dataloader, test_loader, net, opt,
                      loss, max_batches = np.inf):

	for epoch in range(nb_epochs):
		total_loss, time_spent = train_loop(training_dataloader, net, opt, loss, max_batches)
		print(f"Time spent at epoch {epoch}: {time_spent}")
		print(f"Training loss: {total_loss}")

		val_loss = test_loop(test_loader, net, loss, max_batches)
		print(f"Validation loss: {val_loss}")

def time_sensitive_loss(pred, output, data,
                        time_axis=7, eps=1e-3, multip=0.25):
	t = data[:, time_axis]
	L = 1/torch.sqrt(multip*t+eps) * torch.square(pred-output)
	return L.mean()

def unweighted_mse(pred, output, *args, **kwargs):
	return torch.square(pred-output).mean()

if __name__ == '__main__':
	# TODO allow empty storage_dir and script_filepath argument if Ninter is given
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['generate', 'train',
	                                       'train_one_path', 'train_in_cluster', 'train_in_out'])

	# only relevant for generation
	parser.add_argument("--alpha_range", nargs=2, type=float, default=(0, 0.49))
	parser.add_argument("--multiple_range", nargs=2, type=float, default=(0.5, 3))
	parser.add_argument("--coefs_range", nargs=2, type=float, default=(-0.1, 0.1))
	parser.add_argument("--max_n_coefs", type=int, default=5)

	# generation parameters
	parser.add_argument("--N", type=int)
	parser.add_argument("--Ninter", type=int, default=200)
	parser.add_argument("--script_filepath", type=str, required=True)
	parser.add_argument("--add_compilation", action=argparse.BooleanOptionalAction)
	parser.add_argument("--add_distribution_generation", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--dist_config_file", type=str, default=None)
	parser.add_argument("--n_dist", type=int, default=10)
	parser.add_argument("--samples_dir", type=str)
	parser.add_argument("--with_polynomial", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--compilation_template_fpath", type=str,
	                    default="cpp_simul/cpp_ips_template_compilation.sh")
	parser.add_argument("--compiled_file", type=str,
	                    default="cpp_simul/sde_mkl_fin")
	parser.add_argument("--cpp_file_fpath",
	                    default="cpp_simul/ips_sim_gemini_ubuntu.cpp")

	parser.add_argument("--lower_bd", type=float, default=-5)
	parser.add_argument("--upper_bd", type=float, default=5)
	parser.add_argument("--grid_size", type=float, default=0.1)

	# training parameters
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--hidden_layers", type=int, nargs="+", default=[512, 512, 512])
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--max_batches", type=int, default=10000)
	parser.add_argument("--smoke_run", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--loss", type=str, choices=['mse', 'time_weighting'], default='mse')
	parser.add_argument("--weight_decay", default=0.0, type=float)
	parser.add_argument("--no_epochs", type=int, default=10)
	parser.add_argument("--training_range_config_file", type=str)

	# data parameters
	parser.add_argument("--storage_dir", default="storage")
	parser.add_argument("--dist_dir", type=str)
	parser.add_argument("--seed_cluster_path_idx", type=int)
	parser.add_argument("--figure_dir", type=str)
	parser.add_argument("--sample_k_paths_train", type=int)
	parser.add_argument("--sample_k_paths_val", type=int)

	# evaluation parameters
	parser.add_argument("--plot_trajectories", action=argparse.BooleanOptionalAction, default=False)

	args = parser.parse_args()
	params_to_save_path = Path(args.script_filepath).stem + "_params.json"

	eps = np.power(args.Ninter, -1+1e-4)
	small_dt = 0.1 * eps * eps
	dt = 1000 * small_dt

	if args.mode == 'generate':
		n_inter = None
		if args.n_dist is None:
			raise ValueError("n_dist cannot be None in the generation mode")
		if args.add_distribution_generation:
			if args.dist_config_file is None:
				dist_conf_full_path = os.path.join(args.samples_dir, "config_file.txt")
				if not os.path.exists(dist_conf_full_path):
					raise ValueError(f"you did not specify dist_config_file and the default value does not exist in {args.samples_dir}")
			else:
				dist_conf_full_path = os.path.join(args.samples_dir, args.dist_config_file)
				if not os.path.exists(dist_conf_full_path):
					raise ValueError(f"The only possible path for dist_config_file is inside samples_dir, "
					                 f"{dist_conf_full_path} does not exist")
			distribution_files = parse_dist_config_file(dist_conf_full_path)
			_, n_inter = generate_and_dump_distributions(distribution_files[0], distribution_files[1],
			                                            args.n_dist, args.samples_dir, n_samples=args.Ninter)

		alphas, multips, coefs, distnames = sample_interaction_parameters(args.N, args.alpha_range,
		                                                                    args.multiple_range, args.coefs_range,
		                                                                    init_dist_dir=args.samples_dir,
		                                                                    with_polynomial=args.with_polynomial,
		                                                                    max_n_coefs=args.max_n_coefs)
		alphas = [float(a) for a in alphas]
		multips = [float(m) for m in multips]
		if args.with_polynomial:
			coefs = [[float(c) for c in clist] for clist in coefs]
		else:
			coefs = None
		if n_inter is None:
			n_inter = args.Ninter
		script, storage_names = create_runner_sh_script(args.compiled_file, args.cpp_file_fpath,
		                                                alphas, multips, coefs,
		                                                add_compilation=args.add_compilation,
		                                                return_storage_fnames=True,
		                                                compilation_template=args.compilation_template_fpath,
		                                                storage_dir=args.storage_dir,
		                                                init_distributions=distnames, Ninter=n_inter,
		                                                )
		with open(args.script_filepath, "w") as f:
			f.write(script)

		update_json(params_to_save_path, {"alphas": alphas, "multipliers": multips,
		                                  "coefs": coefs, "storage_names": storage_names})

		makedirs(args.storage_dir, exist_ok=True)

		print(f"Generation ended successfully. Distributions can be found in: {args.samples_dir}\n"
		      f"Script that will run generation is in {args.script_filepath}\n")

	elif args.mode.startswith("train"):
		with open(params_to_save_path, "r") as f:
			params = json.load(f)

		xgrid = np.arange(args.lower_bd, args.upper_bd, args.grid_size)
		stop_at_k = np.inf if not args.smoke_run else 2
		validation_prototype = None
		validation_path = None

		train_pos_array = None
		val_pos_array = None
		train_time_indices = None
		val_time_indices = None
		if args.mode == 'train':
			# TODO rewrite it so that it uses the closest path mechanism for validation and maybe for finding other training paths as well
			# TODO get two validations - one that falls inside the trained range, so it is interpolation, the other one for extrapolation
			train_params, val_params = split_param_dict_train_val(params,
			                                                      output_max=(args.sample_k_paths_train, args.sample_k_paths_val))

			train_dataset_prototype, train_time_indices = build_whole_dataset_parametrised(xgrid, train_params,
			                                                                               stop_at_k=stop_at_k, Ninter=args.Ninter)
			val_dataset_prototype, train_time_indices = build_whole_dataset_parametrised(xgrid, val_params,
			                                                                             stop_at_k=stop_at_k, Ninter=args.Ninter)

			train_dataset = DatasetParametrised(xgrid, train_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			val_dataset = DatasetParametrised(xgrid, val_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			# TODO add dumping a model to some folder
			input_size = train_dataset.input_size
			M = train_time_indices.shape[0]
			dataset_prototype = train_dataset_prototype[:M]
			validation_prototype = val_dataset_prototype[:M]

		elif args.mode == 'train_in_cluster':
			if args.seed_cluster_path_idx is None:
				idx = np.random.randint(len(params['storage_names']))
			else:
				idx = args.seed_cluster_path_idx
			storage_filelist = [Path(params['storage_names'][i]).name for i in range(len(params['storage_names'])) if i != idx]
			cluster = build_a_cluster(params, storage_filelist,
			                          sample_k_paths=args.sample_k_paths_train,
			                          storage_dir=args.storage_dir, idx=idx)

			validation_path = find_closest_path_cluster(cluster, storage_filelist)

			train_dataset_prototype, train_time_indices, pos_arrays = build_whole_dataset_parametrised(xgrid, cluster,
			                                                                                           Ninter=args.Ninter,
			                                                                                           return_rarified_pos_arr=True)
			train_dataset = DatasetParametrised(xgrid, train_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			M = train_time_indices.shape[0]
			dataset_prototype = train_dataset_prototype[:M]
			train_pos_array = pos_arrays[0]
			input_size = train_dataset.input_size

		elif args.mode == 'train_in_out':
			if args.training_range_config_dile is None:
				raise ValueError("if mode = train_in_out, then you need to pass config file to training_range_config_dile")

			if args.different_dists:
				if args.dist_dir is None:
					raise ValueError("if different_dists is True, then you need to pass dist_dir")
				dist_fpath = os.path.join(args.dist_dir, "distribution_logfile.txt")
				params = update_param_file_with_dist_data(params, dist_fpath)

			range_params = parse_range_config_file(args.training_range_config_file)
			params_in, params_part_int, params_tot_out = filter_params(params, range_params)

			dataset_prototype, train_time_indices = build_whole_dataset_parametrised(xgrid, params_in,
			                                                                         Ninter=args.Ninter, stop_at_k=stop_at_k)
			validation_prototype, val_time_indices = build_whole_dataset_parametrised(xgrid, params_tot_out,
			                                                                          Ninter=args.Ninter, stop_at_k=stop_at_k)
		elif args.mode == 'train_one_path':
			trunc_params = {}
			for key, items in params.items():
				trunc_params[key] = [items[0]] # this can be randomised as well
			dataset_prototype, train_time_indices = build_whole_dataset_parametrised(xgrid, trunc_params,
			                                                                         Ninter=args.Ninter, stop_at_k=stop_at_k)
			dataset = DatasetParametrised(xgrid, dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			train_length = int(0.75*len(dataset))
			train_dataset, val_dataset = random_split(dataset, lengths=[train_length, len(dataset) - train_length])
			input_size = dataset.input_size

			validation_path = find_closest_path((trunc_params['alphas'][0], trunc_params['multipliers'][0]),
			                                    storage_filelist=os.listdir(args.storage_dir))
			M = train_time_indices.shape[0]
		else:
			raise ValueError("incorrect mode argument")

		if validation_prototype is None and validation_path is not None:
			validation_params = path_to_params(validation_path, return_dict=True, return_as_list=True)
			validation_params["storage_names"] = [os.path.join(args.storage_dir, validation_path)]
			print(validation_params)
			validation_prototype, val_time_indices, val_pos_array = build_whole_dataset_parametrised(xgrid, validation_params,
			                                                        Ninter=args.Ninter, stop_at_k=stop_at_k,
			                                                                          return_rarified_pos_arr=True)
			val_dataset = DatasetParametrised(xgrid, validation_prototype, max_polynomial_degree=args.max_n_coefs)
			val_pos_array = val_pos_array[0]

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
		print("Created training set of size {} and validation set of size {}".format(len(train_dataset),
		                                                                             len(val_dataset)))
		model, opt = get_model(input_size=input_size,
		                       hidden_layers=args.hidden_layers,
		                       learning_rate=args.learning_rate,
		                       weight_decay=args.weight_decay)

		if args.loss == 'mse':
			loss_fn = unweighted_mse
		else:
			loss_fn = time_sensitive_loss

		tidx = np.linspace(0, M-2, num=9, dtype=int)
		evaluator_kwargs = dict(network=model,
		                      xgrid=xgrid,
		                      time_indices=tidx,
		                      dt=dt, eps=eps)
		evaluator = Evaluator(proto_dataset=dataset_prototype, **evaluator_kwargs)

		train_over_epochs(args.no_epochs, train_loader, val_loader,
		                  model, opt, loss_fn, max_batches=args.max_batches)

		os.makedirs(args.figure_dir, exist_ok=True)
		evaluator.evaluate_grid(figure_fpath=os.path.join(args.figure_dir, "train.png"))
		if train_pos_array is not None:
			fpath = os.path.join(args.figure_dir, "train_path_evaluation.png")
			evaluator.evaluate_trajectory_exact(xinit=0, pos_arr=train_pos_array,
		                                    time_indices=train_time_indices, plot_fpath=fpath)

		if validation_prototype is not None:
			evaluator_on_unseen = Evaluator(proto_dataset=validation_prototype, **evaluator_kwargs)
			evaluator_on_unseen.evaluate_grid(figure_fpath=os.path.join(args.figure_dir, "validation.png"))
			if val_pos_array is not None:
				fpath = os.path.join(args.figure_dir, "val_path_evaluation.png")
				evaluator_on_unseen.evaluate_trajectory_exact(xinit=0, pos_arr=val_pos_array,
			                                                time_indices=val_time_indices, plot_fpath=fpath)
