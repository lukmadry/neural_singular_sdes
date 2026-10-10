import h5py
import numpy as np
import argparse
import time

import json
from os import makedirs
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from nets import get_model
from helpers import update_json, build_progress_bar, function_factory
from evals import Evaluator

# TODO create two evaluators, one for training set and one for validation set. After each epoch do a check on these evaluators and dump the result somewehere
# TODO refactor so that this is a main file which provides access to a backend

def sample_interaction_parameters(N, max_n_coefs, alph_range, multip_range, coefs_range):
	"""

	:param N:
	:param alph_range:
	:param multip_range:
	:param coefs_range:
	:return:
	"""
	alphas = np.random.uniform(alph_range[0], alph_range[1], N)
	multips = np.random.uniform(multip_range[0], multip_range[1], N)
	n_coefs = np.random.randint(low=2, high=max_n_coefs, size=N) # low=2 is there because of Horner method in cpp file
	coefs = [[np.random.uniform(low=coefs_range[0], high=coefs_range[1]) for _ in range(k)] for k in n_coefs]
	return alphas, multips, coefs

def create_runner_sh_script(compilation_fpath, cpp_file_fpath, Ninter, alphas, multips, coefs,
                            return_storage_fnames=True, add_compilation=True,
                            compilation_template=None, storage_dir="storage"):
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
	for local_alph, local_multip, local_coefs in zip(alphas, multips, coefs):
		s = f"./{compilation_fpath} --N {Ninter} --a {local_alph} --m {local_multip} "
		storage_filename = (f"{storage_dir}/N_{Ninter}_alph_{local_alph}_mult_{local_multip}_coefs_" +
		                    "_".join([str(x) for x in local_coefs]) + ".hdf5")
		s += "--f " + storage_filename
		s +=  f" --p {len(local_coefs)}"
		for coef in local_coefs:
			s += f" {coef}"
		big_script += s + "\n"
		if return_storage_fnames:
			storages.append(storage_filename)
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
                                       time_indices, positions_arr):
	fn = function_factory(*fn_parameters)
	total_x = []
	for i, ti in enumerate(time_indices[1:], 1):
		vals = compute_drift_vec(xgrid, positions_arr[i, :], fn)
		total_x.append((fn_parameters, ti, vals, positions_arr[0, :]))
	return total_x

def build_whole_dataset_parametrised(xgrid, params_dict,
                                     Ninter=150, stop_at_k=np.inf):
	alphas, multips, coefs, data = params_dict['alphas'], params_dict['multipliers'], params_dict['coefs'], params_dict['storage_names']
	eps = 1/np.power(Ninter, 1-1e-4)
	dt = 0.1 * eps * eps # reproducing the computation of dt in the .cpp file
	M = int(1/dt) + 1
	time_indices = np.arange(1, M, 1000) * dt

	if stop_at_k < np.inf:
		time_indices = time_indices[:2*stop_at_k]

	all_datasets = []
	print("Progress at loading storage: ")
	for nb, (a, m, c, storage_fname) in enumerate(zip(alphas, multips, coefs, data)):
		# we will first check if Ninter matches the file

		if storage_fname.startswith("--f"):
			storage_fname = storage_fname.replace("--f ", "") # quick fix to a small bug in a previous iteration

		fname = Path(storage_fname).name
		ninter_idx = fname.index("N_")
		ninter = int(fname[ninter_idx:ninter_idx+7].split("_")[1])
		if ninter != Ninter:
			raise ValueError
		with h5py.File(storage_fname, "r") as f:
			positions_arr = f['drift_results'][:]

		positions_arr = positions_arr.reshape(-1, Ninter).astype(np.float32)
		part_dataset = build_part_of_dataset_parametrised(xgrid, (a, m, c, eps), time_indices, positions_arr)
		all_datasets.extend(part_dataset)

		if nb > stop_at_k: # mostly for testing
			break

		print(build_progress_bar(nb, len(data)), end="\r")

	print(build_progress_bar(len(data), len(data)), end="\n")
	return all_datasets

def split_param_dict_train_val(param_dict, train_size=0.75):
	idx = np.random.permutation(len(param_dict['alphas']))
	Ntrain = int(len(param_dict['alphas']) * train_size )

	train_dict, val_dict = {}, {}

	for tag in param_dict.keys():
		train_v = [param_dict[tag][i] for i in idx[:Ntrain]]
		val_v = [param_dict[tag][i] for i in idx[Ntrain:]]
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
		output = net(data)
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
	parser.add_argument('--mode', choices=['generate', 'train', 'train_one_path'])

	# only relevant for generation
	parser.add_argument("--alpha_range", nargs=2, type=float, default=(0, 0.49))
	parser.add_argument("--multiple_range", nargs=2, type=float, default=(0.5, 3))
	parser.add_argument("--coefs_range", nargs=2, type=float, default=(-0.1, 0.1))
	parser.add_argument("--max_n_coefs", type=int, default=5)


	parser.add_argument("--N", type=int)
	parser.add_argument("--Ninter", type=int, default=200)
	parser.add_argument("--script_filepath", type=str, required=True)
	parser.add_argument("--add_compilation", action=argparse.BooleanOptionalAction)
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
	parser.add_argument("--hidden_layers", type=int, nargs="+", default=[256, 256])
	parser.add_argument("--learning_rate", type=float, default=1e-2)
	parser.add_argument("--max_batches", type=int, default=10000)
	parser.add_argument("--smoke_run", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--loss", type=str, choices=['mse', 'time_weighting'], default='mse')
	parser.add_argument("--weight_decay", default=0.0, type=float)

	# data parameters
	parser.add_argument("--storage_dir", default="storage")
	parser.add_argument("--figure_fpath", type=str)
	parser.add_argument("--sample_k_paths", type=int)

	args = parser.parse_args()
	params_to_save_path = Path(args.script_filepath).stem + "_params.json"

	eps = np.power(args.Ninter, -1+1e-4)
	small_dt = 0.1 * eps * eps
	dt = 1000 * small_dt

	if args.mode == 'generate':
		alphas, multips, coefs = sample_interaction_parameters(args.N, args.max_n_coefs, args.alpha_range,
		                                                       args.multiple_range, args.coefs_range)
		alphas = [float(a) for a in alphas]
		multips = [float(m) for m in multips]
		coefs = [[float(c) for c in clist] for clist in coefs]
		script, storage_names = create_runner_sh_script(args.compiled_file, args.cpp_file_fpath,
		                                                args.Ninter, alphas, multips, coefs,
		                                                add_compilation=args.add_compilation,
		                                                return_storage_fnames=True,
		                                                compilation_template=args.compilation_template_fpath,
		                                                storage_dir=args.storage_dir
		                                                )
		with open(args.script_filepath, "w") as f:
			f.write(script)

		update_json(params_to_save_path, {"alphas": alphas, "multipliers": multips, "coefs": coefs, "storage_names": storage_names})

		makedirs(args.storage_dir, exist_ok=True)

	elif args.mode.startswith("train"):
		with open(params_to_save_path, "r") as f:
			params = json.load(f)

		xgrid = np.arange(args.lower_bd, args.upper_bd, args.grid_size)
		stop_at_k = np.inf if not args.smoke_run else 2
		if args.mode == 'train':
			train_params, val_params = split_param_dict_train_val(params)

			train_dataset_prototype = build_whole_dataset_parametrised(xgrid, train_params,
			                                                           stop_at_k=stop_at_k, Ninter=args.Ninter)
			val_dataset_prototype = build_whole_dataset_parametrised(xgrid, val_params,
			                                                         stop_at_k=stop_at_k, Ninter=args.Ninter)

			train_dataset = DatasetParametrised(xgrid, train_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			val_dataset = DatasetParametrised(xgrid, val_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			# TODO add dumping a model to some folder
			input_size = train_dataset.input_size
			dataset_prototype = train_dataset_prototype[:400] # TODO check if it's [:400] for one or sth else
			# TODO ideally you would evaluate at path from val_dataset...
		elif args.mode == 'train_one_path':
			trunc_params = {}
			for key, items in params.items():
				trunc_params[key] = [items[0]]

			dataset_prototype = build_whole_dataset_parametrised(xgrid, trunc_params,
			                                                     Ninter=args.Ninter, stop_at_k=stop_at_k)
			print('dataset prototype 00 ', dataset_prototype[0][0])
			dataset = DatasetParametrised(xgrid, dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			train_length = int(0.75*len(dataset))
			train_dataset, val_dataset = random_split(dataset, lengths=[train_length, len(dataset) - train_length])
			input_size = dataset.input_size
		else:
			raise NotImplementedError

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

		print("Created training set of size {} and validation set of size {}".format(len(train_dataset),
		                                                                             len(val_dataset)))
		model, opt = get_model(input_size=input_size,
		                       hidden_layers=args.hidden_layers,
		                       learning_rate=args.learning_rate,
		                       weight_decay=args.weight_decay)

		evaluator = Evaluator(network=model,
		                      xgrid=xgrid,
		                      time_indices=[1, 10, 75, 100, 150, 200, 250, 300],
		                      dt=dt, eps=eps,
		                      proto_dataset=dataset_prototype)

		if args.loss == 'mse':
			loss_fn = unweighted_mse
		else:
			loss_fn = time_sensitive_loss

		train_over_epochs(10, train_loader, val_loader,
		                  model, opt, loss_fn)

		evaluator.evaluate_grid(figure_fpath=args.figure_fpath)