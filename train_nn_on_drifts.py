import os
from functools import partial

import numpy as np
import argparse
import torch
import json

from os import makedirs
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from src.datasets import build_whole_dataset_parametrised, split_param_dict_train_val, DatasetParametrised, \
	DatasetParametrisedHistogramNoPolynomial
from src.generate import sample_interaction_parameters, create_runner_sh_script
from src.nets import get_model
from src.helpers import update_json, path_to_params, find_closest_path, \
	find_closest_path_cluster, build_a_cluster, parse_range_config_file, filter_params
from src.helpers import parse_dist_config_file, generate_and_dump_distributions, update_param_file_with_dist_data
from src.evals import Evaluator
from src.training import train_over_epochs, time_sensitive_loss, unweighted_mse

# think about how to best parametrise distributions
# in the directory you should have some parameters of the distribution. Let us maybe first try moving gaussian
# so that it is easy to parametrise and do interpolation/extrapolation thing later
# or just some shape that's moving from left to right so that we can parametrise it with a single number


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
	parser.add_argument("--overwrite_json", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--add_verification_script", action=argparse.BooleanOptionalAction, default=True)

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
	parser.add_argument("--different_dists", action=argparse.BooleanOptionalAction, default=False)

	# data parameters
	parser.add_argument("--storage_dir")
	parser.add_argument("--dist_dir", type=str)
	parser.add_argument("--seed_cluster_path_idx", type=int)
	parser.add_argument("--experiment_data_dir", type=str)
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
		                                                add_verification_script=args.add_verification_script,
		                                                )
		with open(args.script_filepath, "w") as f:
			f.write(script)

		update_json(params_to_save_path, {"alphas": alphas, "multipliers": multips,
		                                  "coefs": coefs, "storage_names": storage_names},
		            replace_old_json=args.overwrite_json)

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

		if args.experiment_data_dir is None:
			raise ValueError("experiment_data_dir cannot be None in the training mode")

		if args.mode == 'train':
			# TODO rewrite it so that it uses the closest path mechanism for validation and maybe for finding other training paths as well
			# TODO get two validations - one that falls inside the trained range, so it is interpolation, the other one for extrapolation
			train_params, val_params = split_param_dict_train_val(params,
			                                                      output_max=(args.sample_k_paths_train, args.sample_k_paths_val))

			train_dataset_prototype, train_time_indices = build_whole_dataset_parametrised(xgrid, train_params,
			                                                                               stop_at_k=stop_at_k, Ninter=args.Ninter)
			val_dataset_prototype, val_time_indices = build_whole_dataset_parametrised(xgrid, val_params,
			                                                                           stop_at_k=stop_at_k, Ninter=args.Ninter)

			train_dataset = DatasetParametrised(xgrid, train_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			val_dataset = DatasetParametrised(xgrid, val_dataset_prototype, max_polynomial_degree=args.max_n_coefs)
			# TODO add dumping a model to some folder
			input_size = train_dataset.input_size
			M = train_time_indices.shape[0]
			dataset_prototype = train_dataset_prototype[:M]
			validation_prototype = val_dataset_prototype[:M]

		elif args.mode == 'train_in_cluster':
			if args.different_dists:
				raise NotImplementedError("training on varying distributions not supported for clusters")

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
			if args.training_range_config_file is None:
				raise ValueError("if mode = train_in_out, then you need to pass config file to training_range_config_dile")

			if args.different_dists:
				if args.dist_dir is None:
					raise ValueError("if different_dists is True, then you need to pass dist_dir")
				dist_fpath = os.path.join(args.dist_dir, "distribution_logfile.txt")
				params = update_param_file_with_dist_data(params, dist_fpath)

			range_params = parse_range_config_file(args.training_range_config_file)
			params_in, params_part_int, params_tot_out, params_indices_in, params_indices_partially_out, params_indices_tot_out = (
				filter_params(params, range_params, return_indices=True))
			train_dataset_prototype, train_time_indices, pos_arrays = build_whole_dataset_parametrised(xgrid, params_in,
			                                                                                           Ninter=args.Ninter,
			                                                                                           stop_at_k=stop_at_k,
			                                                                                           with_init_dist=args.different_dists,
			                                                                                           return_rarified_pos_arr=True)
			train_pos_array = pos_arrays[0]
			M = train_time_indices.shape[0]
			validation_prototype, val_time_indices, pos_arrays = build_whole_dataset_parametrised(xgrid, params_tot_out,
			                                                                                      Ninter=args.Ninter,
			                                                                                      stop_at_k=stop_at_k,
			                                                                                      with_init_dist=args.different_dists,
			                                                                                      return_rarified_pos_arr=True)
			val_pos_array = pos_arrays[0]

			if args.different_dists:
				train_dataset = DatasetParametrisedHistogramNoPolynomial(xgrid, train_dataset_prototype)
				val_dataset = DatasetParametrisedHistogramNoPolynomial(xgrid, validation_prototype)
			else:
				train_dataset = DatasetParametrised(xgrid, train_dataset_prototype, with_histogram=False)
				val_dataset = DatasetParametrised(xgrid, validation_prototype, with_histogram=False)
			input_size = train_dataset.input_size
			dataset_prototype = train_dataset_prototype[:M]
		elif args.mode == 'train_one_path':

			if args.different_dists:
				raise NotImplementedError("training on varying distributions not supported for one path training")

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
		print("Created model")
		if args.loss == 'mse':
			loss_fn = unweighted_mse
			print("Loss: mse")
		elif args.loss != "mse" and not args.different_dists:
			loss_fn = time_sensitive_loss
			print("Loss: time_sensitive_loss with time_axis=7")
		elif args.loss != "mse" and args.different_dists:
			loss_fn = partial(time_sensitive_loss, time_axis=2)
			print("Loss: time_sensitive_loss with time_axis=2")
		else:
			# IDE complained, actually we never reach here
			raise ValueError

		tidx = np.linspace(0, M-2, num=9, dtype=int)
		evaluator_kwargs = dict(network=model,
		                      xgrid=xgrid,
		                      time_indices=tidx,
		                      dt=dt, eps=eps)
		evaluator = Evaluator(proto_dataset=dataset_prototype, **evaluator_kwargs)

		train_over_epochs(args.no_epochs, train_loader, val_loader,
		                  model, opt, loss_fn, max_batches=args.max_batches)

		figure_dir = os.path.join(args.experiment_data_dir, "figures")
		os.makedirs(figure_dir, exist_ok=True)

		with open(os.path.join(args.experiment_data_dir, "kwargs.json"), "w") as f:
			json.dump(vars(args), f)
		model_state_dict = os.path.join(args.experiment_data_dir, "model_data.pth")
		torch.save(model.state_dict(), model_state_dict)
		indices_json_path = os.path.join(args.experiment_data_dir, "indices.json")
		if args.mode == 'train_in_out':
			with open(indices_json_path, "w") as f:
				# noinspection unbound-local-variable
				json.dump({"in": params_indices_in, "pout": params_indices_partially_out, "totout": params_indices_tot_out}, f)

		evaluator.evaluate_grid(figure_fpath=os.path.join(figure_dir, "train.png"))
		if train_pos_array is not None:
			fpath = os.path.join(figure_dir, "train_path_evaluation.png")
			evaluator.evaluate_trajectory_exact(xinit=0, pos_arr=train_pos_array,
		                                        time_indices=train_time_indices, plot_fpath=fpath)

		if validation_prototype is not None:
			evaluator_on_unseen = Evaluator(proto_dataset=validation_prototype, **evaluator_kwargs)
			evaluator_on_unseen.evaluate_grid(figure_fpath=os.path.join(figure_dir, "validation.png"))
			if val_pos_array is not None:
				fpath = os.path.join(figure_dir, "val_path_evaluation.png")
				evaluator_on_unseen.evaluate_trajectory_exact(xinit=0, pos_arr=val_pos_array,
			                                                time_indices=val_time_indices, plot_fpath=fpath)
