import unittest
import os
from glob import glob
import numpy as np
import h5py
import json

from helpers import find_closest_path, find_closest_path_cluster, build_a_cluster, filter_params
from train_nn_on_drifts import (build_part_of_dataset_parametrised, build_whole_dataset_parametrised, DatasetParametrised,
                                sample_interaction_parameters, create_runner_sh_script, DatasetParametrisedHistogramNoPolynomial)
from helpers import parse_dist_config_file, generate_and_dump_distributions
from nets import get_model
from evals import Evaluator

class TestDatasetBuilding(unittest.TestCase):

	Ninter = 200
	storage_dir = "storage_200_ps"
	dx = 0.01
	xgrid = np.arange(-10, 10, dx)
	alph = 0.2
	multip = 1
	coefs = [0.05, -0.05]
	eps = 1 / np.power(Ninter, 1 - 1e-4)
	dt = 0.1 * eps * eps  # reproducing the computation of dt in the .cpp file
	M = int(1 / dt) + 1
	filenames = glob(os.path.join(storage_dir, f"N_{Ninter}_*.hdf5"))
	param_dict_fpath = f"smoke_run_N{Ninter}_params.json"

	# needs to be checked as creation of this dataset is handled in C++
	def test_dimensions(self):
		filenames = glob(os.path.join(self.storage_dir, f"*_N_{self.Ninter}_*.hdf5"))

		desired_shape = (self.M, self.Ninter) # note that this is transposed later on
		for fname in filenames:
			with h5py.File(fname) as f:
				pos_arr = f['drift_results'][:]
			actual_shape = pos_arr.shape
			self.assertEqual(desired_shape, actual_shape)

	def test_build_part_of_dataset(self):
		fname = self.filenames[0]

		print("Testing the following file: ", fname)
		with h5py.File(fname) as f:
			pos_arr = f['drift_results'][:]
		pos_arr = pos_arr.reshape(-1, self.Ninter)

		time_indices = np.arange(0, self.M, 1000) * self.dt
		self.assertEqual(len(time_indices), 400)
		fn_params = (self.alph, self.multip, self.coefs, self.eps)
		part_outcome = build_part_of_dataset_parametrised(self.xgrid,
						                                   fn_params,
						                                   time_indices[:3],
						                                   pos_arr)
		self.assertEqual(len(part_outcome), 2) # we pass three time_indices, but we do not count the first one
		for j, part in enumerate(part_outcome):
			self.assertEqual(len(part), 4)
			self.assertEqual(part[1], time_indices[j+1]) # we do not start from zero time !
			self.assertEqual(part[2].shape, self.xgrid.shape)
			self.assertEqual(part[3].shape, (self.Ninter,))

	def test_build_whole_dataset(self):
		with open(self.param_dict_fpath) as f:
			param_dict = json.load(f)

		all_dataset, _ = build_whole_dataset_parametrised(self.xgrid, param_dict, self.Ninter, stop_at_k=2)
		self.assertEqual(len(all_dataset), 3*4) # the cutoff for testing is sharp, i.e. > not >=
		self.all_dataset = all_dataset

	def test_torch_dataset_no_histogram(self):
		if not hasattr(self, 'all_dataset'):
			self.test_build_whole_dataset()

		dataset = DatasetParametrised(self.xgrid, self.all_dataset, 5, with_histogram=False)
		self.assertEqual(dataset.no_param, 7) # no histogram
		self.assertEqual(dataset.input_size, 9)
		self.assertEqual(dataset.length_of_one_fdataset, self.xgrid.shape[0])
		self.assertEqual(len(dataset), self.xgrid.shape[0] * 12)

		self.assertEqual(len(dataset[0][0]), 9)


class TestDatasetBuiltWithHist(unittest.TestCase):


	Ninter = 150
	storage_dir = "storage_narrow_init_dist"
	dx = 0.01
	xgrid = np.arange(-10, 10, dx)
	alph = 0.2
	multip = 1
	coefs = [0.05, -0.05]
	eps = 1 / np.power(Ninter, 1 - 1e-4)
	dt = 0.1 * eps * eps  # reproducing the computation of dt in the .cpp file
	M = int(1 / dt) + 1
	filenames = glob(os.path.join(storage_dir, f"N_{Ninter}_*.hdf5"))
	param_dict_fpath = f"smoke_run_N{Ninter}_params.json"


	def test_build_part_of_dataset_and_torch(self):
		# they are together because we use the part_outcome to create Torch dataset

		fname = self.filenames[0]

		print("Testing the following file: ", fname)
		with h5py.File(fname) as f:
			pos_arr = f['drift_results'][:]
		pos_arr = pos_arr.reshape(-1, self.Ninter)

		time_indices = np.arange(0, self.M, 1000) * self.dt
		self.assertEqual(len(time_indices), 225)
		fn_params = (self.alph, self.multip, None, self.eps)
		part_outcome = build_part_of_dataset_parametrised(self.xgrid,
						                                   fn_params,
						                                   time_indices[:3],
						                                   pos_arr,
						                                    with_init_dist=True)

		self.assertEqual(len(part_outcome), 2) # we pass three time_indices, but we do not count the first one
		for j, part in enumerate(part_outcome):
			self.assertEqual(len(part), 4)
			self.assertEqual(part[1], time_indices[j+1]) # we do not start from zero time !
			self.assertEqual(part[2].shape, self.xgrid.shape)
			self.assertEqual(part[3].shape, (29,))

		bins_edges = np.linspace(-3, 3, num=30)
		h, _ = np.histogram(pos_arr[0, :], bins=bins_edges, density=True)
		h /= h.sum()

		is_close = np.allclose(h, part_outcome[0][-1])
		self.assertTrue(is_close)
		self.part_outcome = part_outcome

		dataset = DatasetParametrisedHistogramNoPolynomial(xgrid=self.xgrid,
		                                                   all_datasets=self.part_outcome)

		self.assertEqual(dataset.input_size, 33)
		self.assertEqual(dataset.no_param, 2)
		u, y = dataset[0]
		self.assertEqual(u.shape[0], 33)


class TestTrajectoryFinder(unittest.TestCase):

	storage_list = ['N_200_alph_0.25_mult_1_coefs_0.5.hdf5',
	                'N_200_alph_0.30_mult_1_coefs_0.5.hdf5',
	                'N_200_alph_0.35_mult_1.25_coefs_0.5.hdf5',
	                'N_200_alph_0.4_mult_1.25_coefs_0.5.hdf5',
	                'N_200_alph_0.32_mult_1.25_coefs_0.5.hdf5',
	                ]

	fn_params_for_sec = (0.31, 1.1)
	fn_params_for_third = (0.34, 1.2)
	fn_params_for_first = (0.30, 1)

	alphas_empty_inter = [0.305, 0.31]
	mults_empty_inter = [1.05, 1.1]

	alphas_nonempty_inter = [0.30, 0.305, 0.31]
	multipliers_nonempty_inter = [1, 1.20, 1.21]

	alphas_nonempty_inter_sec = [0.30, 0.305, 0.303]
	multipliers_nonempty_inter_sec = [1, 1.05, 1.03]

	alphas_nonempty_inter_wrong = [0.30, 0.305]
	multipliers_nonempty_inter_wrong = [1, 1.05]

	params_for_inout = {"alphas": [0.3, 0.31, 0.32, 0.322, 0.324, 0.33, 0.331, 0.329, 0.34, 0.35, 0.345], # 11 in total
	                    "mults": [1, 1.1, 1.05, 1.03, 1.09, 1.08, 1.07, 1.04, 1.02, 1.11, 1.12]}

	params_ranges = {"alphas": [0.32, 0.3401], "mults": [1.019, 1.099]} # 6 alphas in, 7 for mults, 6 common for both, 4 totally out

	def test_onepath_closest_finder(self):

		cpath_sec = find_closest_path(self.fn_params_for_sec, self.storage_list)
		cpath_third = find_closest_path(self.fn_params_for_third, self.storage_list)
		cpath_first = find_closest_path(self.fn_params_for_first, self.storage_list) # should filter out the second one

		self.assertEqual(cpath_sec, self.storage_list[1])
		self.assertEqual(cpath_third, self.storage_list[2])
		self.assertEqual(cpath_first, self.storage_list[0])

	def test_cluster_finder(self):

		param_dict = {"alphas": self.alphas_empty_inter, "multipliers": self.mults_empty_inter}
		cpath = find_closest_path_cluster(param_dict, self.storage_list, assume_empty_intersection=True)
		cpath_two = find_closest_path_cluster(param_dict, self.storage_list, assume_empty_intersection=False)
		self.assertEqual(cpath, self.storage_list[1])
		self.assertEqual(cpath_two, self.storage_list[1])

		param_dict_nonempty_wrong = {"alphas": self.alphas_nonempty_inter_wrong,
		                             "multipliers": self.multipliers_nonempty_inter_wrong}
		cpath_ne = find_closest_path_cluster(param_dict_nonempty_wrong, self.storage_list, assume_empty_intersection=True)
		self.assertEqual(cpath_ne, self.storage_list[1])

		param_dict_nonempty = {"alphas": self.alphas_nonempty_inter, "multipliers": self.multipliers_nonempty_inter}
		cpath_ne_filter = find_closest_path_cluster(param_dict_nonempty, self.storage_list, assume_empty_intersection=False)
		self.assertEqual(cpath_ne_filter, self.storage_list[4])

		param_dict_nonempty_sec = {"alphas": self.alphas_nonempty_inter_sec, "multipliers": self.multipliers_nonempty_inter_sec}
		cpath_ne_filter = find_closest_path_cluster(param_dict_nonempty_sec, self.storage_list,
		                                            assume_empty_intersection=True)
		self.assertEqual(cpath_ne_filter, self.storage_list[1])

	def test_cluster_building(self):

		params = {"alphas": [0.33], "multipliers": [1.19],
		          "storage_names": 'N_200_alph_0.33_mult_1.19_coefs_0.5.hdf5'}
		st_list = self.storage_list + ['N_200_alph_0.33_mult_1.2_coefs_0.5.hdf5']

		cluster = build_a_cluster(params, st_list, 2)
		self.assertEqual(cluster['alphas'], [0.33, 0.33, 0.32])
		self.assertEqual(cluster['multipliers'], [1.19, 1.2, 1.25])

	def test_inout_filtering(self):

		params_in, params_partout, params_tot_out = filter_params(self.params_for_inout,
		                                                          self.params_ranges)

		self.assertEqual(len(params_in["alphas"]), 6)
		self.assertEqual(len(params_in['mults']), 6)

		self.assertEqual(len(params_tot_out["alphas"]), 4)
		self.assertEqual(len(params_tot_out["mults"]), 4)

		self.assertEqual(len(params_partout["alphas"]), 1)
		self.assertEqual(len(params_partout["mults"]), 1)

		self.assertEqual(params_in["alphas"], [0.322, 0.324, 0.33, 0.331, 0.329, 0.34])


class TestFilteringWithInitDist(unittest.TestCase):

	pass

class TestGenerationInitDist(unittest.TestCase):

	dist_config_file_full_path = "tests/init_dist_moving_gaussian/config_file.txt"
	init_dist_dir = "tests/init_dist_moving_gaussian"
	storage_dir = "tests/storage_trajs"
	fnames = [f"mixture_{i}.hdf5" for i in range(5)]
	N = 10
	alph_range = (0.3, 0.35)
	mult_range = (1, 1.5)
	alphas = [0.3, 0.32, 0.305, 0.315]
	multips = [1.1, 1.2, 1.15, 1.02]
	cpp_fpath = "cpp_simul/ips_sim_lookup_table_load_init_dist.cpp"
	compilation_path = "cpp_simul/sde_mkl_w_lt_and_load_id"

	target_s = ("./cpp_simul/sde_mkl_w_lt_and_load_id --a 0.3 --m 1.1 --distname {dist_one} "
	            "--f {storage_dir}/N_{inter_one}_alph_0.3_mult_1.1_init_dist_mixture_0.hdf5\n"
	            "./cpp_simul/sde_mkl_w_lt_and_load_id --a 0.32 --m 1.2 --distname {dist_two}"
	            " --f {storage_dir}/N_{inter_two}_alph_0.32_mult_1.2_init_dist_mixture_1.hdf5\n")

	dist_text_result = [["gauss -1 0.25"], ["gauss 1 0.25"]]

	def test_reading_config_file(self):
		dist_texts = parse_dist_config_file(self.dist_config_file_full_path)
		self.assertEqual(dist_texts, [["gauss -1 0.25"], ["gauss 1 0.25"]])

	def test_generating_dist_interpolated_mixts(self):

		filenames, dist_sizes = generate_and_dump_distributions(self.dist_text_result[0],
					                                            self.dist_text_result[1],
					                                            n_dist=5,
					                                            n_samples=400, samples_dir=self.init_dist_dir)

		with open(os.path.join(self.init_dist_dir, "distribution_logfile.txt"), "r") as f:
			logfile = f.readlines()
			for l in logfile:
				fn, nr = l.split("|||")[:2]
				nr = int(nr.strip("\n"))
				self.assertEqual(dist_sizes[fn], nr)

		self.assertEqual(len(filenames), 5)
		for f, testf in zip(filenames, self.fnames):
			self.assertEqual(f, os.path.join(self.init_dist_dir, testf))

		means = []
		for f in filenames:
			with h5py.File(f, "r") as f:
				dist = f['init_distribution'][:]
			means.append(np.mean(dist))

		means = np.array(means)
		self.assertTrue(np.all(np.diff(means) > 0))

		for key, val in dist_sizes.items():
			self.assertTrue(val > 300)

	def test_sample_interaction_params_with_dists(self):

		alphas, multips, coefs, distnames = sample_interaction_parameters(self.N, self.alph_range, self.mult_range,
										                              init_dist_dir=self.init_dist_dir,
										                              with_polynomial=False)

		self.assertIsNotNone(distnames)
		self.assertIsNone(coefs)

		for d in distnames:
			self.assertTrue(os.path.exists(d))

		_, dist_sizes = generate_and_dump_distributions(self.dist_text_result[0],
		                                                        self.dist_text_result[1],
		                                                        n_dist=5,
		                                                        n_samples=100, samples_dir=self.init_dist_dir)

		mixtures = [os.path.join(self.init_dist_dir, "mixture_0.hdf5"), os.path.join(self.init_dist_dir, "mixture_1.hdf5")]
		ninters = [str(dist_sizes[x]) for x in mixtures]
		s, snames = create_runner_sh_script(self.compilation_path, self.cpp_fpath,
		                                    alphas=self.alphas[:2], multips=self.multips[:2],
		                                    add_compilation=False, storage_dir=self.storage_dir,
		                                    Ninter=dist_sizes, init_distributions=mixtures)

		target_string = (self.target_s.replace("{dist_one}", mixtures[0]).replace("{dist_two}", mixtures[1])
		                 .replace("{storage_dir}", self.storage_dir).replace("{inter_one}", ninters[0]).replace("{inter_two}", ninters[1]))

		self.assertEqual(s, target_string)


class TestEval(unittest.TestCase):

	fn_params_nohist = (0.35, 1, (0.2, -0.3, 3, 0, 0), 1e-6)
	fn_params_no_polynom = (0.35, 1, None, 1e-6)
	net_nohist, _ = get_model(input_size=9, hidden_layers=[128, 128], learning_rate=1e-2)
	net_hist, _ = get_model(input_size=4+29, hidden_layers=[128, 128], learning_rate=1e-2)
	xgrid = np.linspace(-5, 5, num=500)
	Ninter_nohist = 300
	eps_nohist = 1 / np.power(Ninter_nohist, 1 - 1e-4)
	dt_nohist = 0.1 * eps_nohist * eps_nohist
	nsteps_nohist = int(1/dt_nohist) + 1

	Ninter_hist = 150
	eps_hist = 1 / np.power(Ninter_hist, 1 - 1e-4)
	dt_hist = 0.1 * eps_hist * eps_hist
	nsteps_hist = int(1/dt_hist) + 1

	storage_dir_no_hists = "storage_300_small_range"
	storage_dir_hists = "storage_narrow_init_dist"

	def test_eval_nohist(self):

		trajectories = glob(os.path.join(self.storage_dir_no_hists, "*.hdf5"))

		with h5py.File(trajectories[0], "r") as f:
			pos_arr = f['drift_results'][:]

		pos_arr = pos_arr.reshape(-1, self.Ninter_nohist)
		time_skeleton = np.arange(0, self.nsteps_nohist, 1000)
		time_indices = time_skeleton * self.dt_nohist

		M = pos_arr.shape[0]
		tidx = np.linspace(0, M-2, num=9)
		with self.assertRaises(TypeError): # no fn_params for evaluator
			evaluator = Evaluator(network=self.net_nohist, xgrid=self.xgrid,
			                      time_indices=tidx, dt=self.dt_nohist, eps=self.eps_nohist, pos_arr=pos_arr,
			                      with_histogram=False)


		evaluator = Evaluator(network=self.net_nohist, xgrid=self.xgrid,
		                      time_indices=tidx, dt=self.dt_nohist, eps=self.eps_nohist, pos_arr=pos_arr,
		                      with_histogram=False, fn_params=self.fn_params_nohist)
		# we don't care if these are not real parameters for this array
		self.assertIsNotNone(evaluator.fn_caller.coefs)

		with self.assertRaises(TypeError):
			evaluator._evaluate_grid_with_proto_dataset() # you need protodataset to have ground truths

		total_x = build_part_of_dataset_parametrised(self.xgrid, self.fn_params_nohist,
		                                             time_indices=time_indices, positions_arr=pos_arr)
		self.assertEqual(len(total_x), 898) # -1 because we start from index 1
		evaluator_proto = Evaluator(network=self.net_nohist, xgrid=self.xgrid,
		                      time_indices=tidx, dt=self.dt_nohist, eps=self.eps_nohist, proto_dataset=total_x,
		                      with_histogram=False, fn_params=self.fn_params_nohist)

		self.assertEqual(evaluator_proto.fn_caller.input_size, 9)
		xinput_test = evaluator_proto.fn_caller.create_input_with_time(0.001)
		self.assertEqual(xinput_test.shape, (500, 9))

		preds, y_inputs = evaluator_proto._evaluate_grid_with_proto_dataset()
		self.assertEqual(len(preds), 9) # this is only for presentation
		with self.assertRaises(ValueError): # sentinel activated
			Evaluator(network=self.net_nohist, xgrid=self.xgrid,
		                      time_indices=tidx, dt=self.dt_nohist, eps=self.eps_nohist, proto_dataset=total_x,
		                      with_histogram=False, fn_params=self.fn_params_nohist, pos_arr=pos_arr)

		real_params = total_x[0][0]
		self.assertEqual(evaluator_proto.fn_params, real_params) # fn_params_nohist got overridden


	def test_eval_creation_hist(self):

		trajectories = glob(os.path.join(self.storage_dir_hists, "*.hdf5"))

		with h5py.File(trajectories[0], "r") as f:
			pos_arr = f['drift_results'][:]

		pos_arr = pos_arr.reshape(-1, self.Ninter_hist)
		time_skeleton = np.arange(0, self.nsteps_hist, 1000)
		time_indices = time_skeleton * self.dt_hist

		total_x = build_part_of_dataset_parametrised(self.xgrid, self.fn_params_no_polynom,
		                                             time_indices=time_indices, positions_arr=pos_arr,
		                                             with_init_dist=True)

		self.assertEqual(len(total_x), 224)
		evaluator_proto = Evaluator(network=self.net_hist, xgrid=self.xgrid, time_indices=time_indices,
		                            dt=self.dt_hist, with_histogram=True, proto_dataset=total_x, eps=self.eps_hist)
		self.assertIsNone(evaluator_proto.fn_caller.coefs)
		self.assertEqual(evaluator_proto.fn_caller.input_size, 33)
		preds, y_inputs = evaluator_proto._evaluate_grid_with_proto_dataset()
		self.assertEqual(len(preds), 225) # the same as time_indices

		bins_edges = np.linspace(-3, 3, num=30)
		h = np.histogram(pos_arr[0, :], bins=bins_edges, density=True)[0]
		h /= h.sum()
		if_close = np.allclose(evaluator_proto.fn_caller.hist, h)
		self.assertTrue(if_close)



if __name__ == '__main__':
	unittest.main()