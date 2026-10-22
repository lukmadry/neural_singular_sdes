import unittest
import os
from glob import glob
import numpy as np
import h5py
import json

from helpers import find_closest_path, find_closest_path_cluster, build_a_cluster, filter_params
from train_nn_on_drifts import build_part_of_dataset_parametrised, build_whole_dataset_parametrised, DatasetParametrised

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


class TestEval(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()