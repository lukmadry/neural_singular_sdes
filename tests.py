import unittest
import os
from glob import glob
import numpy as np
import h5py
import json

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

		all_dataset = build_whole_dataset_parametrised(self.xgrid, param_dict, self.Ninter, stop_at_k=2)
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

class TestEval(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()