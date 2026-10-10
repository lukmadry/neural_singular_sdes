import numpy as np
import matplotlib.pyplot as plt
import torch
from random import sample
from helpers import OneFunctionOnAGrid, none_sentinel

"""

a class that preloads a network, function parameters and allows us to evaluate performance 
of a network in a structured way, for every timepoint

"""
class Evaluator:
	def __init__(self, network, xgrid,
	             time_indices, dt, eps, pos_arr = None,
	             proto_dataset = None, fn_params = None):
		"""

		:param network:
		:param fn_params:
		:param xgrid:
		:param times:
		:param storage_directory:
		"""
		self.network = network
		self.fn_params = fn_params
		self.time_indices = np.asarray(time_indices, dtype=np.int32)
		self.xgrid = xgrid
		self.dt = dt
		self.eps = eps

		none_sentinel(pos_arr, proto_dataset)

		if proto_dataset is not None:
			self.proto_dataset = proto_dataset
			self.fn_params = self.proto_dataset[0][0]
		else:
			if fn_params is not None:
				raise ValueError
		self.fn_caller = OneFunctionOnAGrid(xgrid, *self.fn_params)


	# we just pass a sequence of times, because we already hold ground truths inside
	def _evaluate_grid_with_proto_dataset(self):
		"""

		:return:
		"""
		ts = self.dt * self.time_indices
		x_inputs = [self.fn_caller.create_input_with_time(t) for t in ts]
		y_inputs = [self.proto_dataset[t][2] for t in self.time_indices]
		preds = [self.network.forward(torch.as_tensor(x, dtype=torch.float32)).detach().numpy() for x in x_inputs]

		return (preds, y_inputs)

	def evaluate_grid(self, figure_fpath=None, sample_max_paths=16):
		preds, y_inputs = self._evaluate_grid_with_proto_dataset()
		errors = [np.linalg.norm(p-y) for (p, y) in zip(preds, y_inputs)]

		if figure_fpath is None:
			return self.time_indices, errors, preds, y_inputs

		if len(errors) > sample_max_paths:
			idx = sample(self.time_indices, sample_max_paths)
			preds = [preds[i] for i in idx]
			y_inputs = [y_inputs[i] for i in idx]
			errors = [errors[i] for i in idx]
		else:
			idx = self.time_indices
		print("saving the figure with plots")
		square_root_int = int(np.sqrt(len(errors)))
		addon = int((len(errors) - square_root_int ** 2) > 0)
		height = square_root_int + addon
		width = square_root_int + addon
		fig_stat, axs_stat = plt.subplots(height, width, figsize=(10, 10), sharex=True, sharey=False)
		fig_stat.suptitle(f"Comparing predictions with ground truths for parameters:\n "
		                  f"alpha: {self.fn_params[0]:.2f}, multip: {self.fn_params[1]:.2f}.\nOrange is prediction, red is ground truth")
		for i, tidx in enumerate(idx):
			ih = i // height
			iw = i % width
			axs_stat[ih, iw].plot(self.xgrid, preds[i], color='orange')
			axs_stat[ih, iw].plot(self.xgrid, y_inputs[i], color='red')
			axs_stat[ih, iw].set_title(f"Time: {self.dt*tidx:.2f}")
		fig_stat.savefig(figure_fpath)
		return idx, errors, preds, y_inputs

	def evaluate_trajectory(self):
		pass

	def load_state_dict(self, state_dict):
		self.network.load_state_dict(state_dict)



if __name__ == "__main__":
	pass