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
	             proto_dataset = None, fn_params = None, with_histogram = True, bins = 30):
		"""

		network - accepts inputs as:
		(\alpha, m, polynomial coefficients*, time, x)

		*currently these are essentially zero (21/10)

		fn_params = (alpha, m, polynomial coefficients, eps)

		proto_dataset - see train_nn_on_drifts.build_whole_dataset for the construction, but it contains:
		a list of training points, M for every combination of parameters of function f
		every element of a list is a tuple of:
		(parameters of a function, time t, j -> f(x - X^j_t) for x in xgrid, positions at time zero)

		TODO add validation of proto_dataset
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
			self.proto_dataset = None

		if with_histogram:
			if proto_dataset is None:
				raise ValueError("if with_histogram is True then you need to pass proto_dataset")
			self.fn_caller = OneFunctionOnAGrid(xgrid, *self.fn_params,
			                                    with_histogram=with_histogram,
			                                    histogram=proto_dataset[0][-1])
		else:
			self.fn_caller = OneFunctionOnAGrid(xgrid, *self.fn_params)

	# we just pass a sequence of times, because we already hold ground truths inside
	def _evaluate_grid_with_proto_dataset(self):
		"""

		:return:
		"""
		ts = self.dt * self.time_indices
		x_inputs = [self.fn_caller.create_input_with_time(t) for t in ts]
		y_inputs = [self.proto_dataset[t][2] for t in self.time_indices]
		print(x_inputs[0].shape)
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

	def evaluate_trajectory_exact(self, xinit, pos_arr, time_indices=None,
	                              noise_increments=None, with_plots=True, plot_fpath=None):

		if with_plots and plot_fpath is None:
			raise ValueError("if you want to plot the results, you need to supply the path of the .png that will be saved")
		tidx = time_indices if time_indices is not None else self.time_indices
		if noise_increments is None:
			noise_increments = np.random.randn(len(tidx)) * np.sqrt(self.dt)
		x = xinit
		neural_x = xinit

		exact_trajectory = [x]
		neural_trajectory = [neural_x]

		for t in range(len(tidx)): # you should be able to just pass dt * 1000 or something like this
			upd = self.fn_caller.apply_fn_pos_arr(x, t, pos_arr) * self.dt + noise_increments[t]
			x += upd
			exact_trajectory.append(x)

			xinput = self.fn_caller.create_input_with_time_and_pos(tidx[t], neural_x)
			drift = self.network.forward(torch.as_tensor(xinput, dtype=torch.float32)).detach().numpy().flatten()[0]
			neural_upd = drift * self.dt + noise_increments[t]
			neural_x += neural_upd
			neural_trajectory.append(neural_x)

		error = np.linalg.norm(np.array(exact_trajectory) - np.array(neural_trajectory))
		if with_plots:
			self.plot_trajectories(exact_trajectory, neural_trajectory, plot_fpath)
		return exact_trajectory, neural_trajectory, error

	def plot_trajectories(self, exact_trajectory, neural_trajectory, plot_fpath):
		fig = plt.figure(figsize=(10, 10))
		plt.plot(exact_trajectory, label='Exact', color='green')
		plt.plot(neural_trajectory, label='Neural', color='red')
		plt.legend()
		plt.savefig(plot_fpath)

	def evaluate_trajectory_interpolation(self, noise_increments, time_indices=None):
		pass

	def load_state_dict(self, state_dict):
		self.network.load_state_dict(state_dict)



if __name__ == "__main__":
	pass