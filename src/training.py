import time

import numpy as np
import torch

from src.helpers import build_progress_bar


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
