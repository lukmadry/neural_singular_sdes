import numpy as np

def train_epoch(model, opt, xs, ys, batch_size, nb_batches, loss_fn):

	losses = []
	for i in range(nb_batches):
		xbatch = xs[i*batch_size:(i+1)*batch_size]
		ybatch = ys[i*batch_size:(i+1)*batch_size]

		ybatch_pred = model(xbatch)
		loss = loss_fn(ybatch, ybatch_pred)
		losses.append(loss.item())
		#optimizer
		opt.zero_grad()
		loss.backward()
		opt.step()

	return np.mean(losses)

def validation_epoch(model, xs, ys, batch_size, nb_batches, loss_fn):

	losses = []
	for i in range(nb_batches):
		xbatch = xs[i*batch_size:(i+1)*batch_size]
		ybatch = ys[i*batch_size:(i+1)*batch_size]

		ybatch_pred = model(xbatch)
		loss = loss_fn(ybatch, ybatch_pred)
		losses.append(loss.item())

	return np.mean(losses)

def build_progress_bar(i, M, length=50, p_sign='@'):
	prop = int(float(i)/float(M)* length)
	s = "[" + p_sign * prop + '_' * (length - prop) + "]" + f" {100*float(i/M):.2f}%"
	return s

def inverse_power_law(x, eps=1e-6, p = 0.5):
	return np.sign(x)* 1/(eps+np.abs(x)**p)

def ou_drift(x):
	return -x
