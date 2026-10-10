import numpy as np
import time
import h5py
# import numba as nb

from helpers import build_progress_bar, inverse_power_law, ou_drift


# TODO turn to numba to speed it up, or maybe C++ directly, there's nothing complicated here
def euler_step(x, dt, sqdt, b, N):
	diffs = x[:, None] - x[None, :] # double check if it's actually faster than looping, you allocate a big matrix here
	update = dt * b(diffs).mean(axis=1) + np.random.normal(loc=0, scale=sqdt, size=N)
	return x + update


N = 150
alph = 0.4
eps = N**(-1+1e-4) # TODO check if it needs to be this drastic

dt = 0.1*eps**2
T = 1
sqdt = np.sqrt(2*dt)
x_0 = np.random.uniform(low=-1, high=1, size=N) # they all have to be iid otherwise interchangeability goes to trash
x = x_0.copy()
M = int(T/dt) + 1

pbar_length = 100
verbosity_step = int(M / pbar_length)
checkpoint_steps = int(M / verbosity_step)
intermediate = np.zeros((M, N))
j = 0

print(f"number of particles: {N}, number of steps M: {M}, time step: {dt}")
fname = f"ips_{N}_{alph}.hdf5"
t0 = time.perf_counter()
for i in range(M):
	if i % verbosity_step == 0:
		print(build_progress_bar(i, M), end="\r")
	intermediate[i, :] = x
	x = euler_step(x, dt, sqdt, inverse_power_law, N)

	# TODO add here dumping data in chunks because for moderate even number of particles it becomes unsustainable

t1 = time.perf_counter()

print(f"\nSolving SDE numerically took {(t1 - t0):.2f} seconds")
with h5py.File(fname, "w") as f:
	f.create_dataset("ips_steps", data=intermediate)