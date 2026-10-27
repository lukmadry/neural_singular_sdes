import h5py
import numpy as np
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dist")
	parser.add_argument("--run")
	parser.add_argument("--N", type=int)
	parser.add_argument("--info_txt")

	args = parser.parse_args()

	with h5py.File(args.dist, "r") as f:
		init_dist = f['init_distribution'][:]

	with h5py.File(args.run, "r") as f:
		pos_arr = f['drift_results'][:]

	pos_arr = pos_arr.reshape(-1, args.N)
	close = np.allclose(init_dist, pos_arr[0, :])
	with open(args.info_txt, "w") as f:
		f.write("SUCCESS" if close else "FAIL")