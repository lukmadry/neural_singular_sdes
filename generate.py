import itertools
import os

import numpy as np


def sample_interaction_parameters(N, alph_range, multip_range, coefs_range=None,
                                  init_dist_dir=None, with_polynomial=False, max_n_coefs=None, ret_full_paths=True):
	"""
	Samples parameters for an interaction function and initial distribution

	Parameters
	--------
	N : int
		number of parameters that are going to be sampled
	alph_range : list or a tuple of size two, for range of \alpha to be sampled
	multip_range : same as above for multiplier
	coefs_range : same but for taylor coefficients
	init_dist_dir : str, directory where possible initial distributions are stored. Distributions have to be in .hdf5 format
	with_polynomial : bool, if True then the function will have polynomial factor. If False, then coefs_range will be ignored
	max_n_coefs : maximal degree of polynomial if applicable
	ret_full_paths : bool, if True filenames of distributions will be returned with a directory

	Returns
	-------
	alphas, multips, coefs, dist_files

	alphas : np.ndarray of size N of \alpha in specified range
	multips : the same for multipliers
	coefs : list of lists of size max_n_coefs if with_polynomial = True, else None
	dist_files : list of initial distributions files sampled from init_dist_dir of size N
	"""
	alphas = np.random.uniform(alph_range[0], alph_range[1], N)
	multips = np.random.uniform(multip_range[0], multip_range[1], N)

	if with_polynomial:
		if coefs_range is None and max_n_coefs is None:
			raise ValueError("if with_polynomial == True, then you need to pass range of coefs and their max nr")
		n_coefs = np.random.randint(low=2, high=max_n_coefs,
		                            size=N)  # low=2 is there because of Horner method in cpp file
		coefs = [[np.random.uniform(low=coefs_range[0], high=coefs_range[1]) for _ in range(k)] for k in n_coefs]
	else:
		coefs = None

	if init_dist_dir is not None:
		all_files = [x for x in os.listdir(init_dist_dir) if x.endswith(".hdf5")]
		dist_idx = np.random.choice(len(all_files), size=N, replace=True)
		dist_files = [all_files[i] for i in dist_idx]
		if ret_full_paths:
			dist_files = [os.path.join(init_dist_dir, x) for x in dist_files]
	else:
		dist_files = None
	return alphas, multips, coefs, dist_files


def create_runner_sh_script(compilation_fpath, cpp_file_fpath, alphas, multips, coefs=None,
                            return_storage_fnames=True, add_compilation=True, Ninter=None,
                            compilation_template=None, storage_dir="storage", init_distributions=None,
                            add_verification_script=False):
	"""
	Create a .sh script that will generate McKean-Vlasov trajectories and store them in specified folder.

	:param compilation_fpath:
	:param cpp_file_fpath:
	:param alphas:
	:param multips:
	:param coefs:
	:param return_storage_fnames:
	:param add_compilation:
	:param Ninter: int or a dict str -> int, where str is a filepath to an initial_distribution and int is
	    a corresponding Ninter
	:param compilation_template:
	:param storage_dir:
	:param init_distributions:
	:return:
	"""
	big_script = ""
	if add_compilation:
		if compilation_template is None or compilation_fpath is None:
			raise ValueError("if you want to add a compilation script, need to add template (sh script)")
		with open(compilation_template, "r") as f:
			temp = f.read()
		big_script += temp.replace("{FILENAME}", cpp_file_fpath).replace("{OUTPUT_NAME}", compilation_fpath)
		big_script += '\nprintf "finished compilation"\n'
	if return_storage_fnames:
		storages = []

	if Ninter is None:
		raise ValueError("Ninter has to be either int or a dict of ints")
	elif not isinstance(Ninter, (int, dict)):
		raise TypeError("Ninter has to be either int or a dict of ints")

	dist_name_iterator = init_distributions if init_distributions is not None else itertools.repeat(None)
	if init_distributions is None:
		ninter_iterator = itertools.repeat(Ninter)
	else:
		ninter_iterator = [Ninter[x]['size'] for x in dist_name_iterator]
	if add_verification_script and init_distributions is not None:
		s = f"{compilation_fpath} --a 0.35 --m 1 --distname test_init.hdf5 --f verification_run.hdf5 --max_iter 1000"
		s += "\npython3 verify_starting_dists.py --dist test_init.hdf5 --run verification_run.hdf5 --N 150 --info_txt passing_info.txt\n"
		s += f'if grep -q "FAIL" "passing_info.txt"; then echo "FAILED TO PROPERLY LOAD A DISTRIBUTION" exit 1; fi'
		s += "\n"
		s += "rm passing_info.txt verification_run.hdf5"
		s += "\n"
		big_script += s

	if coefs is not None:
		for loc in zip(alphas, multips, coefs, dist_name_iterator, ninter_iterator):
			local_alph, local_multip, local_coefs, local_distname, local_ninter = loc
			s = f"./{compilation_fpath} --a {local_alph} --m {local_multip} --distname {local_distname} "
			storage_filename = (f"{storage_dir}/N_{local_ninter}_alph_{local_alph}_mult_{local_multip}_coefs_" +
			                    "_".join([str(x) for x in local_coefs]) + ".hdf5")
			s += "--f " + storage_filename
			s +=  f" --p {len(local_coefs)}"
			if local_distname is None:
				s += f"--N {Ninter}"
			for coef in local_coefs:
				s += f" {coef}"
			big_script += s + "\n"
			if return_storage_fnames:
				storages.append(storage_filename)
	else:
		for loc in zip(alphas, multips, dist_name_iterator, ninter_iterator):
			local_alph, local_multip, local_distname, local_ninter = loc
			s = f"./{compilation_fpath} --a {local_alph} --m {local_multip} --distname {local_distname} "
			if local_distname.endswith(".hdf5"):
				local_distname = local_distname.strip(".hdf5")
				local_distname = local_distname.split("/")[-1]
			storage_filename = (f"{storage_dir}/N_{local_ninter}_alph_{local_alph}_mult_{local_multip}_" +
			                    f"init_dist_{local_distname}.hdf5")
			s += "--f " + storage_filename
			if return_storage_fnames:
				storages.append(storage_filename)
			big_script += s + "\n"
	if return_storage_fnames:
		return big_script, storages
	return big_script
