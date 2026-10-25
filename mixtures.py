import numpy as np
import os
import h5py
from scipy.stats import norm, multivariate_normal, uniform, pareto
from numbers import Number
from helpers import interpolate_pdfs, sample_full_dist

# TODO I realised when this was already well underway that scipy stats has an interface for mixtures, which would be
# probably cleaner to use. the advantage of my version is that it handles automatically flipping pareto, which is not so clear

# this is a file copied from another project of mine, so there's more stuff than directly needed.
# in our case multivar is unncessary

# base class that will allow us to easily add distributions together if needed
def extend_or_concat(arr_a, arr_b):
	if isinstance(arr_a, list):
		arr_a.extend(arr_b)
		return arr_a
	elif isinstance(arr_a, np.ndarray):
		return np.concatenate([arr_a, arr_b], axis=0)
	else:
		TypeError

class Mixture:
	def __init__(self, centers=None, weights=None, random_state=None):
		if centers is not None:
			if isinstance(centers, Number):
				centers = [centers]

			self.centers = np.array(centers)
		else:
			self.centers = []
		self.N = len(self.centers)
		if random_state is not None:
			self.seed = random_state
		else:
			self.seed = np.random.default_rng()

		self.distributions = []
		if weights is not None:
			if len(weights) != self.N:
				raise ValueError(f"length of weights parameter should be equal to centers. Centers are: {self.centers}, weights: {weights}")
			self.weights = weights
		else:
			self.weights = [1]*self.N

		self.multivar = False # by default, we're dealing with one-dim objects
		self._flips = [False]*self.N # in case we need to artifially flip the distribution. It will be flipped around center
		if any(self._flips) and self.multivar:
			raise NotImplementedError("flipping the distribution is not implemented for multivariate")

	def get_flips(self):
		return self._flips

	def pdf(self, grid):
		"""
		grid: if multivar=False then just a 1D array. if multivar=True then an array in shape (N, 2)
		returns an array in the same shape as input
		"""
		weights = self.weights/np.sum(self.weights)
		if not self.multivar:
			final_density = np.zeros_like(grid)
			for i in range(self.N):
				if self._flips[i]:
					new_grid = 2*self.centers[i] - grid
					temp = weights[i]*self.distributions[i].pdf(new_grid)
				else:
					temp = weights[i]*self.distributions[i].pdf(grid)
				final_density += temp
			return final_density
		else:
			final_density = np.zeros(grid.shape[:-1]) # maybe we can parametrize it or leave more choice, idk now
			for i in range(self.N):
				final_density += weights[i]*self.distributions[i].pdf(grid)
			return final_density

	def rvs(self, no_samples):

		weights = self.weights/np.sum(self.weights)
		idx = self.seed.choice(a=self.N, size=no_samples, p=weights)
		samples = []
		state_dict = self.seed.bit_generator.state
		if not self.multivar:
			for i in idx:
				samples.append(self.distributions[i].rvs(random_state=self.seed))
		else:
			for i in idx:
				samples.append(self.distributions[i].rvs(random_state=self.seed).flatten())
		return np.array(samples)

	def __len__(self):
		return self.N

	def __getitem__(self, item):
		return self.distributions[item]

	def extend(self, distribution):
		if not isinstance(distribution, Mixture):
			raise ValueError("Distributions can only be extended with Distributions")
		self.distributions.extend(distribution.distributions)
		self.weights = extend_or_concat(self.weights, distribution.weights)
		#self.weights.extend(distribution.weights)
		self.centers = np.concatenate([self.centers, distribution.centers], axis=0)
		self.N += len(distribution)
		self._flips += distribution.get_flips()

# TODO not sure if this is strictly necessary, if anything you do similar thing with width, so maybe this can be made uniform
class ScaleArgMixin:

	def set_scale(self, scales, allow_neg=True):
		if isinstance(scales, Number):
			scales = [scales]

		self.scales = np.array(scales)

		if not ((len(self.scales) == self.N) or len(self.scales) == 1):
			raise ValueError("scales argument should be either equal to length of centers or take only one element")

		if len(self.scales) == 1:
			self.scales = [self.scales[0]] * self.N # not really optimal, TODO change

		if not allow_neg:
			for s in self.scales:
				if s < 0:
					raise ValueError("scales parameter cannot be negative or contain negative values if a list/array")
		else:
			for i in range(self.N):
				if self.scales[i] < 0:
					self.scales[i] = abs(self.scales[i])
					self._flips[i] = True

class GaussianMixture(Mixture, ScaleArgMixin):
	def __init__(self, centers, scales, weights=None, scale_is_variance=False, random_state=None):
		super().__init__(centers, weights, random_state=random_state)
		self.set_scale(scales)
		self.scale_is_variance = scale_is_variance

		self.multivar = False
		if not isinstance(self.centers[0], Number):
			self.multivar = True

		for c, s in zip(self.centers, self.scales):
			if scale_is_variance:
				s = np.sqrt(s)
			if self.multivar:
				dist = multivariate_normal(mean=c, cov=s)
			else:
				dist = norm(loc=c, scale=s)
			self.distributions.append(dist)

		if weights is not None:
			self.weights = weights
		else:
			self.weights = [1]*self.N
		self.weights = np.array(self.weights)

class UniformMixture(Mixture):
	def __init__(self, centers, widths, weights=None, random_state=None):
		super().__init__(centers, weights)
		if isinstance(widths, Number):
			widths = [widths]

		self.widths = np.array(widths)
		self.distributions = []

		if not ((len(self.widths) == self.N) or len(self.widths) == 1):
			raise ValueError("widths argument should be either equal to length of centers or take only one element")

		if len(self.widths) != self.N:
			self.widths = [self.widths[0]] * self.N

		for c, w in zip(self.centers, self.widths):
			dist = uniform(loc=c-w/2, scale=w)
			self.distributions.append(dist)

class ParetoMixture(Mixture, ScaleArgMixin):
	def __init__(self, b, centers, scales, weights=None, random_state=None):
		super().__init__(centers=centers, weights=weights, random_state=random_state)
		self.set_scale(scales, allow_neg=True)

		if isinstance(b, Number):
			self.bs = [b]
		else:
			self.bs = b

		if len(self.bs) == 1:
			self.bs = [self.bs[0]] * self.N

		for b, c, s in zip(self.bs, self.centers, self.scales):
			# if we want to start from zero, then we subtract one, because standard pareto starts from one
			# then 0 is as minus one, and zero is as one (so the beginning)
			self.distributions.append(pareto(b=b, loc=c-s, scale=s))

class UniformBox:
	def __init__(self, corner, shapes, random_state=None):
		self.corner = np.array(corner)
		self.shapes = np.array(shapes)

		self.dist_x = uniform(corner[0] - shapes[0]/2, scale=shapes[0])
		self.dist_y = uniform(corner[1] - shapes[1]/2, scale=shapes[1])

		self.random_state = random_state

	def pdf(self, grid):
		"""
		grid: np.array with shape (N, K,  2)

		returns an array with shape (N, K)
		"""
		flat_grid = grid.reshape(-1, 2)
		pdf_x = self.dist_x.pdf(flat_grid[:, 0])
		pdf_y = self.dist_y.pdf(flat_grid[:, 1])
		pdf = pdf_x * pdf_y # since boxes are independent uniforms
		return pdf.reshape(grid.shape[:-1])

	def rvs(self, no_samples=None, random_state=None):
		"""
		no_samples : int
		"""
		if no_samples is None:
			no_samples = 1
		rs = None
		if random_state is None and self.random_state is not None:
			rs = self.random_state
		elif self.random_state is None and random_state is not None:
			rs = random_state
		elif self.random_state is not None and random_state is not None:
			rs = random_state
		samples_x = self.dist_x.rvs(no_samples, random_state=rs)
		samples_y = self.dist_y.rvs(no_samples, random_state=rs)
		return np.stack([samples_x, samples_y], axis=1)

class UniformBoxMixture(Mixture):
	def __init__(self, centers, shapes, weights=None, random_state=None):
		super().__init__(centers, weights, random_state=random_state)
		self.shapes = np.array(shapes)
		if self.centers.ndim != self.shapes.ndim and self.centers.shape[0] != self.shapes.shape[0]:
			raise ValueError("centers and shapes should have the same dimensions")
		if self.shapes.shape[0] != self.N:
			self.shapes = [self.shapes[0]] * self.N
		for (cx, cy), (wx, wy) in zip(self.centers, self.shapes):
			self.distributions.append(UniformBox(corner=(cx-wx/2, cy-wy/2),
			                                     shapes=(wx, wy),
			                                     random_state=random_state))
		self.multivar = True

	def __repr__(self) -> str:
		return f"Uniform boxes distribution.\nCenters: {self.centers}, shapes: {self.shapes}, weights: {self.weights}"


def sample_spiral(n_samples, low=np.pi, high=4*np.pi, radius_multiplier: float=1):
	thet = np.random.uniform(low=low, high=high, size=n_samples)
	x = thet * np.sin(thet)
	y = thet * np.cos(thet)
	if np.abs(radius_multiplier - 1) > 1e-3: # no point in doing anything if multiplier is almost one
		x *= radius_multiplier
		y *= radius_multiplier
	return x, y

def build_spiral_distribution(n_samples, sigma):
	samples = sample_spiral(n_samples, radius_multiplier=0.25)
	gm = GaussianMixture(centers=samples, scales=sigma)
	return gm

# TODO I learned about existence of YAML afterwards, consider migrating config file later to it
def build_distribution_from_text(text):
	if isinstance(text, str):
		lines = text.split("\n")
	elif isinstance(text, list):
		lines = text
		for l in lines:
			if not isinstance(l, str):
				raise TypeError("text should be a string or a list of strings")
	else:
		raise TypeError("text should be a string or a list of strings")

	gaussians_c = []
	gaussians_s = []
	gaussians_weights = []
	uniforms_c = []
	uniforms_w = []
	uniforms_weights = []
	paretos_b = []
	paretos_c = []
	paretos_s = []
	paretos_weights = []
	for line in lines:
		line = line.strip()
		if len(line) == 0:
			continue
		line_split = line.split()
		dist_type = line_split[0]
		if dist_type == 'gauss':
			mean, scale = float(line_split[1]), float(line_split[2])
			gaussians_c.append(mean)
			gaussians_s.append(scale)
			if len(line_split) > 3:
				gaussians_weights.append(float(line_split[3]))
			else:
				gaussians_weights.append(1)
		elif dist_type == 'uniform':
			center, width = float(line_split[1]), float(line_split[2])
			uniforms_c.append(center)
			uniforms_w.append(width)
			if len(line_split) > 3:
				uniforms_weights.append(float(line_split[3]))
			else:
				uniforms_weights.append(1)
		elif dist_type == 'pareto':
			b, loc, scale = float(line_split[1]), float(line_split[2]), float(line_split[3])
			paretos_b.append(b)
			paretos_c.append(loc)
			paretos_s.append(scale)
			if len(line_split) > 4:
				paretos_weights.append(float(line_split[4]))
			else:
				paretos_weights.append(1)

	mixture = Mixture()
	if len(gaussians_c):
		gauss = GaussianMixture(centers=gaussians_c, scales=gaussians_s, weights=gaussians_weights)
		mixture.extend(gauss)
	if len(uniforms_c):
		uniform = UniformMixture(centers=uniforms_c, widths=uniforms_w, weights=uniforms_weights)
		mixture.extend(uniform)
	if len(paretos_b):
		paretor = ParetoMixture(centers=paretos_c, scales=paretos_s, b=paretos_b, weights=paretos_weights)
		mixture.extend(paretor)
	return mixture

def interpolate_mixtures(mixture_one, mixture_two, w):

	if not isinstance(w, float):
		raise TypeError("w should be a float")
	if not 1 >= w >= 0:
		raise ValueError("w should be a positive integer between 0 and 1")
	m = Mixture()
	m.extend(mixture_one)
	nmone = len(mixture_one)
	for i in range(nmone):
		m.weights[i] = w
	m.extend(mixture_two)
	nmtwo = len(mixture_two)
	for i in range(nmtwo):
		m.weights[i] = 1-w
	return m

def generate_and_dump_distributions(textone, texttwo, n_dist, samples_dir, n_samples = 400):

	mixture_one = build_distribution_from_text(textone)
	mixture_two = build_distribution_from_text(texttwo)
	os.makedirs(samples_dir, exist_ok=True)
	filenames = []
	dist_sizes = {}
	weights = np.linspace(0, 1, num=n_dist, endpoint=True)
	for i in range(n_dist):
		m = interpolate_mixtures(mixture_one, mixture_two, weights[i])
		samples = m.rvs(n_samples)
		# TODO experiment with sampling 100 times and picking the one that has the best distribution
		# for instance say that we impose symmetry and we check how good the symmetry is
		# or maybe we can impose the symmetry by hand ?
		#samples = sample_full_dist(n_samples, dense_grid=grid, very_dense_density=pdf, second_level_refinement=True)
		fname = os.path.join(samples_dir, "mixture_" + str(i) + ".hdf5")
		with h5py.File(fname, "w") as f:
			f.create_dataset("init_distribution", data=samples)
		filenames.append(fname)
		dist_sizes[fname] = len(samples)

	with open(os.path.join(samples_dir, "distribution_logfile.txt"), "w") as f:
		info_to_dump = ""
		for fname, s in dist_sizes.items():
			info_to_dump += f"{fname}|||{s}\n"
		f.write(info_to_dump)
	return filenames, dist_sizes

def parse_dist_config_file(dist_config_file):
	with open(dist_config_file, "r") as f:
		config = f.readlines()
	if config[0] != 'DIST_ONE\n':
		raise ValueError(f"incorrect config file format, instead of DIST_ONE we got {config[0]}")
	curr_dist_nr = 1
	distributions_texts = []
	curr_dist = []
	for line in config[1:]:
		if not line.startswith("DIST"):
			curr_dist.append(line.strip("\n"))
		else: # new distribution
			curr_dist_nr += 1
			distributions_texts.append(curr_dist)
			curr_dist = []

	distributions_texts.append(curr_dist)
	if curr_dist_nr != len(distributions_texts): # essentially a sanity check
		raise ValueError(f"incorrect config file format, we have {len(distributions_texts)} distributions, but"
		                 f" the number of distributions counted is {curr_dist_nr}")
	return distributions_texts