#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <H5Cpp.h>
#include <mkl.h> // over time get rid of this additional dependence and replace it with #pragma or lookup tables

// #include <Accelerate/Accelerate.h>
// #include <highfive/H5File.hpp>

// Helper to handle mean computation over (N - 1) off-diagonal terms
inline double inverse_power_law(double x, double eps) {
	double abs_x = std::abs(x);
	double val = 1.0 / (eps + std::pow(abs_x, 0.4));
	return (x > 0.0) ? val : ((x < 0.0) ? -val : 0.0);
}

// writing h5 functions, thanks gemini
void write_hdf5(const std::string& filename, const double* data, int Ninter, int Nsteps) {
	// 1. Create file (overwrite if exists)
	H5::H5File file(filename, H5F_ACC_TRUNC);

	hsize_t dims[2] = {
			static_cast<hsize_t>(Ninter), 
			static_cast<hsize_t>(Nsteps)
		};

	// 2. Define dimensions
	// hsize_t dims[1] = {size};
	H5::DataSpace dataspace(2, dims);

	// 3. Create dataset
	H5::DataSet dataset = file.createDataSet(
		"drift_results", 
		H5::PredType::NATIVE_DOUBLE, 
		dataspace
	);

	// 4. Write memory buffer to dataset
	dataset.write(data, H5::PredType::NATIVE_DOUBLE);
}

// not entirely sure why but this appears to be faster than CPU instructions
void compute_polynom_horner(double* x_ptr, std::vector<double> &coefs, double* output, int Ng, int D) {
	#pragma omp simd
	for (int i = 0; i < Ng; i++) {
		double x = x_ptr[i];
		double val = coefs[D - 1];
		for (int m = D - 2; m >= 0; m--) {
			val = coefs[m] + x * val;
		}
		output[i] = x * val; // Scale by x for non-zero constant term
		output[i] = std::max(-1.0, std::min(1.0, output[i])); // we do not want to deal with possible explosion, just add small fluctuations
	}
}

void fill_in_lookup_table(double pow, double high, double grid_step, int size, double *table_ptr){
	double cval = 0;
	for (int i = 0; i < size; i++){
		table_ptr[i] = cval;
		cval += grid_step;
	}
	std::vector<double> alph_big_vec(size, pow);
	vdPow(size, table_ptr, alph_big_vec.data(), table_ptr);
}

inline double lookup_value(double x, double grid_step, double *table_ptr, int idx_max, double alph){
	int k = static_cast<int> (x / grid_step);
	if (k > idx_max)
		return std::pow(x, alph); // I don't think there's a way out of this, I would like to have a decay
	return table_ptr[k];
}


void run_simulation(int Ninter, double alpha_arg, double multip_arg, std::vector<double> &taylor_coefs, int p_degree, std::string fname) {
	// -------------------------------------------------------------
	// Simulation Parameters
	// -------------------------------------------------------------
	const int N = Ninter;
	const double alph = alpha_arg; // this will be replaced by a arbitrary value.
	const double eps = std::pow(N, -1.0 + 1e-4);
	// const double p_law = 0.4;

	const double dt = 0.1 * eps * eps;
	const double T = 1.0;
	const double sqdt = std::sqrt(2.0 * dt);

	const int M = static_cast<int>(T / dt) + 1;
	const int pbar_length = 100;
	const int verbosity_step = std::max(1, M / pbar_length);
	const int save_every = 1000;
	const int saved_rows = static_cast<int>(M / save_every) + 1;
	const double lookup_table_grid_step = 0.001;
	const double lookup_table_max = 15.0; // TODO choose this by checking the gaussian tail estimates
	const int lookup_table_size = static_cast<int>(lookup_table_max / lookup_table_grid_step);

	const double multiplier = multip_arg;
	const int taylor_coef_len = p_degree;

	int current_pos_to_save = 0;

	std::cout << "Number of particles: " << N 
	          << ", number of steps M: " << M 
	          << ", time step: " << dt
	          << ", alpha: " << alph << "\n";

	// Flat vector representing an (M x N) row-major matrix
	std::vector<double> intermediate(static_cast<size_t>(saved_rows) * N); // reduce it
	std::vector<double> x(N);
	std::vector<double> x_next(N);

	std::vector<double> diffs(N);
	std::vector<double> abs_diffs(N);
	std::vector<double> pow_results(N);
	std::vector<double> p_vec(N, alph); 
	std::vector<double> taylor_coef_vec(taylor_coef_len * N, 0);
	std::vector<double> taylor_powers_vec(taylor_coef_len * N, 0);
	std::vector<double> taylor_polynom(N);
	std::vector<double> temporary(N);
	std::vector<double> lookup_table_pow(lookup_table_size);

	// gemini suggested these would change something but they did not change much
	double* __restrict__ diffs_ptr = diffs.data();
	double* __restrict__ abs_ptr = abs_diffs.data();
	double* __restrict__ pow_ptr = pow_results.data();
	double* __restrict__ x_ptr = x.data();
	double* __restrict__ x_next_ptr = x_next.data();
	double* __restrict__ taylor_ptr = taylor_polynom.data();
	double* lookup_table_ptr = lookup_table_pow.data();
	const double* __restrict__ p_ptr = p_vec.data();

	// Random Engines
	std::mt19937_64 rng(42); // TODO Fixed seed for reproducibility; swap for std::random_device{}()
	std::uniform_real_distribution<double> unif_dist(-1.0, 1.0);
	std::normal_distribution<double> norm_dist(0.0, sqdt);

	// Initial condition: Uniform[-1, 1]
	for (int i = 0; i < N; ++i) {
		x[i] = unif_dist(rng);
	}

	fill_in_lookup_table(alph, lookup_table_max, lookup_table_grid_step, lookup_table_size, lookup_table_ptr);

	// for (int m = 0; m < p_degree; m++)
	// 	std::cout << taylor_coefs[m] << " ";
	// std::cout << std::endl;
	const double inv_N = 1.0 / static_cast<double>(N);
	auto t0 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < M; ++i) {
		if (i % verbosity_step == 0) {
			double progress = (100.0 * i) / M;
			std::cout << "\rProgress: [" << static_cast<int>(progress) << "%] Step " << i << "/" << M << std::flush;
		}

		// Check if we are supposed to save it, and if so then save current positions
		if (i % save_every == 0){
			std::copy(x.begin(), x.end(), intermediate.begin() + static_cast<size_t>(current_pos_to_save) * N);
			current_pos_to_save++;
		}
		
		// std::cout << "we reach here1" << std::endl;
		// Compute interactions without extra allocations
		#pragma omp simd
		for (int j = 0; j < N; ++j) {
			double drift_sum = 0.0;
			double x_j = x_ptr[j];

			#pragma omp simd
			for (int k = 0; k < N; ++k) {
				diffs_ptr[k] = x_j - x_ptr[k];
			}
			// these are optimised to handle non-rational powers, otherwise no need to bother
			// vdAbs(N, diffs.data(), abs_diffs.data());
			// vdPow(N, abs_diffs.data(), p_vec.data(), pow_results.data());

			// std::cout << "we reach here2" << std::endl;
			// compute_polynom_horner(diffs.data(), taylor_coefs, taylor_ptr, N, taylor_coef_len);
			double sum = 0.0;

			// std::cout << "we reach here3" << std::endl;
			#pragma omp simd
			for (int k = 0; k < N; ++k) {
				pow_ptr[k] = lookup_value(std::abs(diffs[k]), lookup_table_grid_step, lookup_table_ptr, lookup_table_size, alph);
				double sign = static_cast<double>(diffs_ptr[k] > 0.0) - static_cast<double>(diffs_ptr[k] < 0.0);
				double val = multiplier / (eps + pow_ptr[k]) * (1 + taylor_polynom[k]);
				sum += sign * val;
			};
			;

			// std::cout << "we reach here4" << std::endl;
			double noise = norm_dist(rng);
			x_next_ptr[j] = x_j + dt * sum / N + noise;
			// std::fill(taylor_polynom.begin(), taylor_polynom.end(), 0.0);
		}

		x = x_next;
	}

	std::cout << "\rProgress: [100%] Step " << M << "/" << M << "\n";

	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = t1 - t0;
	std::cout << "Simulation elapsed time: " << elapsed.count() << " s\n";

	const double* i_ptr = intermediate.data();
	write_hdf5(fname, i_ptr, N, current_pos_to_save);
}

int main(int argc, char** argv) {
	int i = 1;
	double alph(0.2), multip(1);
	int N(150);
	std::string fname("test.hdf5");
	std::vector<double> p_coefs;
	while (i < argc){
		if (std::string(argv[i]) == "--N"){
			N = atoi(argv[i+1]);
		}
		if (std::string(argv[i]) == "--a"){
			alph = atof(argv[i+1]);
		}
		if (std::string(argv[i]) == "--m"){
			multip = atof(argv[i+1]);
		}
		if (std::string(argv[i]) == "--f"){
			fname = argv[i+1]; // will be generated from a script anyway
		}
		if (std::string(argv[i]) == "--p"){
			int k = atoi(argv[i+1]); // a bit annoying but this is not a production script
			for (int m = 0; m < k; m++){
				p_coefs.push_back(atof(argv[i+1+m]));
			}
		}
		i++;
	}

	if (p_coefs.size() == 0){
		p_coefs.push_back(0);
		p_coefs.push_back(0);
	}
	run_simulation(N, alph, multip, p_coefs, p_coefs.size(), fname);
	return 0;
}