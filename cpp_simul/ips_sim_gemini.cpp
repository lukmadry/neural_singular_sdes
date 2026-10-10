#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <Accelerate/Accelerate.h>
// #include <highfive/H5File.hpp>
// only works on mac because of accelerate

// Helper to handle mean computation over (N - 1) off-diagonal terms
inline double inverse_power_law(double x, double eps) {
	double abs_x = std::abs(x);
	double val = 1.0 / (eps + std::pow(abs_x, 0.4));
	return (x > 0.0) ? val : ((x < 0.0) ? -val : 0.0);
}

void run_simulation() {
	// -------------------------------------------------------------
	// Simulation Parameters
	// -------------------------------------------------------------
	const int N = 150;
	const double alph = 0.4;
	const double eps = std::pow(N, -1.0 + 1e-4);
	// const double p_law = 0.4;

	const double dt = 0.1 * eps * eps;
	const double T = 1.0;
	const double sqdt = std::sqrt(2.0 * dt);

	const int M = static_cast<int>(T / dt) + 1;
	const int pbar_length = 100;
	const int verbosity_step = std::max(1, M / pbar_length);

	std::cout << "Number of particles: " << N 
	          << ", number of steps M: " << M 
	          << ", time step: " << dt << "\n";

	// Flat vector representing an (M x N) row-major matrix
	std::vector<double> intermediate(static_cast<size_t>(M) * N);
	std::vector<double> x(N);
	std::vector<double> x_next(N);

	std::vector<double> diffs(N);
	std::vector<double> abs_diffs(N);
	std::vector<double> pow_results(N);
	std::vector<double> p_vec(N, alph); // Constant vector for exponent p

	// gemini suggested these would change something but they did not
	double* __restrict__ diffs_ptr = diffs.data();
	double* __restrict__ abs_ptr = abs_diffs.data();
	double* __restrict__ pow_ptr = pow_results.data();
	double* __restrict__ x_ptr = x.data();
	double* __restrict__ x_next_ptr = x_next.data();
	const double* __restrict__ p_ptr = p_vec.data();

	// Random Engines
	std::mt19937_64 rng(42); // Fixed seed for reproducibility; swap for std::random_device{}()
	std::uniform_real_distribution<double> unif_dist(-1.0, 1.0);
	std::normal_distribution<double> norm_dist(0.0, sqdt);

	// Initial condition: Uniform[-1, 1]
	for (int i = 0; i < N; ++i) {
		x[i] = unif_dist(rng);
	}

	const double inv_N = 1.0 / static_cast<double>(N);

	auto t0 = std::chrono::high_resolution_clock::now();

	// -------------------------------------------------------------
	// Main Euler-Maruyama Integration Loop
	// -------------------------------------------------------------
	for (int i = 0; i < M; ++i) {
		if (i % verbosity_step == 0) {
			double progress = (100.0 * i) / M;
			std::cout << "\rProgress: [" << static_cast<int>(progress) << "%] Step " << i << "/" << M << std::flush;
		}

		// Store current positions at step i (Row-Major: i * N + j)
		std::copy(x.begin(), x.end(), intermediate.begin() + static_cast<size_t>(i) * N);

		// Compute interactions without extra allocations
		for (int j = 0; j < N; ++j) {
			double drift_sum = 0.0;
			double x_j = x_ptr[j];

			for (int k = 0; k < N; ++k) {
				diffs_ptr[k] = x_j - x_ptr[k];
			}
			vvfabs(abs_ptr, diffs_ptr, &N);
			vvpow(pow_ptr, p_ptr, abs_ptr, &N);
			double sum = 0.0;
			for (int k = 0; k < N; ++k) {
				double sign = static_cast<double>(diffs_ptr[k] > 0.0) - static_cast<double>(diffs_ptr[k] < 0.0);
				double val = 1.0 / (eps + pow_ptr[k]);
				sum += sign * val;
				
			};
			;
			double noise = norm_dist(rng);
			x_next_ptr[j] = x_j + dt * sum / N + noise;
		}

		x = x_next;
	}

	std::cout << "\rProgress: [100%] Step " << M << "/" << M << "\n";

	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = t1 - t0;
	std::cout << "Simulation elapsed time: " << elapsed.count() << " s\n";

	// // -------------------------------------------------------------
	// // Dump Output Matrix directly to HDF5
	// // -------------------------------------------------------------
	// std::string fname = "ips_" + std::to_string(N) + "_" + std::to_string(alph) + ".hdf5";
	
	// HighFive::File file(fname, HighFive::File::Overwrite);
	// std::vector<size_t> dims = {static_cast<size_t>(M), static_cast<size_t>(N)};

	// HighFive::DataSetCreateProps props;
	// props.add(HighFive::Chunking({std::min(static_cast<size_t>(M), size_t(1000)), static_cast<size_t>(N)}));
	// props.add(HighFive::Deflate(6)); // Level 6 gzip compression

	// HighFive::DataSet dataset = file.createDataSet<double>("/data", HighFive::DataSpace(dims), props);
	// dataset.write_raw(intermediate.data());

	// std::cout << "Data written successfully to " << fname << "\n";
}

int main() {
	run_simulation();
	return 0;
}