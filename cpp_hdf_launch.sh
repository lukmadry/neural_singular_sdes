g++ -O3 -march=native \
	-I/usr/include/mkl \
	-I/usr/include/hdf5/serial \
	-L/usr/lib/x86_64-linux-gnu/hdf5/serial \
	cpp_simul/ips_sim_gemini_ubuntu.cpp \
	-lmkl_rt -lhdf5_cpp -lhdf5 \
	-o cpp_simul/sde_mkl