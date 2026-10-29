python3 tests.py

printf "testing distribution dependent run"
python3 train_nn_on_drifts.py --mode generate --N 5 --cpp_file_fpath cpp_simul/ips_sim_lookup_table_load_init_dist.cpp --compiled_file cpp_simul/sde_lt_init_dist_test --Ninter 100 --add_distribution_generation --add_compilation --samples_dir init_dist_moving_gaussian --no-with_polynomial --script_filepath tests/generation_test_script.sh
chmod +x tests/generation_test_script.sh
./tests/generation_test_script.sh
python3 train_nn_on_drifts.py --mode train_in_out --different_dists --storage_dir tests/storage_narrow_init_dist --dist_dir tests/init_dist_moving_gaussian --experiment_data_dir tests/exp_data --training_range_config_file tests/configs/in_out.txt --script_filepath tests/generation_test_script.sh --Ninter 150 --max_batches 10 --no_epochs 1 --smoke_run