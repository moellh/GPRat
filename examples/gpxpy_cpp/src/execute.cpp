#include "../install_cpp/include/gpxpy_c.hpp"
#include "../install_cpp/include/utils_c.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <hpx/algorithm.hpp>
#include <iostream>
#include <string>

auto now = std::chrono::high_resolution_clock::now;

int main(int argc, char *argv[])
{
    // number of training points, number of rows/columns in the kernel matrix
    const int N_TRAIN_START = 1 << 10;  // 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    const int N_TRAIN_END = 1 << 10;    // 7,   8,   9,   10,   11,   12,   13,   14,  15

    const int N_TEST = 8;

    const int LOOPS = 1;
    const int OPTIMIZE_ITERATIONS = 1;

    // 2^NUM_CORES_EXPONENT CPU cores are used by HPX
    const std::size_t NUM_CORES = 4;

    const int N_REGRESSORS = 8;

    // number of tiles per dimension
    const int N_TRAIN_TILES = 4;

    // number of regressors, i.e. number of previous points incl. current point
    // considered for each entry in the kernel matrix
    std::string train_in_path = "../../../data/data_1024/training_input.txt";
    std::string train_out_path = "../../../data/data_1024/training_output.txt";
    std::string test_in_path = "../../../data/data_1024/test_input.txt";

    // Add number of threads to arguments
    std::vector<std::string> args(argv, argv + argc);
    args.push_back("--hpx:threads=" + std::to_string(NUM_CORES));

    // Convert arguments to char* array
    std::vector<char *> cstr_args;
    for (auto &arg : args)
    {
        cstr_args.push_back(const_cast<char *>(arg.c_str()));
    }
    int hpx_argc = static_cast<int>(cstr_args.size());
    char **hpx_argv = cstr_args.data();

    // Start HPX runtime with arguments
    utils::start_hpx_runtime(hpx_argc, hpx_argv);

    for (std::size_t n_train_tiles = N_TRAIN_TILES; n_train_tiles <= N_TRAIN_TILES; n_train_tiles *= 2)  // NOTE: currently all cores
    {
        for (std::size_t n_train = N_TRAIN_START; n_train <= N_TRAIN_END; n_train *= 2)
        {
            for (std::size_t loop = 0; loop < LOOPS; loop++)
            {
                // Total time ---------------------------------------------- {{{
                auto start_total = now();

                // Compute tile sizes and number of predict tiles
                int n_train_tile_size = utils::compute_train_tile_size(n_train, n_train_tiles);
                auto [n_test_tiles, n_test_tile_size] = utils::compute_test_tiles(N_TEST, n_train_tiles, n_train_tile_size);

                // Hyperparameters for Adam optimizer
                std::vector<double> M = { 0.0, 0.0, 0.0 };
                gpxpy_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPTIMIZE_ITERATIONS, M };

                // Load data from files
                gpxpy::GP_data training_input(train_in_path, n_train);
                gpxpy::GP_data training_output(train_out_path, n_train);
                gpxpy::GP_data test_input(test_in_path, N_TEST);

                // GP construct time --------------------------------------- {{{
                auto start_init_gp = now();

                std::vector<bool> trainable = { true, true, true };

                // GP for CPU computation
                std::string target = "cpu";
                gpxpy::GP gp_cpu(training_input.data, training_output.data, n_train_tiles, n_train_tile_size, 1.0, 1.0, 0.1, N_REGRESSORS, trainable);

                // GP for GPU computation
                target = "gpu";
                int device = 0;
                int n_streams = 32;
                gpxpy::GP gp_gpu(training_input.data, training_output.data, n_train_tiles, n_train_tile_size, 1.0, 1.0, 0.1, N_REGRESSORS, trainable, device, n_streams);

                auto init_time = now() - start_init_gp;  // ----------------- }}}

                // Cholesky factorization time ----------------------------- {{{
                auto start_cholesky = now();

                std::vector<std::vector<double>> cpu_tiles = gp_cpu.cholesky();
                std::vector<std::vector<double>> gpu_tiles = gp_gpu.cholesky();

                auto cholesky_time = now() - start_cholesky;
                // ------------ }}}

                // Optimize time (for OPTIMIZE_ITERATIONS) ----------------- {{{
                auto start_opt = now();
                /*
                std::vector<double> losses = gp.optimize(hpar);
                */
                auto opt_time = now() - start_opt;  // ---------------------- }}}

                // Optimize step time ------------------ {{{
                auto start_opt_step = now();
                /*
                std::vector<double> losses = gp.optimize(hpar);
                */
                auto opt_step_time = now() - start_opt_step;  // ---------------------- }}}

                // Calculate loss time ------------------- {{{
                auto start_calc_loss = now();
                /* double loss_cpu = gp_cpu.calculate_loss();
                double loss_gpu = gp_gpu.calculate_loss();
                double calc_loss_err = std::abs(loss_cpu - loss_gpu);
                std::cout << "Calc Loss error: " << calc_loss_err << std::endl; */
                auto calc_loss_time = now() - start_calc_loss;  // -- }}}

                // Predict time -------------------------------------------- {{{
                auto start_pred = now();
                /* std::vector<double> cpu_pred = gp_cpu.predict(test_input.data, n_test_tiles, n_test_tile_size);
                std::vector<double> gpu_pred = gp_gpu.predict(test_input.data, n_test_tiles, n_test_tile_size); */
                auto pred_time = now() - start_pred;  // ----------------- }}}

                // Predict & Uncertainty time  ----------------------------- {{{
                auto start_pred_uncer = now();
                /* std::vector<std::vector<double>> pred_uncer_cpu = gp_cpu.predict_with_uncertainty(test_input.data, n_test_tiles, n_test_tile_size);
                std::vector<std::vector<double>> pred_uncer_gpu = gp_gpu.predict_with_uncertainty(test_input.data,n_test_tiles , n_test_tile_size);
                double pred_uncer_err = 0;
                for (std::size_t j = 0; j < pred_uncer_cpu[0].size(); j++)
                {
                    pred_uncer_err += std::abs(pred_uncer_cpu[0][j] - pred_uncer_gpu[0][j]);
                }
                for (std::size_t j = 0; j < pred_uncer_cpu[1].size(); j++)
                {
                    pred_uncer_err += std::abs(pred_uncer_cpu[1][j] - pred_uncer_gpu[1][j]);
                }
                std::cout << "Pred Uncer error: " << pred_uncer_err << std::endl; */
                auto pred_uncer_time = now() - start_pred_uncer;  // -------- }}}

                // Predictions with full covariance time ------------------- {{{
                auto start_pred_full_cov = now();
                std::vector<std::vector<double>> pred_full_cpu = gp_cpu.predict_with_full_cov(test_input.data, n_test_tile_size, n_test_tile_size);
                std::vector<std::vector<double>> pred_full_gpu = gp_gpu.predict_with_full_cov(test_input.data, n_test_tile_size, n_test_tile_size);
                auto pred_full_cov_time = now() - start_pred_full_cov;  // -- }}}

                auto total_time = now() - start_total;  // ----------------- }}}

                // Append parameters & times as CSV
                std::ofstream outfile("../output.csv", std::ios::app);

                // If file is empty, write the header
                if (outfile.tellp() == 0)
                {
                    outfile << "target,n_train,n_tiles,";
                    outfile << "cholesky_time,";
                    // outfile << "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,"
                    outfile << "n_loop\n";
                }
                outfile << target << ","
                        // << cores << ","
                        << n_train << ","
                        // << N_TEST << ","
                        << n_train_tiles << ","
                        // << N_REGRESSORS << ","
                        // << OPTIMIZE_ITERATIONS << ","
                        // << total_time.count() << ","
                        // << init_time.count() << ","
                        << cholesky_time.count() << ","
                        // << opt_time.count() << "," << pred_uncer_time.count() << "," << pred_full_cov_time.count() << "," << pred_time.count() << ","
                        << loop << "\n";
                outfile.close();
            }
        }
    }

    // Stop HPX runtime
    utils::stop_hpx_runtime();

    return 0;
}
