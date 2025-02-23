import argparse
import logging
import os
import time

import gpflow
import numpy as np
import tensorflow as tf

from gpflow_logger import setup_logging
from utils import (
    init_model,
    load_data,
    predict_with_full_cov,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
parser.add_argument(
    "--n_cores",
    type=int,
)
parser.add_argument(
    "--n_train",
    type=int,
)
parser.add_argument(
    "--n_test",
    type=int,
)
parser.add_argument(
    "--n_tiles",
    type=int,
)
parser.add_argument(
    "--n_reg",
    type=int,
)
parser.add_argument(
    "--n_loops",
    type=int,
)
args = parser.parse_args()

if not args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
log_filename = "./gpflow_logs.log"

PRECISION = "float64"

TRAIN_IN_FILE = "../../../data/generators/msd_simulator/data/input_data.txt"
TRAIN_OUT_FILE = "../../../data/generators/msd_simulator/data/output_data.txt"
TEST_IN_FILE = "../../../data/generators/msd_simulator/data/input_data.txt"
TEST_OUT_FILE = "../../../data/generators/msd_simulator/data/output_data.txt"



def execute(n_cores, n_train, n_test, n_tiles, n_reg, n_loops):
    setup_logging(log_filename, True, logger)

    # Check if TensorFlow is using GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        logger.info(f"GPUs available: {physical_devices}")
    else:
        logger.info("No GPUs found. Using CPU.")

    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:
        if not file_exists or os.stat(file_path).st_size == 0:
            logger.info("Write output file header")
            header = "n_cores,n_train,n_test,n_reg,i_loop,time\n"
            output_file.write(header)

        if PRECISION == "float32":
            gpflow.config.set_default_float(np.float32)
        else:
            gpflow.config.set_default_float(np.float64)

        tf.config.threading.set_intra_op_parallelism_threads(n_cores)
        for i_loop in range(n_loops):
            single_run(output_file, n_cores, n_train, n_test, n_tiles, n_reg, i_loop)

    logger.info(f"completed run: {n_cores}, {n_train}, {n_test}, {n_tiles}, {n_reg}, {n_loops}")


def single_run(csv, n_cores, n_train, n_test, n_tiles, n_reg, i_loop):
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=TRAIN_IN_FILE,
        train_out_path=TRAIN_OUT_FILE,
        test_in_path=TEST_IN_FILE,
        test_out_path=TEST_OUT_FILE,
        size_train=n_train,
        size_test=n_test,
        n_regressors=n_reg,
    )

    model = init_model(
        X_train,
        Y_train,
        k_var=1.0,
        k_lscale=1.0,
        noise_var=0.1,
        params_summary=False,
    )

    pred_full_cov_t = time.time()
    f_pred, f_full_cov = predict_with_full_cov(model, X_test)
    pred_full_cov_t = time.time() - pred_full_cov_t

    row_data = [n_cores, n_train, n_test, n_tiles, n_reg, i_loop, pred_full_cov_t]
    csv.writerow(row_data)

if __name__ == "__main__":
    execute(args.n_cores, args.n_train, args.n_test, args.n_tiles, args.n_reg, args.n_loops)
