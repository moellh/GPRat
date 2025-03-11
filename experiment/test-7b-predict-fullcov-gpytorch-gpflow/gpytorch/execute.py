import time
import logging
import torch
import gpytorch
import os
import argparse
from csv import writer

from gpytorch_logger import setup_logging
from utils import load_data, ExactGPModel, train, predict_with_full_cov

logger = logging.getLogger()
log_filename = "./gpytorch_logs.log"

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
    "--n_reg",
    type=int,
)
parser.add_argument(
    "--n_loops",
    type=int,
)
args = parser.parse_args()

PRECISION = "float64"

TRAIN_IN_FILE = "../../../data/generators/msd_simulator/data/input_data.txt"
TRAIN_OUT_FILE = "../../../data/generators/msd_simulator/data/output_data.txt"
TEST_IN_FILE = "../../../data/generators/msd_simulator/data/input_data.txt"
TEST_OUT_FILE = "../../../data/generators/msd_simulator/data/output_data.txt"

def execute(n_cores, n_train, n_test, n_reg, n_loops):
    setup_logging(log_filename, True, logger)

    file_path = "./output-gpu.csv" if args.use_gpu else "./output-cpu.csv"
    file_exists = os.path.isfile(file_path)
    output_file = open(file_path, "a", newline="")
    output_writer = writer(output_file)

    if not file_exists:
        logger.info("Write output file header")
        header = ["n_cores", "n_train", "n_test", "n_reg", "i_loop", "time"]
        output_writer.writerow(header)

    # torch.set_num_threads(config["N_CORES"])
    if PRECISION == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    torch.set_num_threads(n_cores)
    for i_loop in range(n_loops):
        single_run(output_writer, n_cores, n_train, n_test, n_reg, i_loop)

    logger.info(f"completed run: {n_cores}, {n_train}, {n_test}, {n_reg}, {n_loops}")

def single_run(csv, n_cores, n_train, n_test, n_reg, i_loop):

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=TRAIN_IN_FILE,
        train_out_path=TRAIN_OUT_FILE,
        test_in_path=TEST_IN_FILE,
        test_out_path=TEST_OUT_FILE,
        size_train=n_train,
        size_test=n_test,
        n_regressors=n_reg,
    )
    if args.use_gpu and torch.cuda.is_available():
        X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)

    # logger.info("Finished loading the data.")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.1
    model = ExactGPModel(X_train, Y_train, likelihood)
    if args.use_gpu and torch.cuda.is_available():
        model = model.to(device)
        likelihood = likelihood.to(device)
    # logger.info("Initialized model.")

    train(model, likelihood, X_train, Y_train, training_iter=1)
    # logger.info("Trained model.")

    pred_full_cov_t = time.time()
    f_pred, f_full_cov = predict_with_full_cov(model, likelihood, X_test)
    pred_full_cov_t = time.time() - pred_full_cov_t
    # logger.info("Finished making predictions.")

    row_data = [n_cores, n_train, n_test, n_reg, i_loop, pred_full_cov_t]
    csv.writerow(row_data)


if __name__ == "__main__":
    execute(args.n_cores, args.n_train, args.n_test, args.n_reg, args.n_loops)
