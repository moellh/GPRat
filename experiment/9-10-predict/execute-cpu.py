import argparse
import logging
import os
import time
from csv import writer

import gpxpy as gpx

from hpx_logger import setup_logging

logger = logging.getLogger()
log_filename = "./hpx_logs.log"

parser = argparse.ArgumentParser()
parser.add_argument("--n_cores", type=int)
parser.add_argument("--n_train", type=int)
parser.add_argument("--n_test", type=int)
parser.add_argument("--n_tiles", type=int)
parser.add_argument("--n_reg", type=int)
parser.add_argument("--n_loops", type=int)
args = parser.parse_args()

TRAIN_IN_FILE = "../../data/generators/msd_simulator/data/input_data.txt"
TRAIN_OUT_FILE = "../../data/generators/msd_simulator/data/output_data.txt"
TEST_IN_FILE = "../../data/generators/msd_simulator/data/input_data.txt"


def execute(n_cores, n_train, n_test, n_tiles, n_reg, n_loops):
    # setup logging
    setup_logging(log_filename, True, logger)

    # append log to ./output.csv
    file_exists = os.path.isfile("./output-cpu.csv")
    output_file = open("./output-cpu.csv", "a", newline="")
    output_writer = writer(output_file)

    # write headers
    if not file_exists:
        logger.info("Write output file header")
        header = ["n_cores", "n_train", "n_test", "n_tiles", "n_reg", "i_loop", "time"]
        output_writer.writerow(header)

    gpx.start_hpx([], n_cores)
    for i_loop in range(n_loops):
        single_run(output_writer, n_cores, n_train, n_test, n_tiles, n_reg, i_loop)
    gpx.stop_hpx()
    logger.info(f"completed run: {n_cores}, {n_train}, {n_test}, {n_tiles}, {n_reg}, {n_loops}")
    if os.path.isfile("apex_profiles.csv"):
        new_filename = f"apex-cpu/apex_profiles_{n_cores}_{n_train}_{n_test}_{n_tiles}_{n_reg}_{n_loops}.csv"
        os.rename("apex_profiles.csv", new_filename)
        logger.info("moving csv")


def single_run(csv, n_cores, n_train, n_test, n_tiles, n_reg, i_loop):

    n_tile_size = gpx.compute_train_tile_size(n_train, n_tiles)
    m_tiles, m_tile_size = gpx.compute_test_tiles(n_test, n_tiles, n_tile_size)
    train_in = gpx.GP_data(TRAIN_IN_FILE, n_train)
    train_out = gpx.GP_data(TRAIN_OUT_FILE, n_train)
    test_in = gpx.GP_data(TEST_IN_FILE, n_test)

    gp_cpu = gpx.GP(
        train_in.data,
        train_out.data,
        n_tiles,
        n_tile_size,
        trainable=[True, True, True],
    )

    pred_t = time.time()
    _ = gp_cpu.predict(test_in, m_tiles, m_tile_size)
    pred_t = time.time() - pred_t

    row_data = [n_cores, n_train, n_test, n_tiles, n_reg, i_loop, pred_t]
    csv.writerow(row_data)


if __name__ == "__main__":
    execute(args.n_cores, args.n_train, args.n_test, args.n_tiles, args.n_reg, args.n_loops)
