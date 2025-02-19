#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

run_experiment() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for tile in "${N_TILES[@]}"; do
        for reg in "${N_REG[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            python3 execute.py --n_cores $core --n_train $train --n_tiles $tile --n_reg $reg --n_loops $loop
          done
        done
      done
    done
  done
}

# N_CORES=(6 12 24 48)
# N_TRAIN=(1024)
# N_TILES=(1 2 4 8 16 32 64)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment
#
# N_CORES=(6 12 24 48)
# N_TRAIN=(2048)
# N_TILES=(1 2 4 8 16 32 64 128)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment
#
# N_CORES=(6 12 24 48)
# N_TRAIN=(4096)
# N_TILES=(2 4 8 16 32 64 128)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment
#
# N_CORES=(6 12 24 48)
# N_TRAIN=(8092)
# N_TILES=(4 8 16 32 64 128 256)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment
#
# N_CORES=(6 12 24 48)
# N_TRAIN=(16384)
# N_TILES=(8 16 32 64 128 256)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment
#
# N_CORES=(6 12 24 48)
# N_TRAIN=(32768)
# N_TILES=(512) # (16 32 64 128 256 512)
# N_REG=(8)
# N_LOOPS=(10)
#
# run_experiment

# cpu part of test 5 uses results from test 1 and this (n_train=65536, optimal n_cores)
N_CORES=(48)
N_TRAIN=(65536)
N_TILES=(16 32 64 128 256 512)
N_REG=(8)
N_LOOPS=(10)

run_experiment
