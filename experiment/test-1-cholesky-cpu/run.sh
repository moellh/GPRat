#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

N_CORES=(6 12 24 48)
N_REG=(8)
N_LOOPS=(11)

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

N_TRAIN=(1024)
N_TILES=(1 2 4 8 16 32 64)
run_experiment

N_TRAIN=(2048)
N_TILES=(2 4 8 16 32 64 128)
run_experiment

N_TRAIN=(4096)
N_TILES=(2 4 8 16 32 64 128)
run_experiment

N_TRAIN=(8192)
N_TILES=(4 8 16 32 64 128 256)
run_experiment

N_TRAIN=(16384)
N_TILES=(8 16 32 64 128 256)
run_experiment

N_TRAIN=(32768)
N_TILES=(16 32 64 128 256)
run_experiment

N_CORES=(48)
N_TRAIN=(65536)
N_TILES=(16 32 64 128 256)
run_experiment
