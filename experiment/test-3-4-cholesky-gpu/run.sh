#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

run_experiment() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for tile in "${N_TILES[@]}"; do
        for reg in "${N_REG[@]}"; do
          for streams in "${N_STREAMS[@]}"; do
            for loop in "${N_LOOPS[@]}"; do
              python3 execute.py --n_cores $core --n_train $train --n_tiles $tile --n_reg $reg --n_streams $streams --n_loops $loop
            done
          done
        done
      done
    done
  done
}

N_CORES=(48)
N_TRAIN=(32768)
N_TILES=(1 2 4 8 16 32 64 128 256 512)
N_REG=(8)
N_STREAMS=(1 2 4 8 16 32 64 128)
N_LOOPS=(11)
run_experiment
