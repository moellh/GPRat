#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

N_CORES=(24) # (1 2 3 4 6 8 12 16 24 32 48)
N_TRAIN=(4096 8192 16384 32768) # 32768, 65536
N_TILES=(8 16 32 64)
N_REG=(8)
N_LOOPS=(10)

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
