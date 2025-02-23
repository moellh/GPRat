#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 11
# Predict
# CPU only
# increasing problem size
# increasing n_tiles
# increasing n_cores
N_CORES=(6 12 24 48)
N_TRAIN=(1024 2048 4096) # (1024 2048 4096 8192 16384 32768)
N_TEST=(1024 2048 4096) # ...
N_TILES=(1 2 4 8 16 32 64 128)

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for tile in "${N_TILES[@]}"; do
      for reg in "${N_REG[@]}"; do
        for loop in "${N_LOOPS[@]}"; do
          python3 execute-cpu.py --n_cores $core --n_train $train --n_tiles $tile --n_reg $reg --n_loops $loop
        done
      done
    done
  done
done

# Test 12
# Predict
# GPU only
# increasing problem size
# increasing n_tiles
# increasing n_streams
N_CORES=(48)
N_STREAMS=(1 2 4 8 16 32 64 128)
N_TRAIN=(1024 2048 4096) # (1024 2048 4096 8192 16384 32768)
N_TEST=(1024 2048 4096) # ...
N_TILES=(1 2 4 8 16 32 64 128)

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for tile in "${N_TILES[@]}"; do
      for reg in "${N_REG[@]}"; do
        for streams in "${N_STREAMS[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            python3 execute-gpu.py --n_cores $core --n_train $train --n_tiles $tile --n_reg $reg --n_loops $loop --n_streams $streams
          done
        done
      done
    done
  done
done
