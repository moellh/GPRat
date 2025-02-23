#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 9,10
# Predict
# GPRat (GPU, CPU)
# increasing problem size
# increasing n_tiles
# increasing n_cores

N_CORES=(48) # TODO: opt value
N_STREAMS=(4) # TODO: opt value
N_REG=(8)
N_LOOPS=(10)

N_TRAIN=(1024 2048 4096) # TODO: ...
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32 64 128) # TODO: ...

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for test in "${N_TEST[@]}"; do
      for tile in "${N_TILES[@]}"; do
        for reg in "${N_REG[@]}"; do
          for streams in "${N_STREAMS[@]}"; do
            for loop in "${N_LOOPS[@]}"; do
              python3 execute-cpu.py --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
              python3 execute-gpu.py --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop --n_streams $streams
            done
          done
        done
      done
    done
  done
done
