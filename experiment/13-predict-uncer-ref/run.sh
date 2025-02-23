#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 15
# Predict with Full Cov
# CPU, GPU with GPyTorch
# increasing problem size
# increasing n_tiles
N_CORES=(6 12 24 48)
N_REG=(8)
N_LOOPS=(10)
N_TRAIN=(1024 2048 4096) # (1024 2048 4096 8192 16384 32768, 65536 fuer 4+ tiles)
N_TEST=(1024 2048 4096) # ...
N_TILES=(1 2 4 8 16 32 64 128)

cd gpytorch

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for test in "${N_TEST[@]}"; do
      for tile in "${N_TILES[@]}"; do
        for reg in "${N_REG[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            python3 run_gpytorch_cpu.sh --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
            python3 run_gpytorch_gpu.sh --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
          done
        done
      done
    done
  done
done

cd ..

# Test 15
# Predict with Full Cov
# CPU, GPU with GPflow
# increasing problem size
# increasing n_tiles
N_CORES=(6 12 24 48)
N_REG=(8)
N_LOOPS=(10)
N_TRAIN=(1024 2048 4096) # (1024 2048 4096 8192 16384 32768, 65536 fuer 4+ tiles)
N_TEST=(1024 2048 4096) # ...
N_TILES=(1 2 4 8 16 32 64 128)

cd gpflow

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for test in "${N_TEST[@]}"; do
      for tile in "${N_TILES[@]}"; do
        for reg in "${N_REG[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            python3 run_gpflow_cpu.sh --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
            python3 run_gpflow_gpu.sh --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
          done
        done
      done
    done
  done
done

cd ..
