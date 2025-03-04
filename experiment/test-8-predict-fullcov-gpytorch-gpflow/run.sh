#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 8b

# GPyTorch
N_CORES=(48)
N_TRAIN=(128 256 512 1024 2048 4096 8192 16384 32768)
N_TEST=(128 256 512 1024 2048 4096 8192 16384 32768)
N_REG=(8)
N_LOOPS=(10)

cd gpytorch

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for test in "${N_TEST[@]}"; do
      for reg in "${N_REG[@]}"; do
        for loop in "${N_LOOPS[@]}"; do
          python3 run_gpytorch_cpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
          python3 run_gpytorch_gpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
        done
      done
    done
  done
done

cd ..

# GPflow
N_CORES=(48)
N_TRAIN=(128 256 512 1024 2048 4096 8192 16384 32768)
N_TEST=(128 256 512 1024 2048 4096 8192 16384 32768)
N_REG=(8)
N_LOOPS=(10)

cd gpflow

for core in "${N_CORES[@]}"; do
  for train in "${N_TRAIN[@]}"; do
    for test in "${N_TEST[@]}"; do
      for reg in "${N_REG[@]}"; do
        for loop in "${N_LOOPS[@]}"; do
          python3 run_gpflow_cpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
          python3 run_gpflow_gpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
        done
      done
    done
  done
done

cd ..
