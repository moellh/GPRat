#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

N_REG=(8)
N_LOOPS=(11)

# Test 13 for GPyTorch
N_CORES=(6 12 24 48)
N_TRAIN=(1024 2048 4096) # TODO: higher
N_TEST=(1024 2048 4096) # TODO: higher

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

# Test 13 for GPflow
N_CORES=(6 12 24 48)
N_TRAIN=(1024 2048 4096) # TODO: higher
N_TEST=(1024 2048 4096) # TODO: higher

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
