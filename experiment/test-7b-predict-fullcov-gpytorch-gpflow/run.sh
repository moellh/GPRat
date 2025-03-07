#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 7b

# GPyTorch

cd gpytorch

run_experiment_gpytorch() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for test in "${N_TEST[@]}"; do
        for reg in "${N_REG[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            ./run_gpytorch_cpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
            ./run_gpytorch_gpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
          done
        done
      done
    done
  done
}

N_CORES=(48)
N_REG=(8)
N_LOOPS=(10)

N_TRAIN=(128)
N_TEST=(128)
run_experiment_gpytorch

N_TRAIN=(256)
N_TEST=(256)
run_experiment_gpytorch

N_TRAIN=(512)
N_TEST=(512)
run_experiment_gpytorch

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_gpytorch

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_gpytorch

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_gpytorch

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_gpytorch

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_gpytorch

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_gpytorch

cd ..

# GPflow

cd gpflow

run_experiment_gpflow() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for test in "${N_TEST[@]}"; do
        for reg in "${N_REG[@]}"; do
          for loop in "${N_LOOPS[@]}"; do
            ./run_gpflow_cpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
            ./run_gpflow_gpu.sh --n_cores $core --n_train $train --n_test $test --n_reg $reg --n_loops $loop
          done
        done
      done
    done
  done
}

N_CORES=(48)
N_REG=(8)
N_LOOPS=(10)

N_TRAIN=(128)
N_TEST=(128)
run_experiment_gpflow

N_TRAIN=(256)
N_TEST=(256)
run_experiment_gpflow

N_TRAIN=(512)
N_TEST=(512)
run_experiment_gpflow

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_gpflow

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_gpflow

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_gpflow

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_gpflow

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_gpflow

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_gpflow

cd ..
