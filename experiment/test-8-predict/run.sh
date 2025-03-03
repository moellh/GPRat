#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 8

run_experiment_cpu() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for test in "${N_TEST[@]}"; do
        for tile in "${N_TILES[@]}"; do
          for reg in "${N_REG[@]}"; do
            for loop in "${N_LOOPS[@]}"; do
              python3 execute-cpu.py --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop
            done
          done
        done
      done
    done
  done
}

run_experiment_gpu() {
  for core in "${N_CORES[@]}"; do
    for train in "${N_TRAIN[@]}"; do
      for test in "${N_TEST[@]}"; do
        for tile in "${N_TILES[@]}"; do
          for reg in "${N_REG[@]}"; do
            for streams in "${N_STREAMS[@]}"; do
              for loop in "${N_LOOPS[@]}"; do
                python3 execute-gpu.py --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop --n_streams $streams
              done
            done
          done
        done
      done
    done
  done
}

N_CORES=(48)
N_STREAMS=(32)
N_REG=(8)
N_LOOPS=(10)

# CPU

N_TRAIN=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TEST=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TILES=(128)
run_experiment_cpu

# GPU

N_TRAIN=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TEST=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TILES=(32)
run_experiment_gpu
