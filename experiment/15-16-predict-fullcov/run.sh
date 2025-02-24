#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

N_REG=(8)
N_LOOPS=(11)

# Test 15,16

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

N_CORES=(6 12 24 48)
N_TRAIN=(1024) # TODO: higher
N_TEST=(1024) # TODO: higher
N_TILES=(1 2 4 8 16 32 64)
run_experiment_cpu

N_CORES=(48)
N_STREAMS=(32)
N_TRAIN=(1024) # TODO: higher
N_TEST=(1024) # TODO: higher
N_TILES=(1 2 4 8 16 32 64)
run_experiment_cpu
