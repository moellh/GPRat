#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 7a

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
N_TILES=(1)
N_REG=(8)
N_LOOPS=(50)

N_TRAIN=(1)
N_TEST=(1)
run_experiment_cpu

N_TRAIN=(2)
N_TEST=(2)
run_experiment_cpu

N_TRAIN=(4)
N_TEST=(4)
run_experiment_cpu

N_TRAIN=(8)
N_TEST=(8)
run_experiment_cpu

N_TRAIN=(16)
N_TEST=(16)
run_experiment_cpu

N_TRAIN=(32)
N_TEST=(32)
run_experiment_cpu

N_TRAIN=(64)
N_TEST=(64)
run_experiment_cpu

N_TRAIN=(128)
N_TEST=(128)
run_experiment_cpu

N_TRAIN=(256)
N_TEST=(256)
run_experiment_cpu

N_TRAIN=(512)
N_TEST=(512)
run_experiment_cpu

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_cpu

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_cpu

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_cpu

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_cpu

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_cpu

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_cpu

N_CORES=(48)
N_TILES=(16)
N_REG=(8)
N_LOOPS=(10)

N_TRAIN=(16)
N_TEST=(16)
run_experiment_cpu

N_TRAIN=(32)
N_TEST=(32)
run_experiment_cpu

N_TRAIN=(64)
N_TEST=(64)
run_experiment_cpu

N_TRAIN=(128)
N_TEST=(128)
run_experiment_cpu

N_TRAIN=(256)
N_TEST=(256)
run_experiment_cpu

N_TRAIN=(512)
N_TEST=(512)
run_experiment_cpu

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_cpu

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_cpu

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_cpu

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_cpu

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_cpu

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_cpu

# ---

N_CORES=(48)
N_TILES=(4)
N_REG=(8)
N_STREAMS=(4)
N_LOOPS=(10)

N_TRAIN=(4)
N_TEST=(4)
run_experiment_gpu

N_TRAIN=(8)
N_TEST=(8)
run_experiment_gpu

N_TRAIN=(16)
N_TEST=(16)
run_experiment_gpu

N_TRAIN=(32)
N_TEST=(32)
run_experiment_gpu

N_TRAIN=(64)
N_TEST=(64)
run_experiment_gpu

N_TRAIN=(128)
N_TEST=(128)
run_experiment_gpu

N_TRAIN=(256)
N_TEST=(256)
run_experiment_gpu

N_TRAIN=(512)
N_TEST=(512)
run_experiment_gpu

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_gpu

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_gpu

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_gpu

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_gpu

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_gpu

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_gpu

N_CORES=(48)
N_TILES=(1)
N_REG=(8)
N_STREAMS=(1)
N_LOOPS=(10)

N_TRAIN=(1)
N_TEST=(1)
run_experiment_gpu

N_TRAIN=(2)
N_TEST=(2)
run_experiment_gpu

N_TRAIN=(4)
N_TEST=(4)
run_experiment_gpu

N_TRAIN=(8)
N_TEST=(8)
run_experiment_gpu

N_TRAIN=(16)
N_TEST=(16)
run_experiment_gpu

N_TRAIN=(32)
N_TEST=(32)
run_experiment_gpu

N_TRAIN=(64)
N_TEST=(64)
run_experiment_gpu

N_TRAIN=(128)
N_TEST=(128)
run_experiment_gpu

N_TRAIN=(256)
N_TEST=(256)
run_experiment_gpu

N_TRAIN=(512)
N_TEST=(512)
run_experiment_gpu

N_TRAIN=(1024)
N_TEST=(1024)
run_experiment_gpu

N_TRAIN=(2048)
N_TEST=(2048)
run_experiment_gpu

N_TRAIN=(4096)
N_TEST=(4096)
run_experiment_gpu

N_TRAIN=(8192)
N_TEST=(8192)
run_experiment_gpu

N_TRAIN=(16384)
N_TEST=(16384)
run_experiment_gpu

N_TRAIN=(32768)
N_TEST=(32768)
run_experiment_gpu
