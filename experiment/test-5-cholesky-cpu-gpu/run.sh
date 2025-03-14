#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

N_CORES=(48)
N_REG=(8)

run_experiment_cpu() {
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
}

run_experiment_gpu() {
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
}

N_LOOPS=(10)

N_TRAIN=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TILES=(64)
run_experiment_cpu

N_TRAIN=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
N_TILES=(32)
N_STREAMS=(32)
run_experiment_gpu

N_LOOPS=(50)

N_TRAIN=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
N_TILES=(4)
run_experiment_cpu

N_TRAIN=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)
N_TILES=(1)
N_STREAMS=(1)
run_experiment_gpu
