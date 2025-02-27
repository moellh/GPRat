#!/bin/bash

source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Test 9,10

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
            for loop in "${N_LOOPS[@]}"; do
              python3 execute-gpu.py --n_cores $core --n_train $train --n_test $test --n_tiles $tile --n_reg $reg --n_loops $loop --n_streams $streams
            done
          done
        done
      done
    done
  done
}

N_CORES=(48) # TODO: opt value
N_STREAMS=(32) # TODO: opt value
N_REG=(8)
N_LOOPS=(11)

# CPU

N_TRAIN=(1024)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32 64) # TODO: ...
run_experiment_cpu

N_TRAIN=(2048)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32 64 128) # TODO: ...
run_experiment_cpu

N_TRAIN=(4096)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(2 4 8 16 32 64 128) # TODO: ...
run_experiment_cpu

N_TRAIN=(8192)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(4 8 16 32 64 128 256) # TODO: ...
run_experiment_cpu

N_TRAIN=(16384)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(8 16 32 64 128 256) # TODO: ...
run_experiment_cpu

N_TRAIN=(32768)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(16 32 64 128 256) # TODO: ...
run_experiment_cpu

# N_TRAIN=(65536)
# N_TEST=(1024 2048 4096) # TODO: ...
# N_TILES=(16 32 64 128 256) # TODO: ...
# run_experiment_cpu

# GPU

N_TRAIN=(1024)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

N_TRAIN=(2048)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

N_TRAIN=(4096)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

N_TRAIN=(8192)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

N_TRAIN=(16384)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

N_TRAIN=(32768)
N_TEST=(1024 2048 4096) # TODO: ...
N_TILES=(1 2 4 8 16 32) # TODO: ...
run_experiment_gpu

# N_TRAIN=(65536)
# N_TEST=(1024 2048 4096) # TODO: ...
# N_TILES=(4 8 16 32 64) # TODO: ...
# run_experiment_gpu
