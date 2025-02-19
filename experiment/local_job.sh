#!/bin/bash

gprat_dir="$(realpath ../)"
msd_dir="$gprat_dir/data/generators/msd_simulator"

# Generate data
cd $msd_dir
./run_msd.sh

# Test 1
# Cholesky
# CPU only
# fixed problem size
# increasing tile size
# increasing cores
echo "=== Starting Test 1"
# cd $gprat_dir
# ./compile_gpxpy_python.sh -DGPXPY_WITH_CUDA=ON
# export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
# cd experiment/1-cholesky-cpu/
# rm -rf apex
# mkdir -p apex
# ./run.sh
# timestamp=$(date +"%m-%d_%H-%M-%S")
# results_dir=$HOME/results/1/${timestamp}
# mkdir -p ${results_dir}
# cp output.csv ${results_dir}/output.csv
# cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 1"

# Test 2
# Cholesky
# CPU only
# fixed problem size
# increasing tile size
# increasing cores
# split cholesky steps
echo "=== Starting Test 2"
# cd $gprat_dir
# ./compile_gpxpy_python.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
# export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
# cd experiment/2-cholesky-cpu/
# rm -rf apex
# mkdir -p apex
# ./run.sh
# timestamp=$(date +"%m-%d_%H-%M-%S")
# results_dir=$HOME/results/2/${timestamp}
# mkdir -p ${results_dir}
# cp output.csv ${results_dir}/output.csv
# cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 2"

# Test 3
# Cholesky
# GPU only
# fixed problem size
# increasing tile size
# increasing n_streams
echo "=== Starting Test 3"
cd $gprat_dir
./compile_gpxpy_python.sh -DGPXPY_WITH_CUDA=ON
export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
cd experiment/3-cholesky-gpu/
mkdir -p apex
./run.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/3/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 3"

# Test 4
# Cholesky
# GPU only
# fixed problem size
# increasing tile size
# increasing n_streams
# split cholesky steps, BLAS cally
echo "=== Starting Test 4"
# cd $gprat_dir
# ./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
# export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
# cd experiment/4-cholesky-gpu/
# mkdir -p apex
# ./run_simcl1.sh
# timestamp=$(date +"%m-%d_%H-%M-%S")
# results_dir=$HOME/results/4/${timestamp}
# mkdir -p ${results_dir}
# cp output.csv ${results_dir}/output.csv
# cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 4"

echo "Slurm job finished"
