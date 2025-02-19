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
cd $gprat_dir
./compile_gpxpy_python.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
cd experiment/2-cholesky-cpu/
rm -rf apex
mkdir -p apex
./run.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/2/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 2"

echo "Slurm job finished"
