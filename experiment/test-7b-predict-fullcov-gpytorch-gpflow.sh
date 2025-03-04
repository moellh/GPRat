#!/bin/bash

#SBATCH -w simcl1n1
#SBATCH --job-name="gp_7b"
#SBATCH --output=job_gprat-7b.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

gprat_dir="/data/scratch/mllmanhk/GPRat"
msd_dir="$gprat_dir/data/generators/msd_simulator"
export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/

# Download GPRat repo
mkdir -p $gprat_dir
rm -rf $gprat_dir
git clone https://github.com/moellh/GPRat.git $gprat_dir
cd $gprat_dir
git switch experiment

# Generate data
cd $msd_dir
./run_msd.sh

# Test 7b
echo "=== Starting Test 7b (GPyTorch, GPflow) ==="
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/test-7b-predict-fullcov-gpytorch-gpflow/
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/7b/${timestamp}
mkdir -p ${results_dir}/gpytorch
mkdir -p ${results_dir}/gpflow
cp gpytorch/output-cpu.csv ${results_dir}/gpytorch/output-cpu.csv
cp gpytorch/output-gpu.csv ${results_dir}/gpytorch/output-gpu.csv
cp gpflow/output-cpu.csv ${results_dir}/gpflow/output-cpu.csv
cp gpflow/output-gpu.csv ${results_dir}/gpflow/output-gpu.csv
echo "=== Finished Test 7b"
