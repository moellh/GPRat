#!/bin/bash

#SBATCH -w simcl1n1,simcl1n2
#SBATCH --job-name="gp_4"
#SBATCH --output=job_gprat-4.out #SBATCH --time=24:00:00
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

# Test 4
echo "=== Starting Test 4"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
cd experiment/test-4-cholesky-gpu/
mkdir -p apex
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/4/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 4"
