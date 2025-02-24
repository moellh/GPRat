#!/bin/bash

#SBATCH -w simcl1n2
#SBATCH --job-name="gp_10"
#SBATCH --output=job_gprat-10.out #SBATCH --time=24:00:00
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

# Test 10
echo "=== Starting Test 10"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_PREDICT_STEPS=ON
cd experiment/9-10-predict/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/10/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 10"
