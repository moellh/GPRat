#!/bin/bash

#SBATCH -w simcl1n1
#SBATCH --job-name="gp_test"
#SBATCH --output=job_gprat-test.out
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

# Test
echo "=== Starting Test test ==="
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/test-test-predict-fullcov-gprat/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/test/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test test"
