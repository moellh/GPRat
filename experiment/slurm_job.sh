#!/bin/bash

#SBATCH -w simcl1n1
#SBATCH --job-name="gprat"
#SBATCH --output=job_gprat.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

gprat_dir="/data/scratch/mllmanhk/GPRat"
msd_dir="$gprat_dir/data/generators/msd_simulator"

# Download GPRat repo
mkdir -p $gprat_dir
rm -rf $gprat_dir
git clone https://github.com/moellh/GPRat.git $gprat_dir
cd $gprat_dir
git switch experiment

# Generate data
cd $msd_dir
./run_msd.sh

cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
export PYTHONPATH=$PYTHONPATH:${gprat_dir}/examples/gpxpy_python/install_python/
cd 1-cholesky-cpu-f_ps-i_nt/
./run_simcl1.sh
cp output.csv ~/results/1/output.csv
cp -r apex/ ~/results/1/apex/
