#!/bin/bash

#SBATCH -w simcl1n3,simcl1n4
#SBATCH --job-name="gprat"
#SBATCH --output=job_gprat.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

gprat_dir="/data/scratch/mllmannhk/GPRat"
msd_dir="$GPRat/data/generators/msd_simulator"

# Download GPRat repo
rm -rf gprat_dir
git clone git@github.com:moellh/GPRat.git $gprat_dir
cd $gprat_dir
git checkout experiment

# Generate data
cd $msd_dir
./run_msd.sh
cd $gprat_dir

cd experiments/

cd 1-cholesky-cpu-f_ps-i_nt/
./run_simcl1.sh
cp output.csv ~/results/1/output.csv
cp -r apex/ ~/results/1/apex/
cd $gprat_dir
