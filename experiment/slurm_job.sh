#!/bin/bash

#SBATCH -w simcl1n1,simcl1n2
#SBATCH --job-name="gprat"
#SBATCH --output=job_gprat.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

gprat_dir="/data/scratch/mllmannhk/GPRat"
msd_dir="$GPRat/data/generators/msd_simulator" # run_msd.sh
