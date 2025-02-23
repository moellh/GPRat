#!/bin/bash

#SBATCH -w simcl1n1
#SBATCH --job-name="gprat"
#SBATCH --output=job_gprat.out #SBATCH --time=24:00:00
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

# Test 1
# Cholesky
# CPU only
# fixed problem size
# increasing tile size
# increasing cores
echo "=== Starting Test 1 (incl. 5)"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/1-2-cholesky-cpu/
mkdir -p apex
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/1/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 1"

# Test 2
# Cholesky
# CPU only
# fixed problem size
# increasing tile size
# increasing cores
# split cholesky steps, BLAS calls
echo "=== Starting Test 2 (incl. 6)"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
cd experiment/1-2-cholesky-cpu/
mkdir -p apex
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/2/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 2"

# Test 3
# Cholesky
# GPU only
# fixed problem size
# increasing tile size
# increasing n_streams
echo "=== Starting Test 3 (incl. 5)"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/3-4-cholesky-gpu/
mkdir -p apex
./run_simcl1.sh
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
# split cholesky steps, BLAS calls
echo "=== Starting Test 4 (incl. 6)"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
cd experiment/3-4-cholesky-gpu/
mkdir -p apex
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/4/${timestamp}
mkdir -p ${results_dir}
cp output.csv ${results_dir}/output.csv
cp -r apex/ ${results_dir}/apex/
echo "=== Finished Test 4"

# Test 5,6
# Cholesky
# GPU, CPU
# increasing problem size, incl. 2^16
# increasing tile size
# opt n_cores and n_streams
# for test 6: split cholesky steps, BLAS calls
echo "=== Note: Test 5 and 6 are part of Test 1-4"

# Test 7
# Assembly
# GPU, CPU
# increasing problem size
# increasing n_tiles
# increasing n_reg
# opt n_cores and n_streams
echo "=== Starting Test 7"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_ASSEMBLY_ONLY=ON
cd experiment/7-cholesky-assembly/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/7/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 7"

# Test 8
# Cholesky
# GPU, CPU
# increasing problem size
# increasing n_tiles
# opt n_cores and n_streams
echo "=== Starting Test 7"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_CHOLESKY_STEPS=ON
cd experiment/8-cholesky-only/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/8/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 8"

# Test 9
# Predict
# GPU, CPU
# increasing problem size
# increasing n_tiles
# opt n_cores and n_streams
echo "=== Starting Test 9"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/9-10-predict/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/9/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 9"

# Test 10
# Predict
# GPU, CPU
# increasing problem size
# increasing n_tiles
# steps, BLAS
# opt n_cores and n_streams
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

# Test 11,12
# Predict
# CPU with increasing n_cores, GPU with increasing n_streams
# increasing problem size, incl. 2^16
# increasing tile size
echo "=== Starting Test 11,12"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/11-12-predict/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/11-12/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 11,12"

# Test 13
# Predict with Uncertainty
# GPU, CPU
# increasing problem size
# increasing n_tiles
# opt n_cores and n_streams
echo "=== Starting Test 13"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/13-14-predict-uncer/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/13/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 13"

# Test 14
# Predict with Uncertainty
# GPU, CPU
# increasing problem size
# increasing n_tiles
# steps, BLAS
# opt n_cores and n_streams
echo "=== Starting Test 14"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_PREDICT_UNCER_STEPS=ON
cd experiment/13-14-predict-uncer/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/14/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 14"

# Test 15
# Predict with full covariance
# GPU, CPU
# increasing problem size
# increasing n_tiles
# opt n_cores and n_streams
echo "=== Starting Test 15"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/15-16-predict-fullcov/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/15/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 15"

# Test 15
# GPyTorch, GPflow
# TODO:

# Test 16
# Predict with full covariance
# GPU, CPU
# increasing problem size
# increasing n_tiles
# steps, BLAS
# opt n_cores and n_streams
echo "=== Starting Test 16"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON -DGPRAT_PREDICT_UNCER_STEPS=ON
cd experiment/15-16-predict-fullcov/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/16/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 16"

# Test 17,18
# Predict Full Cov
# CPU with increasing n_cores, GPU with increasing n_streams
# increasing problem size, incl. 2^16
# increasing tile size
echo "=== Starting Test 17,18"
cd $gprat_dir
./compile_gpxpy_python_simcl1.sh -DGPXPY_WITH_CUDA=ON
cd experiment/17-18-predict-fullcov/
mkdir -p apex-cpu
mkdir -p apex-gpu
./run_simcl1.sh
timestamp=$(date +"%m-%d_%H-%M-%S")
results_dir=$HOME/results/17-18/${timestamp}
mkdir -p ${results_dir}
cp output-cpu.csv ${results_dir}/output-cpu.csv
cp output-gpu.csv ${results_dir}/output-gpu.csv
cp -r apex-cpu/ ${results_dir}/apex-cpu/
cp -r apex-gpu/ ${results_dir}/apex-gpu/
echo "=== Finished Test 17,18"

echo "Slurm job finished"
