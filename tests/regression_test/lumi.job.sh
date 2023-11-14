#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=XXXXXXXXXX
#SBATCH --time=02:00:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=8

module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
module load cray-libsci

GENESIS_HOME="$(pwd)/../../"
GENESIS_PATH="${GENESIS_HOME}/src/spdyn/"
GENESIS_TEST="${GENESIS_HOME}/tests/regression_test/"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 "${GENESIS_TEST}/test.py" "${GENESIS_PATH}/spdyn" "lumi" "gpu"
