#!/bin/bash -e
#SBATCH --job-name=test
#SBATCH --account=Project_462000123
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH -o %x-%j.out

ml PrgEnv-gnu
ml craype-x86-trento
ml craype-accel-amd-gfx90a
ml rocm
ml cray-libsci

GENESIS_HOME="$(pwd)/../../"
GENESIS_PATH="${GENESIS_HOME}/src/spdyn/"
GENESIS_TEST="${GENESIS_HOME}/tests/regression_test/"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 "${GENESIS_TEST}/test.py" "${GENESIS_PATH}/spdyn" "lumi"
