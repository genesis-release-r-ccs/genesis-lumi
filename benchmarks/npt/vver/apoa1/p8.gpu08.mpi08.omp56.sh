#!/bin/bash
#SBATCH --job-name=apo_p8
#SBATCH --account=XXXXXXXXXX
#SBATCH --time=02:00:00
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
export PMI_NO_PREINITIALIZE=y
module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
module load cray-libsci
export OMP_NUM_THREADS=56
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SLURM_CPU_BIND="mask_cpu:0xfefefefefefefefe"
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp63.id000.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp63.id001.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp63.id002.out
