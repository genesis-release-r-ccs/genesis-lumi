#!/bin/bash
#SBATCH --job-name=apo_p8
#SBATCH --account=XXXXXXXXXX
#SBATCH --time=02:00:00
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
export PMI_NO_PREINITIALIZE=y
module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
module load cray-libsci
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SLURM_CPU_BIND="mask_cpu:0xfefe000000000000,0xfefe0000,0xfefe,0xfefe00000000"
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp15.id000.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp15.id001.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gpu08.mpi08.omp15.id002.out
