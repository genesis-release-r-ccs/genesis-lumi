#!/bin/bash -e
#SBATCH --job-name=apo_p64
#SBATCH --account=Project_462000123
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -o %x-%j.out
export PMI_NO_PREINITIALIZE=y
module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
module load cray-libsci
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SLURM_CPU_BIND="mask_cpu:0xfefe000000000000,0xfefe0000,0xfefe,0xfefe00000000"
srun "$(pwd)/../../../../src/spdyn/spdyn" p64.inp 2>&1 | tee p64.cray.gpu64.mpi64.omp14.id000.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p64.inp 2>&1 | tee p64.cray.gpu64.mpi64.omp14.id001.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p64.inp 2>&1 | tee p64.cray.gpu64.mpi64.omp14.id002.out
