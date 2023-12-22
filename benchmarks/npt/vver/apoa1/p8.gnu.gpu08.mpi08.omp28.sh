#!/bin/bash -e
#SBATCH --job-name=apo_p8
#SBATCH --account=Project_462000123
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH -o %x-%j.out
export PMI_NO_PREINITIALIZE=y
module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm
module load cray-libsci
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SLURM_CPU_BIND="mask_cpu:0xfefe0000fefe0000,0xfefe0000fefe"
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gnu.gpu08.mpi08.omp28.id000.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gnu.gpu08.mpi08.omp28.id001.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p8.inp 2>&1 | tee p8.gnu.gpu08.mpi08.omp28.id002.out
