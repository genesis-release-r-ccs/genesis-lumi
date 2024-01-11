#!/bin/bash -e
#SBATCH --job-name=p128_node032
#SBATCH --account=Project_462000123
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=4
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
CPU_BIND="mask_cpu:"
CPU_BIND="${CPU_BIND}0xfefe000000000000,"
CPU_BIND="${CPU_BIND}0x00000000fefe0000,"
CPU_BIND="${CPU_BIND}0x000000000000fefe,"
CPU_BIND="${CPU_BIND}0x0000fefe00000000"
export SLURM_CPU_BIND="${CPU_BIND}"

# Warm up
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp

# Benchmark
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp 2>&1 | tee p128.node032.mpi04.omp14.gpu04.gnu.id000.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp 2>&1 | tee p128.node032.mpi04.omp14.gpu04.gnu.id001.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp 2>&1 | tee p128.node032.mpi04.omp14.gpu04.gnu.id002.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp 2>&1 | tee p128.node032.mpi04.omp14.gpu04.gnu.id003.out
srun "$(pwd)/../../../../src/spdyn/spdyn" p128.inp 2>&1 | tee p128.node032.mpi04.omp14.gpu04.gnu.id004.out
