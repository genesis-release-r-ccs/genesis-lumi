#!/bin/bash -e
sbatch p004.node001.mpi04.omp14.gpu04.cray.sh
sbatch p004.node002.mpi02.omp28.gpu02.cray.sh
sbatch p004.node004.mpi01.omp56.gpu01.cray.sh
sbatch p008.node001.mpi08.omp07.gpu08.cray.sh
sbatch p008.node002.mpi04.omp14.gpu04.cray.sh
sbatch p008.node004.mpi02.omp28.gpu02.cray.sh
sbatch p008.node008.mpi01.omp56.gpu01.cray.sh
sbatch p016.node002.mpi08.omp07.gpu08.cray.sh
sbatch p016.node004.mpi04.omp14.gpu04.cray.sh
sbatch p016.node008.mpi02.omp28.gpu02.cray.sh
sbatch p016.node016.mpi01.omp56.gpu01.cray.sh
sbatch p032.node004.mpi08.omp07.gpu08.cray.sh
sbatch p032.node008.mpi04.omp14.gpu04.cray.sh
sbatch p032.node016.mpi02.omp28.gpu02.cray.sh
sbatch p032.node032.mpi01.omp56.gpu01.cray.sh
sbatch p064.node008.mpi08.omp07.gpu08.cray.sh
sbatch p064.node016.mpi04.omp14.gpu04.cray.sh
sbatch p064.node032.mpi02.omp28.gpu02.cray.sh
sbatch p064.node064.mpi01.omp56.gpu01.cray.sh
