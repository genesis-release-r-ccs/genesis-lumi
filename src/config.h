/* src/config.h.  Generated from config.h.in by configure.  */
/* src/config.h.in.  Generated from configure.ac by autoheader.  */

/* c compiler version */
#define COMPILE_CC_VER "gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"

/* c flags */
#define COMPILE_CFLAGS "-O3 -ffast-math -march=native -fallow-argument-mismatch -fopenmp "

/* defined variables */
#define COMPILE_DEFINED_VARIABLES " -DHAVE_MPI_GENESIS -DOMP -DFFTE -DLAPACK -DUSE_GPU -DCUDAGPU -DDSFMT_MEXP=19937 -D__GFORTRAN__"

/* fortran flags */
#define COMPILE_FCFLAGS " -O3 -ffast-math -march=native -ffree-line-length-none -fallow-argument-mismatch -fopenmp  "

/* fortran compiler version */
#define COMPILE_FC_VER "GNU Fortran (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"

/* genesis version */
#define COMPILE_GENESIS_VERSION "$GENESIS_VERSION$"

/* hostname */
#define COMPILE_HOST "foo"

/* ld flags */
#define COMPILE_LDFLAGS " -fopenmp  -llapack -lblas -L/usr/lib/cuda/lib64 -lcudart -lstdc++ "

/* cuda version */
#define COMPILE_NVCC_VER "Build cuda_11.2.r11.2/compiler.29373293_0"

/* username */
#define COMPILE_USER "bar"

/* defined if cuda_gpu is used. */
#define CUDAGPU 1

/* defined if Debug is used. */
/* #undef DEBUG */

/* defined always. */
#define DSFMT_MEXP 19937

/* defined if FFTE is used. */
#define FFTE 1

/* defined if fapp is used. */
/* #undef FJ_PROF_FAPP */

/* defined if fipp is used. */
/* #undef FJ_PROF_FIPP */

/* defined if fipp is used. */
/* #undef FJ_TIMER_2 */

/* defined if fipp is used. */
/* #undef FJ_TIMER_DETAIL */

/* defined if --host=Fugaku is set. */
/* #undef FUGAKU */

/* Define to 1 if you have the <bagel.h> header file. */
/* #undef HAVE_BAGEL_H */

/* Define to 1 if you have the <inttypes.h> header file. */
/* #undef HAVE_INTTYPES_H */

/* defined if MPI is used. */
#define HAVE_MPI_GENESIS 1

/* defined if MPI is used. */
/* #undef HAVE_MPI_H */

/* Define to 1 if you have the <stdint.h> header file. */
/* #undef HAVE_STDINT_H */

/* Define to 1 if you have the <stdio.h> header file. */
/* #undef HAVE_STDIO_H */

/* Define to 1 if you have the <stdlib.h> header file. */
/* #undef HAVE_STDLIB_H */

/* Define to 1 if you have the <strings.h> header file. */
/* #undef HAVE_STRINGS_H */

/* Define to 1 if you have the <string.h> header file. */
/* #undef HAVE_STRING_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
/* #undef HAVE_SYS_STAT_H */

/* Define to 1 if you have the <sys/types.h> header file. */
/* #undef HAVE_SYS_TYPES_H */

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* defined if HM_DISK is used. */
/* #undef HM_DISK */

/* defined if ifort version (>19) is used. */
/* #undef IFORT19 */

/* defined if Intel compiler is used. */
/* #undef INTEL */

/* defined if K-computer compiler is used. */
/* #undef KCOMP */

/* defined if LAPACK is used. */
#define LAPACK 1

/* MPI WRAPPER TYPE */
#define MPITYPE "Open"

/* defined if MS-MPI compiler is used. */
/* #undef MSMPI */

/* defined if OpenMP is used. */
#define OMP 1

/* Name of package */
#define PACKAGE "genesis"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "genesis@riken.jp"

/* Define to the full name of this package. */
#define PACKAGE_NAME "genesis"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "genesis 2.1.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "genesis"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.1.1"

/* defined if pgi and cuda are used. */
/* #undef PGICUDA */

/* defined if PKTIMER is used. */
/* #undef PKTIMER */

/* defined if QSimulate is used. */
/* #undef QSIMULATE */

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
/* #undef STDC_HEADERS */

/* defined if gpu is used. */
#define USE_GPU 1

/* Version number of package */
#define VERSION "2.1.1"

/* defined if _LARGE_INT is used. */
/* #undef _LARGE_INT */

/* defined if _MIXED is used. */
/* #undef _MIXED */

/* defined if _SINGLE is used. */
#define _SINGLE 1

/* defined if GCC gfortran compiler is used. */
#define __GFORTRAN__ 1

/* defined if pgi compiler is used. */
/* #undef __PGI */
