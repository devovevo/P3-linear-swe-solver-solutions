#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include <mpi.h>
#endif

#ifdef CUDA
#include <cuda.h>
#endif

#ifdef MPI
void gather_h();
void init_mpi(double *h, double *u, double *v, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank, int num_procs);
#else
void init(double *u, double *h, double *v, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_);
#endif

void step();

void free_memory();

#ifdef CUDA
void transfer_to_host();
#endif

#ifdef MPI
void gather_h();
#endif