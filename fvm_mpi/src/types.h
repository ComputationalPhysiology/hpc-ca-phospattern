#ifndef TYPES_H
#define TYPES_H
#include <stdlib.h>
#include <hdf5.h>
#include <float.h>

// Structs to hold parameters for reaction diffusion equations
#define NDIMS 3
#define MAX_SPECIES_NAME 20
#define MAX_FILE_NAME 100

#ifndef DOUBLE
#define DOUBLE 0
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#if DOUBLE
typedef double REAL;
#define H5REAL H5T_IEEE_F64LE
#define MPIREAL MPI_DOUBLE
#else
typedef float REAL;
#define H5REAL H5T_IEEE_F32LE
#define MPIREAL MPI_FLOAT
#endif

#define H5U8 H5T_STD_U8LE
typedef unsigned char domain_id;
#define MPIDOMAIN_ID MPI_UNSIGNED_CHAR

#define X 0
#define Y 1
#define Z 2

// Directions of process splitting
#define XY  0
#define YZ  1
#define XZ  2
#define XYZ 3

// Scale factor for automatically setting the diffusion constant based
// on stability criteria. 
#define DT_SCALE 0.95

#endif
