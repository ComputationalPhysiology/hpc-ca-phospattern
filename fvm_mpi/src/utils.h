#ifndef UTILS_H
#define UTILS_H
#include <mpi.h>
#include "types.h"

// Timing types
static REAL timings[];
static const char* timing_names[];
typedef enum {SETUP=0, INIT, GEOM, SPECIES, GHOST, DIFF, BORDERS, BOUNDARIES, STOCH, REACT, FIXED, APPLYDU, CONV, WRITE, BORDERSX, BORDERSY, BORDERSZ, MAINLOOP, DIFFTOT, SIM} timings_t;

// Timing help functions
void push_time(timings_t what);

// Timing help functions
void pop_time(timings_t what);

// Output time info
void output_timings();

// Hack that allows to print a message only on the first proc
#define mpi_printf0(comm, args...) do { int __rank; MPI_Comm_rank(comm, &__rank); if (__rank == 0) { printf(args); } } while(0)

// Hack that allows to print an error message only on the first proc
#define mpi_printf_error0(comm, args...) do { int __rank; MPI_Comm_rank(comm, &__rank); if (__rank == 0) { fprintf(stderr, args); } } while(0)

// Hack to print an error message and then abort
#define mpi_printf_error(comm, args...) do { int __rank; MPI_Comm_rank(comm, &__rank); if (__rank == 0) { fprintf(stderr, args); MPI_Abort(comm, 1);} } while(0)

// Max methods
domain_id max(domain_id a, domain_id b);
int max_int(int a, int b);

// Read an attribute realtive to a group
void read_h5_attr(MPI_Comm comm, hid_t group_id, char* obj_name, char* attr_name, 
    void* buff);

// Write an attribute realtive to a group
void write_h5_attr(MPI_Comm comm, hid_t group_id, char* attr_name, hid_t H5TYPE, 
    hsize_t dims, void* buff);

// Help function to allocate memory
void *mpi_malloc(MPI_Comm comm, size_t bytes);

// Fill memory of a REAL buffer with a value
void memfill(REAL* buff, size_t size, REAL value);

// Fill memory of a size_t buffer with a value
void memfill_size_t(size_t* buff, size_t size, size_t value);

// Fill memory of a size_t buffer with a value
void memfill_long(long* buff, size_t size, long value);

// Max over all processors
int mpi_max_int(MPI_Comm comm, int value);

// Sum over all processors
int mpi_sum_int(MPI_Comm comm, int value);

// Gaussian function
REAL gauss(REAL x, REAL sigma);

// Compute module using floored division. This way the remainder would 
// have the same sign as the divisor
int mod(int dividend, int divisor);

#endif
