#include <math.h>
#include <stdlib.h>
#include <hdf5.h>
#include <assert.h>
#include <time.h>

#include "types.h"
#include "utils.h"

//-----------------------------------------------------------------------------
const static char* timing_names[] = {"Setup", "Init species", "Construct geometry", "Construct species", 
                                     "Communicate ghost values", "Diffusion stencil", 
                                     "Diffusion stencil borders", "Boundary fluxes", 
                                     "Stochastic evaluation", "Reaction stencil", "Fixed Domains", "Apply du", "Convolution", "Write",
                                     "Diffusion stencil borders X", "Diffusion stencil borders Y", "Diffusion stencil borders Z", 
				     "Main Loop time", "Diffusion Total", "Simulation time"};
static unsigned int timing_status[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static unsigned int timing_calls[]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static REAL timings[] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
#define NUM_TIMINGS 20

//-----------------------------------------------------------------------------
REAL timing()
{
  REAL time;
  struct timespec timmer;
  clock_gettime(CLOCK_MONOTONIC, &timmer);
  time = timmer.tv_sec + timmer.tv_nsec*1e-9;
  return time;
}
//-----------------------------------------------------------------------------
void push_time(timings_t what)
{
  assert(!timing_status[what]);
  timing_status[what] = 1;
  timings[what] -= timing();
}
//-----------------------------------------------------------------------------
void pop_time(timings_t what)
{
  assert(timing_status[what]);
  timing_status[what] = 0;
  timings[what] += timing();
  timing_calls[what]++;
}
//-----------------------------------------------------------------------------
void output_timings(MPI_Comm comm)
{
  domain_id i;
  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "%30s %10s %10s %10s %10s\n", "Timings", "avg", "max", "min", "Num");
  mpi_printf0(comm, "---------------------------------------------------------------\n");
  double smax=0, smin=0, savg=0;
  int size;
  MPI_Comm_size(comm, &size);
  for (i=0; i<NUM_TIMINGS; i++)
  {
    assert(!timing_status[i]);
    double tmax, tmin, tavg;
    //MPI_Reduce(&timings[i], &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings[i], &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&timings[i], &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&timings[i], &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    tavg /= size;
    if(i <=WRITE) {
      smax += tmax;
      smin += tmin;
      savg += tavg;
    }
    mpi_printf0(comm, "%30s|%10.1f|%10.1f|%10.1f|%10d\n", timing_names[i], tavg, tmax, tmin, timing_calls[i]);
  }
  double diffusion = timings[GHOST] + timings[DIFF] + timings[BORDERS];

  mpi_printf0(comm, "%30s|%10.1f|%10.1f|%10.1f|%10d\n", "sum", savg, smax, smin, 0);
}
//-----------------------------------------------------------------------------
void read_h5_attr(MPI_Comm comm, hid_t group_id, char* obj_name, char* attr_name, 
		  void* buff)
{
  hid_t attr_id = H5Aopen_by_name(group_id, obj_name, attr_name, H5P_DEFAULT, H5P_DEFAULT);
  if (attr_id==-1)
  {
    mpi_printf_error0(comm, "Could not open attr \"%s%s\"\n\n", obj_name, attr_name);
    MPI_Abort(comm, 1);
  }
  
  hid_t atype = H5Aget_type(attr_id);
  H5Aread(attr_id, atype, buff);

  // If string we add end str character (Note only accepts single strings...)
  if (H5T_STRING == H5Tget_class(atype))
  {
    hsize_t size = H5Tget_size(atype);
    char* str_buff = (char*)buff;
    str_buff[size] = '\0';
  }
  H5Aclose(attr_id);
  
}
//-----------------------------------------------------------------------------
void write_h5_attr(MPI_Comm comm, hid_t group_id, char* attr_name, hid_t H5_TYPE, 
		   hsize_t dims, void* buff)
{

  // Write attr to group
  hid_t attr_space;
  
  // If writing string attribute
  if (H5_TYPE == H5T_STRING)
  {
    attr_space = H5Screate(H5S_SCALAR);
    H5_TYPE = H5Tcopy(H5T_C_S1);
    H5Tset_strpad(H5_TYPE, H5T_STR_NULLTERM);
    H5Tset_size(H5_TYPE, dims);
  }

  // Create a simple data space
  else
  {
    attr_space = H5Screate_simple(1, &dims, NULL);
  }

  hid_t attr_id = H5Acreate2(group_id, attr_name, H5_TYPE, attr_space, \
			    H5P_DEFAULT, H5P_DEFAULT);

  if (attr_id==-1)
  {
    mpi_printf_error0(comm, "Could not create attr \"%s\"\n\n", attr_name);
    MPI_Abort(comm, 1);
  }
  
  H5Awrite(attr_id, H5_TYPE, buff);

  // Clean up
  H5Aclose(attr_id);
  H5Sclose(attr_space);
  
}
//-----------------------------------------------------------------------------
void *mpi_malloc(MPI_Comm comm, size_t bytes)
{
  // Allocate memory
  void *buffer = malloc(bytes);
  
  // Check for error
  if (buffer == NULL) 
  {
    int rank; 
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "Error: Malloc failed for process %d\n", rank);
    fflush (stderr);
    MPI_Abort (comm, 4);
  }
  
  return buffer;
}
//-----------------------------------------------------------------------------
domain_id max(domain_id a, domain_id b)
{
  return  a > b ? a : b;
}
//-----------------------------------------------------------------------------
int max_int(int a, int b)
{
  return  a > b ? a : b;
}
//-----------------------------------------------------------------------------
void memfill(REAL* buff, size_t size, REAL value)
{
  
  size_t i;
  for (i=0; i<size; i++)
    buff[i] = value;

}
//-----------------------------------------------------------------------------
void memfill_size_t(size_t* buff, size_t size, size_t value)
{
  
  size_t i;
  for (i=0; i<size; i++)
    buff[i] = value;

}
//-----------------------------------------------------------------------------
void memfill_long(long* buff, size_t size, long value)
{
  
  size_t i;
  for (i=0; i<size; i++)
    buff[i] = value;

}
//-----------------------------------------------------------------------------
int mpi_max_int(MPI_Comm comm, int value)
{
  int global_max;
  MPI_Allreduce(&value, &global_max, 1, MPI_INT, MPI_MAX, comm);
  return global_max;
}
//-----------------------------------------------------------------------------
int mpi_sum_int(MPI_Comm comm, int value)
{
  int global_sum;
  MPI_Allreduce(&value, &global_sum, 1, MPI_INT, MPI_SUM, comm);
  return global_sum;
}
//-----------------------------------------------------------------------------
REAL gauss(REAL x, REAL sigma)
{
  const REAL M_1_SQRT_SIGMA = 1.0/sqrt(sigma);
  return M_SQRT1_2*M_2_SQRTPI*M_1_SQRT_SIGMA/2*exp(-x*x/2/sigma);
}
//-----------------------------------------------------------------------------
int mod(int dividend, int divisor)
{
  return dividend - divisor*(int)floor(1.0*dividend/divisor);
}
