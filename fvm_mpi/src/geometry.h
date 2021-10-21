#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include <mpi.h>
#include <stdlib.h>

#include "types.h"
#include "options.h"

// Global direction indicators
#define XN 0
#define XP 1 
#define YN 2
#define YP 3 
#define ZN 4
#define ZP 5

// Struct to hold all communication information for MPI runs
typedef struct {
  MPI_Comm comm;
  MPI_Comm comm3d;
  int size;         // Number of all processes. The one given in command line: mpirun -np N
  int rank;         // Number of a current process. One of 0,...,size-1
  int dims[NDIMS];  // Number of processes in each direction
  int coord[NDIMS]; // Coordinates of calling process in cartesian structure
  int neighbors[NDIMS*2];
  MPI_Status  status[NDIMS*2];
  MPI_Request send_req[NDIMS*2];
  MPI_Request receive_req[NDIMS*2];
  
} MPIInfo;

// Enumeration for boundary types
typedef enum {density=0, discrete} boundary_t;

// Enumeration for inner and outer boundaries
typedef enum {inner=0, outer} boundary_pos;

// Struct keeping the geometry information
struct Geometry {

  // The resolution
  REAL h;

  // The Geometry resolution
  REAL h_geom;

  // The volume of each geom voxel
  REAL dV;

  // Subdivisions (difference between model geometry and simulation)
  domain_id subdivisions;
  
  // The physical boundaries of the whole geometry
  REAL global_size[NDIMS];

  // The number of domains
  domain_id num_domains;

  // Global sizes geometry
  hsize_t N[NDIMS];
  
  // Local sizes geometry
  hsize_t n[NDIMS];

  // Offset into global data
  hsize_t offsets[NDIMS];

  // Domain ids
  domain_id* domain_ids;

  // Domain names
  char** domain_names;

  // The domain data
  domain_id* domains;

  // domain connections
  domain_id* domain_connections;

  // Domain volumes
  REAL* volumes;

  // Number of domain voxels
  long* domain_num;

  // Local domain volumes
  REAL* local_volumes;

  // Local number of voxels in each domain
  long* local_domain_num;

  // ghost domains
  domain_id* ghost_domains[NDIMS*2];

  // Num boundaries
  domain_id num_boundaries;

  // Boundary names
  char** boundary_names;

  // Boundary ids
  domain_id* boundary_ids;

  // Total size of each global boundary. Every process has the same values.
  size_t* boundary_size;

  // Size of each local boundary per process. It keeps the number of local
  // boundaries only. Does not count ghost boundaries.
  size_t* local_boundary_size;

  // Boundary type (right now only two types, density and discrete)
  boundary_t* boundary_types;
  
  // Boundary position either inner (between two domains) or outer
  boundary_pos* boundary_positions;

  // Number of boundaries that are discrete (NOT size of boundaries!)
  // e.g. if only ryr is discrete, then num_global_discrete_boundaries = 1
  unsigned short num_global_discrete_boundaries;

  // Local discrete boundary indices. Keeps the boundary indices.
  int* discrete_boundary_indices;
  int* from_bi_to_dbi_mapping;  // given a boundary index -> whether it's discete
  
  // Number of global discrete boundaries. Notice that ghost boundaries are
  // counted twice. It is significant not a mistake.
  int* num_discrete_boundaries;

  // Keep track of what boundaries are discrete
  // Local SIZE of discrete boundaries per process. It counts ghost boundaries
  // as well.
  int* num_local_discrete_boundaries;
  
  // Stores local sizes of discrete boundaries from all processes in 
  // an array accesible only from process with rank 0
  // num_local .. [dbi][rank] - size of discrete boundary at certain process
  int** num_local_discrete_boundaries_rank_0;
  // Exactly 2 times the number of local discrete boundaries
  int** num_local_species_values_at_discrete_boundaries_rank_0;

  // Offsets for discrete boundaries (Needed for communication)
  int** offset_num_local_discrete_boundaries_rank_0;
  int** offset_num_species_values_discrete_boundaries_rank_0;

  // The openess of discrete boundaries (Used in communication and flux computing)
  // When ghost boundary is found, its state is kept in both processes.
  // Thus, in the worst case when all discrete boundaries are ghost, the allocated
  // memory size will be twice as allocated memory size in the case when all boundaries
  // are local. Since the communication takes place only at processor with rank 0,
  // the double states will have exactly the same value. This approach helps to avoid
  // communication between processes that shares ghost boundaries.
  int** open_local_discrete_boundaries; 

  // All local discrete boundaries on rank 0 used to handle all discrete events
  // on rank 0 process
  int** open_local_discrete_boundaries_rank_0;
  int** open_local_discrete_boundaries_correct_order_rank_0; //used in species.c
  
  // Local distributions of global discrete boundaries per process (Needed for 
  // communication for correct evaluation of stochastic boundaries)
  // For ghost dicrete boundaries the information is double, but again, similarily
  // to open_local_discrete_boundaries, is exactly the same
  int** local_distribution_of_global_discrete_boundaries_rank_0;

  // Species values for each discrete boundary
  REAL** species_values_at_local_discrete_boundaries; // PL usunąc local
  REAL** species_values_at_local_discrete_boundaries_rank_0;
  REAL** species_values_at_local_discrete_boundaries_correct_order_rank_0;

  // Num local ghost boundaries for each boundary and dim2
  unsigned short** num_local_ghost_boundaries;

  // Local ghost boundaries for each boundary and dim2 and a three sized voxel for
  // local_ghost_boundaries[boundary_ind][dim2][0] direction of boundary: 
  //   0 -> inner to ghost
  //   1 -> ghost to inner
  // local_ghost_boundaries[boundary_ind][dim2][1] first coordinate of ghost voxel
  // local_ghost_boundaries[boundary_ind][dim2][2] second coordinate of ghost voxel
  unsigned short*** local_ghost_boundaries; 

  // Local boundary indices
  unsigned short** boundaries;

  // Struct keeping information about MPI
  MPIInfo mpi_info;

};

typedef struct Geometry Geometry_t;

// Construction of a Geometry
Geometry_t* Geometry_construct(MPI_Comm comm, arguments_t* arguments);

// Communicate ghost domain voxels
void Geometry_communicate_ghost(Geometry_t* geom);

// Destruction of a Geometry
void Geometry_destruct(Geometry_t* geom);

// Read domain connections
void Geometry_read_domain_connections(hid_t group_id, Geometry_t* geom);

// Read local to process voxel data
void Geometry_read_chunked_domain_data(hid_t group_id, Geometry_t* geom);

// Read boundaries collectively and remove non-local boundaries
void Geometry_read_boundaries(hid_t group_id, Geometry_t* geom);

// Output neighbor information
void Geometry_output_neighbor(Geometry_t* geom);

// Output domain and neighbor information
void Geometry_output_domain_and_boundary(Geometry_t* geom);

// Get boundary type from str
boundary_t Geometry_boundary_type_from_str(MPI_Comm comm, char* boundary_type_str);

// Get domain index from domain name
domain_id Geometry_get_domain_id(Geometry_t* geom, char* domain_name);

// Get boundary index from boundary name
domain_id Geometry_get_boundary_id(Geometry_t* geom, char* boundary_name);

// Get discrete boundary index from boundary index
domain_id Geometry_get_discrete_boundary_id(const Geometry_t* geom, domain_id boundary_id);

// Return what direction a certain boundary is in . It returns a dim2 value 
// between 0 and 5, where X direction: 0,1, Y direction: 2,3, Z direction: 4,5
domain_id  Geometry_get_boundary_dir(Geometry_t* geom, domain_id boundary_id, 
                                     unsigned short index);

#endif
