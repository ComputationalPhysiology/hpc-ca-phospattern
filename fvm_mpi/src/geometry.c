#include <hdf5.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "utils.h"
#include "geometry.h"
#include "options.h"

//disable OpenMP warnings for ICC
#pragma warning(disable:3180)

//-----------------------------------------------------------------------------
const char* boundary_type_strs[2] = {"density", "discrete"};
//-----------------------------------------------------------------------------
Geometry_t* Geometry_construct(MPI_Comm comm, arguments_t* arguments)
{
  
  unsigned int i, di, xi, yi, zi;

  // Create geometry struct
  Geometry_t* geom = mpi_malloc(comm, sizeof(Geometry_t));
  MPIInfo* mpi_info = &geom->mpi_info;

  // Set MPI communicators
  mpi_info->comm = comm;
  
  // Get size and rank
  MPI_Comm_size(comm, &mpi_info->size);
  MPI_Comm_rank(comm, &mpi_info->rank);

  // Create the 3D (2D) cartesian grid
  // FIXME: Not dimension independent code!
  
  // Check first if the --mpi_dist option was provided
  if (arguments->mpi_dist)
  {
    unsigned int offset;
    for (offset=0; offset<3; offset++)
      mpi_info->dims[offset] = arguments->mpi_dist[offset];
  }
  else // --mpi_dist was not provided
  {
    unsigned int proc_no, offset;
    
    switch (arguments->split_plane)
    {
      case XY:
        mpi_info->dims[0] = sqrt(mpi_info->size);
        mpi_info->dims[1] = sqrt(mpi_info->size);
        mpi_info->dims[2] = 1;
        break;
      case YZ:
        mpi_info->dims[0] = 1;
        mpi_info->dims[1] = sqrt(mpi_info->size);
        mpi_info->dims[2] = sqrt(mpi_info->size);
        break;
      case XZ:
        mpi_info->dims[0] = sqrt(mpi_info->size);
        mpi_info->dims[1] = 1;
        mpi_info->dims[2] = sqrt(mpi_info->size);
        break;
      case XYZ:
        proc_no = pow(mpi_info->size, 1/3.);
        for (offset=0; offset<3; offset++)
          mpi_info->dims[offset] = proc_no;
        break;
    }
  }
  
  int periods[NDIMS] = {0,0,0};
  int reorganisation = 0;
  int ndims = NDIMS;
  MPI_Cart_create(comm, ndims, mpi_info->dims, periods, reorganisation, &mpi_info->comm3d);

  // Get the local coord and neigbor information about the cartesian block
  MPI_Cart_get(mpi_info->comm3d, NDIMS, mpi_info->dims, periods, mpi_info->coord);
  int* neighbors = mpi_info->neighbors;

  for (i=0; i<NDIMS*2; i++)
    neighbors[i] = MPI_PROC_NULL;

  MPI_Cart_shift(mpi_info->comm3d, 0, 1, &neighbors[XN], &neighbors[XP]);
  MPI_Cart_shift(mpi_info->comm3d, 1, 1, &neighbors[YN], &neighbors[YP]);
  MPI_Cart_shift(mpi_info->comm3d, 2, 1, &neighbors[ZN], &neighbors[ZP]);
  
  // Output neighbor info
  if(arguments->verbose)
    Geometry_output_neighbor(geom);
  
  // Start reading geometry file
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
  
  // Create a new file collectively and release property list identifier.
  hid_t file_id = H5Fopen(arguments->geometry_file, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);
  
  // Read global attributes
  read_h5_attr(comm, file_id, "/", "global_size", geom->global_size);
  read_h5_attr(comm, file_id, "/", "h", &geom->h_geom);

  // Compute subdivision
  geom->subdivisions = fmax(1.,floor(geom->h_geom/arguments->h));
  geom->h = geom->h_geom/geom->subdivisions;

  // --- READ DOMAINS ---

  // Open domain group
  hid_t domain_idd = H5Gopen(file_id, "domains", H5P_DEFAULT);

  // Read domain specific attributes and create space to read 
  read_h5_attr(comm, file_id, "domains", "num", &geom->num_domains);
  geom->domain_names = mpi_malloc(comm, sizeof(char*)*geom->num_domains);
  geom->domain_ids = mpi_malloc(comm, sizeof(domain_id)*geom->num_domains);
  geom->volumes = mpi_malloc(comm, sizeof(REAL)*geom->num_domains);
  geom->local_volumes = mpi_malloc(comm, sizeof(REAL)*geom->num_domains);
  geom->domain_num = mpi_malloc(comm, sizeof(long)*geom->num_domains);
  geom->local_domain_num = mpi_malloc(comm, sizeof(long)*geom->num_domains);
  memfill(geom->volumes, geom->num_domains, 0.);
  memfill(geom->local_volumes, geom->num_domains, 0.);
  memfill_long(geom->local_domain_num, geom->num_domains, 0);
  memfill_long(geom->domain_num, geom->num_domains, 0);

  // Read domain information
  for (i=0; i<geom->num_domains; i++)
  {
    char name[MAX_SPECIES_NAME];
    sprintf(name,"name_%d", i);
    geom->domain_names[i] = (char*)mpi_malloc(comm, sizeof(char*)*MAX_SPECIES_NAME);
    read_h5_attr(comm, file_id, "domains", name, geom->domain_names[i]);
  }

  // Read domain indices
  read_h5_attr(comm, file_id, "domains", "indices", geom->domain_ids);

  // Read chunked domain data
  Geometry_read_chunked_domain_data(domain_idd, geom);

  // Read in domain connections
  Geometry_read_domain_connections(domain_idd, geom);

  // Communicate ghost domains
  Geometry_communicate_ghost(geom);

  // Compute domain volumes
  geom->dV = geom->h_geom*geom->h_geom*geom->h_geom;
  //#pragma omp parallel private(yi, zi, di)
  {
  long *tmp = calloc(geom->num_domains, sizeof(long));
  //#pragma omp for collapse(2)
  for (xi=0; xi<geom->n[X]; xi++)
  {
    for (yi=0; yi<geom->n[Y]; yi++)
    {
      for (zi=0; zi<geom->n[Z]; zi++)
      {
        if (geom->domains[xi*geom->n[Y]*geom->n[Z]+yi*geom->n[Z]+zi]>=geom->num_domains)
          printf("ERROR:");
        di = geom->domains[xi*geom->n[Y]*geom->n[Z]+yi*geom->n[Z]+zi];
	tmp[di]++;
        //geom->local_domain_num[di] += 1;
      }
    }
  }
  //#pragma omp critical
  for(size_t i=0; i<geom->num_domains; i++) geom->local_domain_num[i] += tmp[i];
  free(tmp);
  }

  for (di=0; di<geom->num_domains; di++)
    geom->local_volumes[di] = geom->local_domain_num[di]*geom->dV;
  
  REAL* buff = NULL;
  long* buff_size = NULL;
  if (mpi_info->rank==0)
  {
    buff = mpi_malloc(comm, sizeof(REAL)*mpi_info->size*geom->num_domains);
    memfill(buff, mpi_info->size*geom->num_domains, 0.);
    buff_size = mpi_malloc(comm, sizeof(long)*mpi_info->size*geom->num_domains);
    memfill_long(buff_size, mpi_info->size*geom->num_domains, 0);
  }

  // Send all local volumes to processor 0
  MPI_Gather(geom->local_volumes, geom->num_domains, MPIREAL, 
	     buff, geom->num_domains, MPIREAL, 0, mpi_info->comm);

  MPI_Gather(geom->local_domain_num, geom->num_domains, MPI_LONG, 
	     buff_size, geom->num_domains, MPI_LONG, 0, mpi_info->comm);

  // Sum all volumes and broadcast them to other processes
  if (mpi_info->rank==0)
  {
    size_t sum_voxels = 0;
    for (i=0; i<mpi_info->size; i++)
    {
      for (di=0; di<geom->num_domains; di++)
      {
        geom->volumes[di] += buff[i*geom->num_domains+di];
        geom->domain_num[di] += buff_size[i*geom->num_domains+di];
        sum_voxels += buff_size[i*geom->num_domains+di];
      }
    }
    
    // Sanity check!
    if (sum_voxels != geom->N[X]*geom->N[Y]*geom->N[Z])
      printf("ERROR: sum_voxels (%zu) != (%zu) geom->N[X]*geom->N[Y]*geom->N[Z]\n", 
	     sum_voxels, (size_t)(geom->N[X]*geom->N[Y]*geom->N[Z]));
    
    for (i=0; i<mpi_info->size; i++)
    {
      for (di=0; di<geom->num_domains; di++)
      {
        buff[i*geom->num_domains+di] = geom->volumes[di];
        buff_size[i*geom->num_domains+di] = geom->domain_num[di];
      }
    }
  } 

  // Send computed volumes from process 0 to all other
  MPI_Scatter(buff, geom->num_domains, MPIREAL,
	      geom->volumes, geom->num_domains, MPIREAL, 0, mpi_info->comm);

  MPI_Scatter(buff_size, geom->num_domains, MPI_LONG, 
	      geom->domain_num, geom->num_domains, MPI_LONG, 0, mpi_info->comm);

  // Free buffer on rank 0
  if (mpi_info->rank==0)
  {
    free(buff);
    free(buff_size);
  }  

  // --- READ BOUNDARIES ---

  // Open boundary group
  hid_t boundary_idd = H5Gopen(file_id, "boundaries", H5P_DEFAULT);

  // Read boundary specific attributes and create space to read 
  read_h5_attr(comm, file_id, "boundaries", "num", &geom->num_boundaries);
  geom->boundary_names = mpi_malloc(comm, sizeof(char*)*geom->num_boundaries);
  geom->boundary_ids = mpi_malloc(comm, sizeof(domain_id)*geom->num_boundaries);
  geom->boundary_types = mpi_malloc(comm, sizeof(boundary_t)*geom->num_boundaries);
  geom->boundary_positions = mpi_malloc(comm, sizeof(boundary_pos)*geom->num_boundaries);

  // Global boundary sizes
  geom->boundary_size = mpi_malloc(comm, sizeof(size_t)*geom->num_boundaries);
  memset(geom->boundary_size, 0, sizeof(size_t)*geom->num_boundaries);

  // local boundary sizes
  geom->local_boundary_size = mpi_malloc(comm, sizeof(size_t)*geom->num_boundaries);
  memset(geom->local_boundary_size, 0, sizeof(size_t)*geom->num_boundaries);
  geom->num_global_discrete_boundaries = 0;

  // All local boundaries
  geom->boundaries = mpi_malloc(comm, sizeof(unsigned short*)*geom->num_boundaries);
  memset(geom->boundaries, 0, sizeof(unsigned short*)*geom->num_boundaries);

  // Local ghost boundaries
  geom->num_local_ghost_boundaries = mpi_malloc(comm, sizeof(unsigned short*)*geom->num_boundaries);
  geom->local_ghost_boundaries = mpi_malloc(comm, sizeof(unsigned short**)*geom->num_boundaries);

  // Read boundary information
  for (i=0; i<geom->num_boundaries; i++)
  {
    char name[MAX_SPECIES_NAME];
    char boundary_type_str[MAX_SPECIES_NAME];
    sprintf(name,"name_%d", i);
    geom->boundary_names[i] = (char*)mpi_malloc(comm, sizeof(char*)*MAX_SPECIES_NAME);
    read_h5_attr(comm, file_id, "boundaries", name, geom->boundary_names[i]);
    geom->boundary_ids[i] = i;
    sprintf(name,"type_%d", i);
    read_h5_attr(comm, file_id, "boundaries", name, boundary_type_str);
    geom->boundary_types[i] = Geometry_boundary_type_from_str(comm, boundary_type_str);
    geom->num_global_discrete_boundaries += geom->boundary_types[i] == discrete ? 1 : 0;
    
    // Init num local boundaries
    geom->num_local_ghost_boundaries[i] = mpi_malloc(comm, sizeof(unsigned short)*NDIMS*2);
    memset(geom->num_local_ghost_boundaries[i], 0, sizeof(unsigned short)*NDIMS*2);

    // Local ghost boundaries
    geom->local_ghost_boundaries[i] = mpi_malloc(comm, sizeof(unsigned short*)*NDIMS*2);
    memset(geom->local_ghost_boundaries[i], 0, sizeof(unsigned short*)*NDIMS*2);
  }

  // Init information about number of local discrete boundaries.  This
  // information should be used in MPI communication.
  geom->num_local_discrete_boundaries = mpi_malloc(comm, sizeof(int)*\
                                                   geom->num_global_discrete_boundaries);
  memset(geom->num_local_discrete_boundaries, 0, sizeof(int)*\
         geom->num_global_discrete_boundaries);

  geom->from_bi_to_dbi_mapping = mpi_malloc(comm, sizeof(int)*geom->num_boundaries);

  geom->discrete_boundary_indices = mpi_malloc(comm, sizeof(int)*	\
                                               geom->num_global_discrete_boundaries);

  geom->num_discrete_boundaries = mpi_malloc(comm, sizeof(int)* \
                                             geom->num_global_discrete_boundaries);

  // Init information about the number of discrete boundaries on each processor. 
  // This information should be used in MPI communication.
  geom->open_local_discrete_boundaries = mpi_malloc(comm, sizeof(int*)*\
                                                    geom->num_global_discrete_boundaries);
  geom->open_local_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->open_local_discrete_boundaries_correct_order_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->num_local_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->num_local_species_values_at_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->offset_num_local_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->offset_num_species_values_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);
  geom->species_values_at_local_discrete_boundaries  = \
    mpi_malloc(comm, sizeof(REAL*)*geom->num_global_discrete_boundaries);
  geom->species_values_at_local_discrete_boundaries_rank_0  = \
    mpi_malloc(comm, sizeof(REAL*)*geom->num_global_discrete_boundaries);
  geom->species_values_at_local_discrete_boundaries_correct_order_rank_0  = \
    mpi_malloc(comm, sizeof(REAL*)*geom->num_global_discrete_boundaries);
  geom->local_distribution_of_global_discrete_boundaries_rank_0 = \
    mpi_malloc(comm, sizeof(int*)*geom->num_global_discrete_boundaries);

  // Set default pointer value to NULL facilitating no-clean up for all non 
  // rank 0 processes
  for (i=0; i<geom->num_global_discrete_boundaries; i++)
  {
    geom->open_local_discrete_boundaries_rank_0[i] = NULL;
    geom->open_local_discrete_boundaries_correct_order_rank_0[i] = NULL;
    geom->num_local_discrete_boundaries_rank_0[i] = NULL;
    geom->num_local_species_values_at_discrete_boundaries_rank_0[i] = NULL;
    geom->offset_num_local_discrete_boundaries_rank_0[i] = NULL;
    geom->offset_num_species_values_discrete_boundaries_rank_0[i] = NULL;
    geom->species_values_at_local_discrete_boundaries_rank_0[i] = NULL;
    geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[i] = NULL;
    geom->local_distribution_of_global_discrete_boundaries_rank_0[i] = NULL;
  }

  // Init memory for open local discrete
  // This information should be used in MPI communication.
  // Read boundaries
  Geometry_read_boundaries(boundary_idd, geom);

  // Output global size info
  if(arguments->verbose)
    Geometry_output_domain_and_boundary(geom);

  // Close groups and file
  H5Gclose(domain_idd);
  H5Gclose(boundary_idd);
  H5Fclose(file_id);
  
  return geom;
}
//-----------------------------------------------------------------------------
void Geometry_communicate_ghost(Geometry_t* geom)
{
  unsigned int ii, oi, dim, dim2, odim, idim, domain_offsets, offset_o;
  unsigned int domain_outer_offsets;
  unsigned int domain_inner_offsets;
  unsigned int size_ghost_values;
  MPI_Comm comm = geom->mpi_info.comm;

  domain_id* ghost_values_send[NDIMS*2];

  memset(ghost_values_send, 0, sizeof(domain_id*)*NDIMS*2);
  memset(geom->ghost_domains, 0, sizeof(domain_id*)*NDIMS*2);

  for (dim2=0; dim2<NDIMS*2; dim2++)
  {
    // X, Y, Z dimension
    size_t loc_dim = dim2/2;
    unsigned char left_right = dim2 % 2;

    // YZ sheet (every thing is contigous)
    if (loc_dim == X)
    {

      // Sheet offset
      if (left_right==0)
        domain_offsets = 0;
      else
        domain_offsets = (geom->n[X]-1)*geom->n[Y]*geom->n[Z];

      // Each outer iteration is a stave
      domain_outer_offsets = geom->n[Z];

      // Contigous!
      domain_inner_offsets = 1;
    }

    // XZ sheet
    else if (loc_dim == Y)
    {
      // Sheet offset
      if (left_right==0)
        domain_offsets = 0;
      else
        domain_offsets = (geom->n[Y]-1)*geom->n[Z];
      
      // Each outer iteration is a sheet
      domain_outer_offsets = geom->n[Y]*geom->n[Z];

      // Contigous!
      domain_inner_offsets = 1;
    }

    // YZ sheet
    else
    {

      // Sheet offset
      if (left_right==0)
        domain_offsets = 0;
      else
        domain_offsets = geom->n[Z]-1;
      
      // Each outer iteration is a sheet
      domain_outer_offsets = geom->n[Y]*geom->n[Z];

      // Each inner iteration is a stave
      domain_inner_offsets = geom->n[Z];
    }

    // If neighbor
    if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL)
    {
      
      // Allocate memory 
      size_ghost_values = 1;
      for (dim=0; dim<NDIMS; dim++)
        if (loc_dim!=dim)
          size_ghost_values*=geom->n[dim];
      
      ghost_values_send[dim2] = mpi_malloc(comm, sizeof(domain_id)*size_ghost_values);
      
      // Allocate ghosted domain values
      geom->ghost_domains[dim2] = mpi_malloc(comm, sizeof(domain_id)*size_ghost_values);
      
      // Get the dimension of the outer and inner loops
      odim = loc_dim != X ? X : Y;
      idim = loc_dim != Z ? Z : Y;

      // Outer loop
      for (oi=0; oi<geom->n[odim]; oi++)
      {
	  
        // Outer offset
        offset_o = domain_offsets+domain_outer_offsets*oi;

        // Inner loop
        for (ii=0; ii<geom->n[idim]; ii++)
        {
            
          // Assign ghost alpha from the nearest inner alpha value
          ghost_values_send[dim2][geom->n[idim]*oi+ii] = \
            geom->domains[offset_o+domain_inner_offsets*ii];
        }
      }
    }
    else
      size_ghost_values = 0;

    // Communicate
    MPI_Isend(ghost_values_send[dim2],				\
	      size_ghost_values, MPIDOMAIN_ID, geom->mpi_info.neighbors[dim2], 2000, 
	      geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
      
    MPI_Irecv(geom->ghost_domains[dim2],			\
	      size_ghost_values, MPIDOMAIN_ID, geom->mpi_info.neighbors[dim2], 2000, 
	      geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);

  }

  // Wait for all processes to end communication
  MPI_Waitall(NDIMS*2, geom->mpi_info.send_req, MPI_STATUS_IGNORE);
  MPI_Waitall(NDIMS*2, geom->mpi_info.receive_req, MPI_STATUS_IGNORE);

  for (dim2=0; dim2<NDIMS*2; dim2++)
    if (ghost_values_send[dim2])
      free(ghost_values_send[dim2]);

}
//-----------------------------------------------------------------------------
void Geometry_read_chunked_domain_data(hid_t group_id, Geometry_t* geom)
{
  unsigned int dim, i;
  hid_t data_id = H5Dopen(group_id, "voxels", H5P_DEFAULT);
  hid_t file_space_id = H5Dget_space(data_id);
  hsize_t maxdims[NDIMS];
  MPI_Comm comm = geom->mpi_info.comm;

  //int rank = H5Sget_simple_extent_ndims(file_space);
  H5Sget_simple_extent_dims(file_space_id, geom->N, maxdims);
  
  // Collect data offset and local data chunk sizes
  size_t* offsets[NDIMS];// = (size_t**)malloc(sizeof(size_t*)*NDIMS);
  size_t* local_sizes[NDIMS];// = (size_t**)malloc(sizeof(size_t*)*NDIMS);
  for (dim=0; dim<NDIMS; dim++)
  {
    
    offsets[dim] = (size_t*)mpi_malloc(comm, sizeof(size_t)*geom->mpi_info.dims[dim]);
    local_sizes[dim] = (size_t*)mpi_malloc(comm, sizeof(size_t)*geom->mpi_info.dims[dim]);

    // Get local size
    size_t local_size = ceil(((REAL)geom->N[dim])/geom->mpi_info.dims[dim]);
    for (i=0; i<geom->mpi_info.dims[dim]; i++)
    {
      offsets[dim][i]= i*geom->N[dim]/geom->mpi_info.dims[dim];
      local_sizes[dim][i] = (i+1)*geom->N[dim]/geom->mpi_info.dims[dim]-offsets[dim][i];
    }

    // Store process dependent offset and local sizes
    geom->offsets[dim] = offsets[dim][geom->mpi_info.coord[dim]];
    geom->n[dim] = local_sizes[dim][geom->mpi_info.coord[dim]];

    //    if (!geom->n[dim])
  }

  // Cleanup
  for (dim=0; dim<NDIMS; dim++)
  {
    free(offsets[dim]);
    free(local_sizes[dim]);
  }
  
  // Allocate local memory
  geom->domains = mpi_malloc(comm, sizeof(domain_id)*geom->n[X]*geom->n[Y]*geom->n[Z]);

  // Select hyperslab to read
  hsize_t count[3] = {1, 1, 1};
  H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, 
		      geom->offsets, NULL, count, geom->n);
  
  // Create memory space
  hid_t mem_space_id  = H5Screate_simple(NDIMS, geom->n, NULL);

  // Read voxels
  H5Dread(data_id, H5T_STD_U8LE, mem_space_id, file_space_id, H5P_DEFAULT, \
  	  geom->domains);

  H5Sclose(mem_space_id);
  H5Sclose(file_space_id);
  H5Dclose(data_id);
}
//-----------------------------------------------------------------------------
void Geometry_read_domain_connections(hid_t group_id, Geometry_t* geom)
{
  hid_t data_id = H5Dopen(group_id, "domain_connections", H5P_DEFAULT);
  hid_t file_space_id = H5Dget_space(data_id);
  hsize_t dims[2], maxdims[2];
  MPI_Comm comm = geom->mpi_info.comm;

  geom->domain_connections = mpi_malloc(comm, sizeof(domain_id)*\
                                        geom->num_domains*geom->num_domains);

  H5Sget_simple_extent_dims(file_space_id, dims, maxdims);
  
  // Create memory space
  hid_t mem_space_id  = H5Screate_simple(2, dims, NULL);

  assert(dims[0]==geom->num_domains);
  assert(dims[1]==geom->num_domains);

  // Read boundary data
  H5Dread(data_id, H5T_STD_U8LE, mem_space_id, file_space_id, H5P_DEFAULT, \
          geom->domain_connections);

  H5Sclose(mem_space_id);
  H5Sclose(file_space_id);
  H5Dclose(data_id);
}
//-----------------------------------------------------------------------------
void Geometry_read_boundaries(hid_t group_id, Geometry_t* geom)
{
  
  unsigned int i, j, k, dim, dim2, voxel, dbi = 0, local_ghost_ind;
  int* local_distribution_of_global_discrete_boundaries_rank_0 = NULL;
  int* local_distribution_of_global_discrete_boundaries = NULL;
  int** local_distribution_of_global_discrete_boundaries_per_direction = NULL;
  hsize_t maxdims[2], dims[2];
  MPIInfo* mpi_info = &geom->mpi_info;
  MPI_Comm comm = geom->mpi_info.comm;

  // Local domain dimensions [x0,x1,y0,y1,z0,z1]
  size_t local_dims[NDIMS*2];
  for (dim=0; dim<NDIMS;dim++)
  {
    local_dims[dim*2] = geom->offsets[dim];
    local_dims[dim*2+1] = geom->offsets[dim]+geom->n[dim];
  }

  // Keep track of local ghost
  unsigned short** local_ghost_boundaries = mpi_malloc(comm, sizeof(unsigned short*)*NDIMS*2);

  // Iterate over boundaries and read values
  for (i=0; i<geom->num_boundaries; i++)
  {
    hid_t data_id = H5Dopen(group_id, geom->boundary_names[i], H5P_DEFAULT);
    hid_t file_space_id = H5Dget_space(data_id);
    // Here we retrieve values of dims and maxdims
    H5Sget_simple_extent_dims(file_space_id, dims, maxdims);
    
    // Store global size of i-th boundary
    geom->boundary_size[i] = dims[0];

    // Allocate memory for boudaries. Each boundary is a set of pairs of
    // voxels where each voxel is NDIMS-dimensional. Therefore we need
    // memory of size 2*NDIMS*size_of_boundary
    unsigned short* boundaries_full = mpi_malloc(comm, sizeof(unsigned short)*\
                                                 geom->boundary_size[i]*NDIMS*2);

    // Create memory space
    hid_t mem_space_id  = H5Screate_simple(2, dims, NULL);
    
    // Read boundary data
    H5Dread(data_id, H5T_STD_U16LE, mem_space_id, file_space_id, H5P_DEFAULT, \
	    boundaries_full);

    // Allocate memory for local ghost_boundaries if neighbor exist in 
    // that direction
    for (dim2=0; dim2<2*NDIMS; dim2++)
    {
      if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL)
        // Allocate more than enough memory to calculat the local ghost boundaries
        local_ghost_boundaries[dim2] = mpi_malloc(comm, sizeof(unsigned short)*\
						  NDIMS*geom->boundary_size[i]);
      else
        local_ghost_boundaries[dim2] = NULL;
    }

    // If discrete, allocate memory to collect order of discrete boundaries.
    // The local boundaries are stored in local_distribution_of_global_discrete_boundaries_per_direction[6]
    // and ghost boundaries are stored in local_distribution_of_global_discrete_boundaries_per_direction[0-5]
    // depending on the direction in which the boundary is ghosted.
    // At the end the array will be flattened with double information if needed.
    if (geom->boundary_types[i] == discrete)
    {
      // Allocate memory for distribution of ghost discrete boundaries in every
      // direction (0-2*NDIMS) plus one for local boundaries
      local_distribution_of_global_discrete_boundaries_per_direction = mpi_malloc(comm ,sizeof(int*)*(2*NDIMS+1));
      for (dim2=0; dim2<2*NDIMS+1; dim2++)
        local_distribution_of_global_discrete_boundaries_per_direction[dim2] =        \
          mpi_malloc(comm, sizeof(int)*geom->boundary_size[i]);
    }
    
    // Check if process owns boundary
    int local_num = 0;
    // Iterate over the whole boundary
    for (j=0; j<geom->boundary_size[i]; j++)
    {
      unsigned char is_local[2] = {1,1};
      const unsigned int j_ind = j*NDIMS*2;

      // Every certain point at boundary consists of two voxels. A boundary
      // is called to be an outer boundary if these two voxels are exactly
      // the same. Otherwise the boundary is an inner boundary.
      int equal_voxels = 1;
      for (dim=0; dim<NDIMS; dim++)
        equal_voxels *= boundaries_full[j_ind+dim] == boundaries_full[j_ind+dim+NDIMS];
      boundary_pos pos = (equal_voxels == 1 ? outer : inner);
      
      // The first time memorise the boundary position
      if (j==0) 
        geom->boundary_positions[i] = pos;
      // and later check consistency of remaining voxels
      else
        assert(geom->boundary_positions[i] == pos);
      
      // Again since one certain point at boundary consists of 2 voxels,
      // we need to check both of them are in this process
      //
      // Example from .h5 file
      // (j,0): 40, 43, 42, 40, 43, 41
      //
      // First three numbers denote a position of the first voxel, and the
      // last three a position of the second voxel.
      for (voxel=0; voxel<2; voxel++)
      {
        for (dim=0; dim<NDIMS; dim++)
        {
          const unsigned int local_voxel = boundaries_full[j_ind+dim+NDIMS*voxel];
          is_local[voxel] *= (unsigned char)(local_dims[dim*2] <= local_voxel && \
                     local_voxel < local_dims[dim*2+1]);
        }
      }

      // Check if both voxels are local, ie. in this process
      if (is_local[0] && is_local[1])
      {
        // Move boundary information forward in memory and offset it to
        // local voxel numbers
        const unsigned int new_j_ind = local_num*dims[1];
        for (dim2=0; dim2<NDIMS*2; dim2++)
          boundaries_full[new_j_ind+dim2] = boundaries_full[j_ind+dim2]-\
              geom->offsets[dim2%NDIMS];
        
        // If discrete we store the global boundary index. 
        // Notice that this boundary is indeed local.
        if (geom->boundary_types[i] == discrete)
        {
          local_distribution_of_global_discrete_boundaries_per_direction[2*NDIMS][local_num] = j;
          //mpi_printf0(comm, "j: %d\n", local_distribution_of_global_discrete_boundaries_per_direction[2*NDIMS][local_num]);
        }

        // Increase counter of local boundary
        local_num += 1;
      }

      // If only one voxel is local
      else if(is_local[0] || is_local[1])
      {

        // Detect which voxel is not local, ie. not in this process?
        voxel = is_local[0] ? 1 : 0;

        // Iterate over all directions and find out which ones are outside
        // boundary
        for (dim=0; dim<NDIMS; dim++)
        {
          // Get the ghost voxel position in the dim direction
          const unsigned int local_voxel = boundaries_full[j_ind+dim+NDIMS*voxel];
              
          // Check if the voxel's position in the dim direction is outside
          // the boundary
          if (local_voxel < local_dims[dim*2] || local_dims[dim*2+1]<=local_voxel)
          {
            // local_dir represents ghost boundary type:
            // inner-to-ghost or ghost-to-inner
            unsigned char local_dir = 0;
            unsigned char left_right = 0;

            // left/down/front boundary
            // If voxel 0 outside 1 direction and if voxel 1 outside -1 direction
            if (local_voxel < local_dims[dim*2])
            {
              left_right = 0;
              local_dir = voxel == 0 ? 1 : 0;
            }
            // right/up/back boundary
            // If voxel 1 outside 1 direction and if voxel 0 outside -1 direction
            else
            {
              left_right = 1;
              local_dir = voxel == 1 ? 0 : 1;
            }

            // Add voxel information for ghost boundary for global dir
            dim2 = dim*2+left_right;

            // Check that the direction has a neighbor
            assert(geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL);

            unsigned int offset = geom->num_local_ghost_boundaries[i][dim2]*NDIMS;

            //printf("offset: %d[%d]\n", offset, NDIMS*geom->boundary_size[i]);
            
            // If discrete we store the global boundary index. 
            // Notice that this is a ghost boundary so we need to store the index
            // in the appropriate place
            if (geom->boundary_types[i] == discrete)
            {
              local_ghost_ind = geom->num_local_ghost_boundaries[i][dim2];
              local_distribution_of_global_discrete_boundaries_per_direction[dim2][local_ghost_ind] = j;
            }

            // Increase counter
            geom->num_local_ghost_boundaries[i][dim2] += 1;

            // Store local dir
            local_ghost_boundaries[dim2][offset] = local_dir;
	    
            // Iterate over all dimensions and skip the present for which we 
            // are outside the boundary. Also offset voxel to local coordinates
            unsigned char c = 1;
            for (k=0; k<NDIMS; k++)
            {
              if (k!=dim)
              {
                local_ghost_boundaries[dim2][offset+c] = \
                    boundaries_full[j_ind+k] - geom->offsets[k];
                c++;
              }
            }
            /*
            printf("Dim2:%d, N(%d), %d-%d,%d\n", dim2, 
             geom->num_local_ghost_boundaries[i][dim2], 
             local_ghost_boundaries[dim2][offset], 
             local_ghost_boundaries[dim2][offset+1], 
             local_ghost_boundaries[dim2][offset+2]);
            */
            // Do not continue
            break;
          }
        }
      }
    }
    
    // Assign memory for local boundaries and copy memory
    geom->boundaries[i] = mpi_malloc(comm, sizeof(unsigned short)*local_num*2*NDIMS);
    memcpy(geom->boundaries[i], boundaries_full,	\
        sizeof(unsigned short)*local_num*2*NDIMS);

    geom->from_bi_to_dbi_mapping[i] = -1;
    
    // Discrete boundaries
    if (geom->boundary_types[i] == discrete)
    {
      // If discrete register the boundary index
      geom->discrete_boundary_indices[dbi] = i;
      geom->from_bi_to_dbi_mapping[i] = dbi;

      // Calculate the total number of local ghost boundaries
      unsigned int local_ghost_num=0;
      for (dim2=0; dim2<NDIMS*2; dim2++)
        local_ghost_num += geom->num_local_ghost_boundaries[i][dim2];
        
      // Total number of discrete boundaries in the process = number of
      // local discrete boundaries + number of ghost discrete boundaries
      unsigned int total_num = local_num + local_ghost_num;

      // Linearise the array with local distribution of discrete boundaries
      // in a way that first local boundaries indices are memorized and then
      // ghost boundary indices.
      local_distribution_of_global_discrete_boundaries = mpi_malloc(comm ,sizeof(int)*total_num);
      memcpy(local_distribution_of_global_discrete_boundaries,
          local_distribution_of_global_discrete_boundaries_per_direction[2*NDIMS], sizeof(int)*local_num);
      

      unsigned int local_dist_offset = local_num;
      int* _temp_pointer;

      for (dim2=0; dim2<2*NDIMS; dim2++)
      {
        _temp_pointer = local_distribution_of_global_discrete_boundaries+local_dist_offset;
        memcpy(_temp_pointer,
            local_distribution_of_global_discrete_boundaries_per_direction[dim2],
            sizeof(int)*geom->num_local_ghost_boundaries[i][dim2]);
        local_dist_offset += geom->num_local_ghost_boundaries[i][dim2];
        free(local_distribution_of_global_discrete_boundaries_per_direction[dim2]);
      }
      free(local_distribution_of_global_discrete_boundaries_per_direction);
      
      // Memorize the number of local discrete boundaries
      geom->num_local_discrete_boundaries[dbi] = total_num;

      // Allocate memory for determining the openess of discrete boundaries
      geom->open_local_discrete_boundaries[dbi] = \
        mpi_malloc(comm, sizeof(int)*total_num);

      // Initialize all discrete boundaries to close
      memset(geom->open_local_discrete_boundaries[dbi], 0, sizeof(int)*total_num);

      // Allocate memory for species at discrete boundaries
      geom->species_values_at_local_discrete_boundaries[dbi] =  \
        mpi_malloc(comm, sizeof(REAL)*total_num*2);

      // Gather the maximal number of discrete boundaries for all processes
      int max_discrete = mpi_max_int(comm, total_num);


      // Store the sum of all discrete boundaries from all processes.
      // In this place the ghost discrete boundaries are counted twice.
      // It is not a mistake, it is significant.
      geom->num_discrete_boundaries[dbi] = mpi_sum_int(comm, total_num);


      // If rank 0 we prepare memory to hold the number of local discrete boundaries 
      // and eventually collect this information
      if (mpi_info->rank == 0)
      {
        // rank 0 storage for number of discrete boundaries 
        // Used to send information about open channels to all processes
        geom->num_local_discrete_boundaries_rank_0[dbi] =  \
          mpi_malloc(comm, sizeof(int)*mpi_info->size);
        geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(int)*mpi_info->size);

        // rank 0 storage for open values 
        // Used to send information about open channels to all processes
        geom->offset_num_local_discrete_boundaries_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(int)*mpi_info->size);

        // Used to hold send information about open channels to all processes
        geom->open_local_discrete_boundaries_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(int)*max_discrete*mpi_info->size);
        geom->open_local_discrete_boundaries_correct_order_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(int)*max_discrete*mpi_info->size);

        // rank 0 storage for species values
        geom->offset_num_species_values_discrete_boundaries_rank_0[dbi] =  \
          mpi_malloc(comm, sizeof(int)*mpi_info->size);
        geom->species_values_at_local_discrete_boundaries_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(REAL)*2*max_discrete*mpi_info->size);
        geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[dbi] = \
          mpi_malloc(comm, sizeof(REAL)*2*max_discrete*mpi_info->size);

        // Buffer to hold memory for gathering information about local 
        // distribution of discrete boundaries
        // PL Wygląda, że tego pierwszego nie potrzebujemy, jest nie wykorzystany
        // PL a potem jest zwalniana pamięć
        local_distribution_of_global_discrete_boundaries_rank_0 =       \
          mpi_malloc(comm, sizeof(int)*max_discrete*mpi_info->size);
        geom->local_distribution_of_global_discrete_boundaries_rank_0[dbi] = 
          mpi_malloc(comm, sizeof(int)*geom->num_discrete_boundaries[dbi]);
      }
      
      // Communicate the number of local boundaries to first boundary
      int* send_buff = &(geom->num_local_discrete_boundaries[dbi]);
      int* receive_buff = NULL;
      int* displs;

      if (mpi_info->rank==0)
        receive_buff = geom->num_local_discrete_boundaries_rank_0[dbi];
        
      // Call gather and get the number of all discrete boundaries to rank 0 process
      MPI_Gather(send_buff, 1, MPI_INT, receive_buff, 1, MPI_INT, 0, comm);
      
      // Translate the information about number of local discrete boundaries into 
      // an offset array used for gatherv calls
      if (mpi_info->rank == 0)
      {
        
        int rank=0;
        geom->offset_num_local_discrete_boundaries_rank_0[dbi][rank] = 0;
        geom->offset_num_species_values_discrete_boundaries_rank_0[dbi][rank] = 0;
        geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi][rank] = \
          geom->num_local_discrete_boundaries_rank_0[dbi][rank]*2;

        /*printf("rank 0: onldb: %d, osvdb0: %d, nlspvdb: %d\n",
               geom->offset_num_local_discrete_boundaries_rank_0[dbi][rank],
               geom->offset_num_species_values_discrete_boundaries_rank_0[dbi][rank],
               geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi][rank]
               );*/
        for (rank=1; rank<mpi_info->size; rank++)
        {
          geom->offset_num_local_discrete_boundaries_rank_0[dbi][rank] = \
            geom->offset_num_local_discrete_boundaries_rank_0[dbi][rank-1] + \
            geom->num_local_discrete_boundaries_rank_0[dbi][rank-1];

          // Species information needs 2 spaces for each local boundary
          geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi][rank] = \
            geom->num_local_discrete_boundaries_rank_0[dbi][rank]*2;

          geom->offset_num_species_values_discrete_boundaries_rank_0[dbi][rank] = \
            geom->offset_num_species_values_discrete_boundaries_rank_0[dbi][rank-1] + \
            geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi][rank-1];

          /*printf("rank 0: onldb: %d, osvdb0: %d, nlspvdb: %d\n",
                 geom->offset_num_local_discrete_boundaries_rank_0[dbi][rank],
                 geom->offset_num_species_values_discrete_boundaries_rank_0[dbi][rank],
                 geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi][rank]
                 );*/
        }
      }

      // Communicate the local distribution of global discrete boundaries
      send_buff = local_distribution_of_global_discrete_boundaries;
      receive_buff = geom->local_distribution_of_global_discrete_boundaries_rank_0[dbi];
      displs = geom->offset_num_local_discrete_boundaries_rank_0[dbi];

      // Do the assymetric MPI call
      MPI_Gatherv(send_buff, geom->num_local_discrete_boundaries[dbi], MPI_INT,
                  receive_buff, geom->num_local_discrete_boundaries_rank_0[dbi], 
                  displs, MPI_INT, 0, comm);

      // Increase counter for discrete boundaries
      dbi += 1;
    
      // Free local memory
      free(local_distribution_of_global_discrete_boundaries);
      free(local_distribution_of_global_discrete_boundaries_rank_0);

    }

    // Assign memory for local ghost boundaries and copy memory
    // Allocate memory for local ghost_boundaries if neighbot exist in 
    // that direction
    for (dim2=0; dim2<2*NDIMS; dim2++)
    {
      if (geom->num_local_ghost_boundaries[i][dim2])
      {
    	size_t memsize = sizeof(unsigned short)*NDIMS*geom->	\
    	  num_local_ghost_boundaries[i][dim2];
    	geom->local_ghost_boundaries[i][dim2] = mpi_malloc(comm, memsize);
    	memcpy(geom->local_ghost_boundaries[i][dim2], local_ghost_boundaries[dim2], 
    	       memsize);
      }
      
      if (local_ghost_boundaries[dim2])
        free(local_ghost_boundaries[dim2]);
    }

    // Store the number of local boundaries
    geom->local_boundary_size[i] = local_num;
    
    // Clean up
    free(boundaries_full);
    
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(data_id);
    
  }
  
  // Free scratch space
  free(local_ghost_boundaries);

}
//-----------------------------------------------------------------------------
void Geometry_destruct(Geometry_t* geom)
{
  domain_id d_id, b_id;

  // domain_ids
  if (geom->domain_ids)
  {
    free(geom->domain_ids);
    geom->domain_ids = 0;
  }
  
  // domain_names
  if (geom->domain_names)
  {
    for (d_id=0; d_id<geom->num_domains; d_id++)
      if (geom->domain_names[d_id])
        free(geom->domain_names[d_id]);
    
    free(geom->domain_names);
    geom->domain_names = 0;
  }
  
  // Domains
  if (geom->domains)
  {
    free(geom->domains);
    geom->domains = 0;
  }

  if (geom->volumes)
  {
    free(geom->volumes);
    geom->volumes = 0;
  }

  // boundary names
  if (geom->boundary_names)
  {
    for (b_id=0; b_id<geom->num_boundaries; b_id++)
      if (geom->boundary_names[b_id])
        free(geom->boundary_names[b_id]);
    
    free(geom->boundary_names);
    geom->boundary_names = 0;
  }

  // boundary_ids
  if (geom->boundary_ids)
  {
    free(geom->boundary_ids);
    geom->boundary_ids=0;
  }

  // boundary size
  if (geom->boundary_size)
  {
    free(geom->boundary_size);
    geom->boundary_size = 0;
  }

  // boundary_types
  if (geom->boundary_types)
  {
    free(geom->boundary_types);
    geom->boundary_types = 0;
  }
  
  // boundary positions
  if (geom->boundary_positions)
  {
    free(geom->boundary_positions);
    geom->boundary_positions = 0;
  }

  if (geom->local_boundary_size)
  {
    free(geom->local_boundary_size);
    geom->local_boundary_size = 0;
  }

  if (geom->boundaries)
  {
    
    for (b_id=0; b_id<geom->num_boundaries; b_id++)
      if (geom->boundaries[b_id])
        free(geom->boundaries[b_id]);
    
    free(geom->boundaries);
    geom->boundaries = 0;
  }

  if (geom->local_ghost_boundaries)
  {
    unsigned char dir;
    for (b_id=0; b_id<geom->num_boundaries; b_id++)
    {
      for (dir=0; dir<NDIMS*2; dir++)
        if (geom->num_local_ghost_boundaries[b_id][dir]>0)
          free(geom->local_ghost_boundaries[b_id][dir]);

      free(geom->local_ghost_boundaries[b_id]);
    }
    free(geom->local_ghost_boundaries);
    geom->local_ghost_boundaries = 0;
  }

  if(geom->num_local_ghost_boundaries)
  {
    for (b_id=0; b_id<geom->num_boundaries; b_id++)
      if (geom->num_local_ghost_boundaries[b_id])
        free(geom->num_local_ghost_boundaries[b_id]);
    
    free(geom->num_local_ghost_boundaries);
    geom->num_local_ghost_boundaries = 0;
  }

  if (geom->volumes)
  {
    free(geom->volumes);
    geom->volumes = 0;
  }

  if (geom->local_volumes)
  {
    free(geom->local_volumes);
    geom->local_volumes = 0;
  }

}
//-----------------------------------------------------------------------------
void Geometry_output_neighbor(Geometry_t* geom)
{
  int* send_buff;
  int* receive_buff;
  unsigned int i;

  MPIInfo* mpi_info = &geom->mpi_info;
  MPI_Comm comm = mpi_info->comm;
  int* neighbors = mpi_info->neighbors;

  mpi_printf0(comm, "Process neigbors (%d -> no neighbor)\n", MPI_PROC_NULL);
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");

  // Send data to process 0 and display it
  send_buff = mpi_malloc(comm, sizeof(int)*9);
  send_buff[0] = mpi_info->coord[0];
  send_buff[1] = mpi_info->coord[1];
  send_buff[2] = mpi_info->coord[2];
  send_buff[3] = neighbors[XN];
  send_buff[4] = neighbors[XP];
  send_buff[5] = neighbors[YN];
  send_buff[6] = neighbors[YP];
  send_buff[7] = neighbors[ZN];
  send_buff[8] = neighbors[ZP];

  // rank dependent Gather call
  if (mpi_info->rank==0)
  {
    receive_buff = mpi_malloc(comm, sizeof(int)*mpi_info->size*9);
    MPI_Gather(send_buff, 9, MPI_INT, 
	       receive_buff, 9, MPI_INT, 0, mpi_info->comm);
    for (i=0; i<mpi_info->size; i++)
      printf("  process %2d: coord(%d,%d,%d); X[%d,%d], Y[%d,%d], Z[%d,%d]\n",
	     i, receive_buff[i*9+0], receive_buff[i*9+1], receive_buff[i*9+2], 
	     receive_buff[i*9+3], receive_buff[i*9+4], receive_buff[i*9+5], \
	     receive_buff[i*9+6], receive_buff[i*9+7], receive_buff[i*9+8]);

    fflush(stdout);
    free(receive_buff);
  }
  else
    MPI_Gather(send_buff, 9, MPI_INT, NULL, 9, MPI_INT, 0, mpi_info->comm);
    
  free(send_buff);
  fflush(stdout);
}
//-----------------------------------------------------------------------------
void Geometry_output_domain_and_boundary(Geometry_t* geom)
{
  int* send_buff;
  int* send_buff1;
  int* receive_buff;
  int* receive_buff1;
  unsigned int i, j, di, dim;

  MPIInfo* mpi_info = &geom->mpi_info;
  MPI_Comm comm = mpi_info->comm;

  mpi_printf0(comm, "\nGeometry attributs:\n");
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");
  mpi_printf0(comm, "  Global sizes:          (%.1f, %.1f, %.1f) um\n", 
	      geom->global_size[0]/1000., geom->global_size[1]/1000., 
	      geom->global_size[2]/1000.);
  mpi_printf0(comm, "  Resolution:            %.3f nm\n", geom->h_geom);
  mpi_printf0(comm, "  Simulation resolution: %.3f nm\n", geom->h);
  mpi_printf0(comm, "  Subdivisions 1D (3D):  %d (%d)\n", geom->subdivisions, 
	      geom->subdivisions*geom->subdivisions*geom->subdivisions);
  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "Domains: %d\n", geom->num_domains);
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");
  for (i=0; i<geom->num_domains; i++)
  {
    mpi_printf0(comm, " %2d : \"%s\", V: %.4e nm^3, Voxels: %zu", 
                geom->domain_ids[i], geom->domain_names[i], geom->volumes[i], 
                (size_t)geom->domain_num[i]);
    unsigned int num_connections = 0;
    for (j=0; j<geom->num_domains; j++)
    {
      if (i==j)
        continue;
      num_connections += geom->domain_connections[i*geom->num_domains+j];
    }
    if (num_connections)
    {
      mpi_printf0(comm, ", Connections: ");
      unsigned int connection = 0;
      domain_id conn_printed = 0;
      for (j=0; j<geom->num_domains; j++)
      {
        if (i==j)
          continue;
        if (geom->domain_connections[i*geom->num_domains+j])
        {
          mpi_printf0(comm, "\"%s\"", geom->domain_names[j]);
          connection += 1;
          conn_printed = 1;
        }
        if (connection != num_connections && conn_printed)
        {
          mpi_printf0(comm, ", ");
          conn_printed = 0;
        }
      }
    }
    mpi_printf0(comm, "\n");
  }
  
  fflush(stdout);
  MPI_Barrier(comm);
  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "  Voxel distributions (%zu,%zu,%zu) : %.2e\n", (size_t)geom->N[0], 
	      (size_t)geom->N[1], (size_t)geom->N[2], 
	      (REAL)(geom->N[0]*geom->N[1]*geom->N[2]));

  fflush(stdout);
    
  REAL* buff = NULL;

  // Collect volumes
  if (mpi_info->rank==0)
    buff = mpi_malloc(comm, sizeof(REAL)*mpi_info->size*geom->num_domains);
  
  // Send all local volumes to processor 0
  MPI_Gather(geom->local_volumes, geom->num_domains, MPIREAL, 
	     buff, geom->num_domains, MPIREAL, 0, mpi_info->comm);

  // Send data to process 0 and display it
  send_buff = mpi_malloc(comm, sizeof(unsigned int)*7);
  send_buff[0] = geom->offsets[0];
  send_buff[1] = geom->offsets[1];
  send_buff[2] = geom->offsets[2];
  send_buff[3] = geom->n[0];
  send_buff[4] = geom->n[1];
  send_buff[5] = geom->n[2];
  send_buff[6] = geom->n[0]*geom->n[1]*geom->n[2];

  // rank dependent Gather call
  if (mpi_info->size>1)
  {
    if (mpi_info->rank==0)
    {
      receive_buff = mpi_malloc(comm, sizeof(unsigned int)*mpi_info->size*7);
      MPI_Gather(send_buff, 7, MPI_UNSIGNED, 
          receive_buff, 7, MPI_UNSIGNED, 0, mpi_info->comm);
      
      for (i=0; i<mpi_info->size; i++)
      {
        printf("    process %2d: offsets(%d,%d,%d), sizes(%d,%d,%d) : %.2e\n",
    	     i, receive_buff[i*7+0], receive_buff[i*7+1], receive_buff[i*7+2], 
    	     receive_buff[i*7+3], receive_buff[i*7+4], receive_buff[i*7+5], \
    	     (REAL)receive_buff[i*7+6]);

        for (dim=0; dim<NDIMS; dim++)
        {
          if (receive_buff[i*7+3+dim]<1)
            mpi_printf_error(comm, "*** ERROR: On rank %d the local size in the %s "\
                 "direction is 0. Use a larger domain or a lower number"\
                 " of processes.\n", mpi_info->rank, 
                 (dim==X)?"X":(dim==Y? "Y":"Z"));
        }
      }
      
      printf("\n  Local volumes: \n");
      for (i=0; i<mpi_info->size; i++)
      {
        printf("    process %2d: ", i);
        for (di=0; di<geom->num_domains; di++)
        {
          printf("\"%s\": %.3g nm^3", geom->domain_names[di], buff[i*geom->num_domains+di]);
          if (di!=geom->num_domains-1)
            printf("; ");
          else
            printf("\n");
        }
      }

      fflush(stdout);
      free(receive_buff);
      free(buff);
    }
    else
    {
      MPI_Gather(send_buff, 7, MPI_UNSIGNED, 
      NULL, 7, MPI_UNSIGNED, 0, mpi_info->comm);
    }
  }
  
  free(send_buff);
    
  // Communicate local boundary info
  if (geom->num_boundaries>0)
  {
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "Boundaries: %d \"name\"(total)[interior-ghosted][\"type\"]\n", geom->num_boundaries);
    mpi_printf0(comm, "-----------------------------------------------------------------------------\n");

    // Send data to process 0 and display it
    send_buff = mpi_malloc(comm, sizeof(unsigned int)*geom->num_boundaries);
    send_buff1 = mpi_malloc(comm, sizeof(unsigned int)*geom->num_boundaries);

    // Fill send buffer with local boundaries and num local ghost boundaries
    for (i=0; i<geom->num_boundaries; i++)
    {
      send_buff[i] = geom->local_boundary_size[i];
      unsigned int num_local_ghost_boundaries = 0;
      for (j=0; j<NDIMS*2; j++)
        num_local_ghost_boundaries += geom->num_local_ghost_boundaries[i][j];
      send_buff1[i] = num_local_ghost_boundaries;
    }

    // Rank dependent Gather call
    if (mpi_info->rank==0)
    {
      char output[1024];
      receive_buff = mpi_malloc(comm, sizeof(unsigned int)*mpi_info->size*geom->num_boundaries);
      receive_buff1 = mpi_malloc(comm, sizeof(unsigned int)*mpi_info->size*geom->num_boundaries);
      MPI_Gather(send_buff, geom->num_boundaries, MPI_UNSIGNED, 
          receive_buff, geom->num_boundaries, MPI_UNSIGNED, 0, mpi_info->comm);

      MPI_Gather(send_buff1, geom->num_boundaries, MPI_UNSIGNED, 
          receive_buff1, geom->num_boundaries, MPI_UNSIGNED, 0, mpi_info->comm);

      // Sum all interiour boundaries
      for (j=0; j<geom->num_boundaries; j++)
      {
        unsigned int num_interior_boundaries = 0;
        unsigned int num_local_ghost_boundaries = 0;
        for (i=0; i<mpi_info->size; i++)
        {
          num_interior_boundaries += receive_buff[i*geom->num_boundaries+j];
          num_local_ghost_boundaries += receive_buff1[i*geom->num_boundaries+j];
        }

        // Print boundary information. Local ghost boundaries need to be 
        // divided by 2 as we get informations from both shared processes.
        printf(" %2d : \"%s\"(%zu)[%d-%d][\"%s\"]\n", geom->boundary_ids[j], 
               geom->boundary_names[j], geom->boundary_size[j], 
               num_interior_boundaries, num_local_ghost_boundaries/2, 
               boundary_type_strs[geom->boundary_types[j]]);
      }
    
      printf("\n");
      printf("  Boundary distributions: [interior-ghosted]\n");
      for (i=0; i<mpi_info->size; i++)
      {
        for (j=0; j<geom->num_boundaries; j++)
        {
          if (j==0)
            sprintf(output, "%s[%d-%d]", geom->boundary_names[j],\
              receive_buff[i*geom->num_boundaries+j],
              receive_buff1[i*geom->num_boundaries+j]);
          else
            sprintf(output, "%s: %s[%d-%d]", output, geom->boundary_names[j], \
              receive_buff[i*geom->num_boundaries+j],
              receive_buff1[i*geom->num_boundaries+j]);
        }
        printf("    process %2d: %s\n", i, output);
      }
      printf("\n");
      printf("  Discrete Boundary distributions:\n");
      int dbi;
      printf("                ");
      for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
        printf("%8s", geom->boundary_names[geom->discrete_boundary_indices[dbi]]);
      
      printf("\n");
      for (i=0; i<mpi_info->size; i++)
      {
        printf("    process %2d:", i);
        for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
        {
          printf("%8d", geom->num_local_discrete_boundaries_rank_0[dbi][i]);
        }
        printf("\n");
      }
      printf("\n");

      free(receive_buff);
      free(receive_buff1);
    }
    else
    {
      MPI_Gather(send_buff, geom->num_boundaries, MPI_UNSIGNED, 
          NULL, geom->num_boundaries, MPI_UNSIGNED, 0, mpi_info->comm);
      MPI_Gather(send_buff1, geom->num_boundaries, MPI_UNSIGNED, 
          NULL, geom->num_boundaries, MPI_UNSIGNED, 0, mpi_info->comm);
    }

    MPI_Barrier(comm);
    free(send_buff);
    free(send_buff1);
  }
  
}
//-----------------------------------------------------------------------------
boundary_t Geometry_boundary_type_from_str(MPI_Comm comm, char* boundary_type_str)
{
  if (!strcmp(boundary_type_str, "density"))
    return density;
  else if (!strcmp(boundary_type_str, "discrete"))
    return discrete;
  mpi_printf_error(comm, "*** ERROR: \"%s\" is not a valid boundary type.\n", boundary_type_str);
  return density;
}
//-----------------------------------------------------------------------------
domain_id Geometry_get_domain_id(Geometry_t* geom, char* domain_name)
{
  domain_id di;
  for (di=0; di< geom->num_domains; di++)
  {
    if (strcmp(geom->domain_names[di], domain_name)==0)
      return di;
  }

  mpi_printf_error(geom->mpi_info.comm, "*** ERROR: \"%s\" is not a valid "\
		   "domain name.\n", domain_name);
  return 0;
}
//-----------------------------------------------------------------------------
domain_id Geometry_get_boundary_id(Geometry_t* geom, char* boundary_name)
{
  domain_id bi;
  for (bi=0; bi< geom->num_boundaries; bi++)
  {
    if (strcmp(geom-> boundary_names[bi], boundary_name)==0)
      return bi;
  }

  mpi_printf_error(geom->mpi_info.comm, "*** ERROR: \"%s\" is not a valid "\
		   "boundary name.\n", boundary_name);
  return 0;
}
//-----------------------------------------------------------------------------
domain_id Geometry_get_discrete_boundary_id(const Geometry_t* geom, domain_id boundary_id)
{
  domain_id dbi;
  if (geom->boundary_types[boundary_id] != discrete)
    mpi_printf_error(geom->mpi_info.comm, "*** ERROR: \"%d\" is not a discrete "\
		   "boundary type.\n", boundary_id);
       
  for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
    if (geom->discrete_boundary_indices[dbi] == boundary_id)
      return dbi;
      
  mpi_printf_error(geom->mpi_info.comm, "*** ERROR: \"%d\" is not a valid "\
		   "boundary index.\n", boundary_id);
  return 0;
}  
//-----------------------------------------------------------------------------
domain_id Geometry_get_boundary_dir(Geometry_t* geom, domain_id boundary_id, 
					unsigned short index)
{
  unsigned int dim;
  
  assert(boundary_id<geom->num_boundaries);
  assert(index<geom->local_boundary_size[boundary_id]);
  
  // Get pointer to local boundary
  unsigned short* loc_boundary = &geom->boundaries[boundary_id][index*2*NDIMS];

  // Iterate over the dimensions and find the direction
  for (dim=0; dim<NDIMS; dim++)
  {
    // If not the same we know the direction
    if (loc_boundary[dim]!=loc_boundary[dim+NDIMS])
    {
      // If it is left,below,front direction
      if (loc_boundary[dim]<loc_boundary[dim+NDIMS])
      {
        return dim*2;
      }
      // else it is right,above,back direction
      else
      {
        return dim*2+1;
      }
    }
  }
  
  // For outer boundaries it does not matter what the direction is since
  // both voxels are exactly the same.
  if (geom->boundary_positions[boundary_id] == outer)
    return 0;
  
  mpi_printf_error(geom->mpi_info.comm, "*** ERROR: Could not find direction "\
		   "of %s[%d].\n", geom->boundary_names[boundary_id], index);
  return 0;
}
//-----------------------------------------------------------------------------
