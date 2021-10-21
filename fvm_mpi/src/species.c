#include <hdf5.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "species.h"
#include "boundaryfluxes.h"
#include "utils.h"

#include "kernel.h"

//disable OpenMP warnings for ICC
#pragma warning(disable:3180)

#define max_num_domains 5   // assuming five domains (cyt,cleft,jsr,nsr,tt)

static int64_t broadcast_char(int64_t t) {
  return t = 0xff & t, t |= (t<<8), t |= (t<<16), t |= (t<<32);
}

void Species_init_du(Species_t* species) { 
  for (int si=0; si<species->num_species; si++) {
    REAL *u1 = species->u1[si];
    REAL *du = species->du[si];
    int total_n = species->n[X]*species->n[Y]*species->n[Z];
    #pragma omp parallel for
    for (int i=0; i<total_n; i++) {
      du[i] = u1[i];
    }
  }
}

#if defined (MANUAL2)
void init_stencil_test(Species_t* species, MPI_Comm comm) {
  const size_t nx = species->n[X], ny = species->n[Y], nz = species->n[Z];
  //int nx2 = nx - 2, ny2 = ny - 2, nz2 = (nz+7)&-8;
  //species->voxels_stencil_test = NULL;
  species->voxels_stencil_test = mpi_malloc(comm, sizeof(domain_id)*nx*ny*nz);
  const domain_id* dm_ids = species->voxels_domain_ids;
  init_stencil_test2(dm_ids, species->voxels_stencil_test, nx, ny, nz);
}

#else

void init_stencil_test(Species_t* species, MPI_Comm comm) {
  const size_t nx = species->n[X], ny = species->n[Y], nz = species->n[Z];
  int nx2 = nx - 2, ny2 = ny - 2, nz2 = nz - 2;
  species->voxels_stencil_test = NULL;
  if(nz<=2) return;  
  species->voxels_stencil_test = mpi_malloc(comm, sizeof(domain_id)*nx2*ny2*nz2/8);
  const domain_id* dm_ids = species->voxels_domain_ids;
  const unsigned int x_shift  = ny*nz,    y_shift  = nz,  z_shift = 1;
  const unsigned int x_shift2  = ny2*nz2, y_shift2 = nz2;

  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      for (int zi=1; zi<nz-1; zi+=8) {
  int offset = xi*x_shift + yi*y_shift + zi;
  int offset2 = (xi-1)*x_shift2/8 + (yi-1)*y_shift2/8 + (zi-1)/8;
  int offsetm = offset - z_shift, offsetp = offset + z_shift;  // Z direction
  int offsetu = offset - y_shift, offsetd = offset + y_shift;  // Y direction
  int offsetb = offset - x_shift, offsetf = offset + x_shift;  // X direction
  int64_t x, xm, xp, xu, xd, xf, xb;
  memcpy( &x, &dm_ids[offset ], 8);
  memcpy(&xm, &dm_ids[offsetm], 8);
  memcpy(&xp, &dm_ids[offsetp], 8);
  memcpy(&xu, &dm_ids[offsetu], 8);
  memcpy(&xd, &dm_ids[offsetd], 8);
  memcpy(&xf, &dm_ids[offsetf], 8);
  memcpy(&xb, &dm_ids[offsetb], 8);
  //x  = *(int64_t *)&dm_ids[offset];
  //xm = *(int64_t *)&dm_ids[offsetm];
  //xp = *(int64_t *)&dm_ids[offsetp];
  //xu = *(int64_t *)&dm_ids[offsetu];
  //xd = *(int64_t *)&dm_ids[offsetd];
  //xf = *(int64_t *)&dm_ids[offsetf];
  //xb = *(int64_t *)&dm_ids[offsetb];
  int flag = x == xm && x == xp && x == xu && x == xd && x == xf && x == xb; 
  if(!flag || x != broadcast_char(x)) {
    species->voxels_stencil_test[offset2] = 1;
  }
  else {
    species->voxels_stencil_test[offset2] = 0;
  } 
      }
    }
  }
}

#endif

//-----------------------------------------------------------------------------
Species_t* Species_construct(Geometry_t* geom, arguments_t* arguments)
{
  size_t i, bi, di, si, dsi, dim, dim1, dim2, n[NDIMS];
  MPI_Comm comm = geom->mpi_info.comm;
  Species_t* species = mpi_malloc(comm, sizeof(Species_t));
  species->geom = geom;

  // Start reading model file
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
  
  // Create a new file collectively and release property list identifier.
  hid_t file_id = H5Fopen(arguments->model_file, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  // Read domains and compare with geom
  domain_id num_domains;
  read_h5_attr(comm, file_id, "/", "num_domains", &num_domains);

  if (num_domains!=geom->num_domains)
    mpi_printf_error(comm, "*** ERROR: Expected the same number of domains in geometry "\
         "and model files. %d!=%d\n", geom->num_domains, num_domains);
  
  // Allocate information about species that are domain specific
  species->num_diffusive = mpi_malloc(comm, sizeof(domain_id)*geom->num_domains);
  species->diffusive = mpi_malloc(comm, sizeof(domain_id*)*geom->num_domains);
  species->sigma = mpi_malloc(comm, sizeof(REAL*)*geom->num_domains);
  species->init = mpi_malloc(comm, sizeof(REAL*)*geom->num_domains);
  species->num_buffers = mpi_malloc(comm, sizeof(domain_id)*geom->num_domains);
  species->k_off = mpi_malloc(comm, sizeof(REAL*)*geom->num_domains); 
  species->k_on = mpi_malloc(comm, sizeof(REAL*)*geom->num_domains); 
  species->tot = mpi_malloc(comm, sizeof(REAL*)*geom->num_domains);
  species->bsp0 = mpi_malloc(comm, sizeof(domain_id*)*geom->num_domains); 
  species->bsp1 = mpi_malloc(comm, sizeof(domain_id*)*geom->num_domains);
  species->local_domain_num = mpi_malloc(comm, sizeof(long)*geom->num_domains);
  memfill_long(species->local_domain_num, geom->num_domains, 0);

  // Read number of species
  read_h5_attr(comm, file_id, "/", "num_species", &species->num_species);
  
  species->bsp0T  = mpi_malloc(comm, sizeof(domain_id)*species->num_species*geom->num_domains);
  species->bsp1T  = mpi_malloc(comm, sizeof(domain_id)*species->num_species*geom->num_domains);
  species->k_onT  = mpi_malloc(comm, sizeof(REAL)*species->num_species*geom->num_domains);
  species->k_offT = mpi_malloc(comm, sizeof(REAL)*species->num_species*geom->num_domains);
  species->totT   = mpi_malloc(comm, sizeof(REAL)*species->num_species*geom->num_domains);

  memset(species->bsp0T,   0, sizeof(domain_id)*species->num_species*geom->num_domains);
  memset(species->bsp1T,   0, sizeof(domain_id)*species->num_species*geom->num_domains);
  memset(species->k_onT,  0., sizeof(REAL)*species->num_species*geom->num_domains);
  memset(species->k_offT, 0., sizeof(REAL)*species->num_species*geom->num_domains);
  memset(species->totT,   0., sizeof(REAL)*species->num_species*geom->num_domains);

  // Allocate num fixed domain species
  species->num_fixed_domain_species = mpi_malloc(comm, sizeof(size_t)*species->num_species);
  memfill_size_t(species->num_fixed_domain_species, species->num_species, 0);
  species->fixed_domain_species = mpi_malloc(comm, sizeof(size_t*)*species->num_species);
  for (i=0; i<species->num_species; i++)
    species->fixed_domain_species[i] = NULL;

  // Assign volume of species voxel
  species->dV = geom->h*geom->h*geom->h;

  // Iterate over domains and check the names and allocate memory for 
  // domain specific information
  REAL max_sigma = 0.;
  REAL* max_sigma_species = mpi_malloc(comm, sizeof(REAL)*species->num_species);
  memfill(max_sigma_species, species->num_species, 0.);
  species->num_all_diffusive = 0;
  unsigned char* all_diffusive = mpi_malloc(comm, sizeof(unsigned char)*species->num_species);
  memset(all_diffusive, 0, sizeof(int)*species->num_species);
  //  memset(species->all_buffers, 0, sizeof(domain_id)*species->num_species);
  species->all_buffers_b = mpi_malloc(comm, sizeof(unsigned char)*species->num_species);
  memset(species->all_buffers_b, 0, sizeof(unsigned char)*species->num_species);
  for (di=0; di<num_domains; di++)
  {
    char name[MAX_SPECIES_NAME];
    char group_name[MAX_SPECIES_NAME];
    char domain_name[MAX_SPECIES_NAME];
    sprintf(name,"domain_name_%zu", di);
    read_h5_attr(comm, file_id, "/", name, domain_name);
    
    // Check domain name
    if (strcmp(geom->domain_names[di], domain_name)!=0)
      mpi_printf_error(comm, "*** ERROR: Expected the domain names to be the "\
           "same in geometry and model files. %s!=%s\n", 
           geom->domain_names[di], domain_name);
    
    // Read num diffusive species for this domain
    read_h5_attr(comm, file_id, domain_name, "num_diffusive", &species->num_diffusive[di]);
    species->init[di] = mpi_malloc(comm, sizeof(REAL)*species->num_species);
    read_h5_attr(comm, file_id, domain_name, "init", species->init[di]);
    
    if (species->num_diffusive[di]>0)
    {
    
      species->diffusive[di] = mpi_malloc(comm, sizeof(domain_id)*species->num_diffusive[di]);
      species->sigma[di] = mpi_malloc(comm, sizeof(REAL)*species->num_diffusive[di]);

      read_h5_attr(comm, file_id, domain_name, "diffusive", species->diffusive[di]);
      read_h5_attr(comm, file_id, domain_name, "sigma", species->sigma[di]);
      
      for (si=0; si<species->num_diffusive[di]; si++)
      {
        // Flag the species as a diffusive species
        dsi = species->diffusive[di][si];
        all_diffusive[dsi] = 1;
        max_sigma = fmax(max_sigma, species->sigma[di][si]);
        max_sigma_species[dsi] = fmax(species->sigma[di][si], max_sigma_species[dsi]);
      }
    }
    else
    {
      species->diffusive[di] = NULL;
      species->sigma[di] = NULL;
    }

    // Read num buffers for this domain
    read_h5_attr(comm, file_id, domain_name, "num_buffers", &species->num_buffers[di]);
    
    // Allocate memory 
    species->k_on[di]  = mpi_malloc(comm, sizeof(REAL)*species->num_species);
    memfill(species->k_on[di], species->num_species, 0.);

    species->k_off[di] = mpi_malloc(comm, sizeof(REAL)*species->num_species);
    memfill(species->k_off[di], species->num_species, 0.);

    species->tot[di] = mpi_malloc(comm, sizeof(REAL)*species->num_species);
    memfill(species->tot[di], species->num_species, 0.);

    species->bsp0[di] = mpi_malloc(comm, sizeof(domain_id)*species->num_species);
    memset(species->bsp0[di], 0, sizeof(domain_id)*species->num_species);

    species->bsp1[di] = mpi_malloc(comm, sizeof(domain_id)*species->num_species);
    memset(species->bsp1[di], 0, sizeof(domain_id)*species->num_species);


    // Get all domain specific buffers
    for (bi=0; bi<species->num_buffers[di]; bi++)
    {

      // Local buffer species array
      domain_id bsp[2];

      // Get groupname of buffer
      sprintf(group_name, "/%s/buffer_%zu/", domain_name, bi);

      // Read in buffer attr
      read_h5_attr(comm, file_id, group_name, "k_off", &species->k_off[di][bi]);
      read_h5_attr(comm, file_id, group_name, "k_on", &species->k_on[di][bi]);
      read_h5_attr(comm, file_id, group_name, "tot", &species->tot[di][bi]);
      read_h5_attr(comm, file_id, group_name, "species", bsp);
      species->bsp0[di][bi] = bsp[0];
      species->bsp1[di][bi] = bsp[1];

      species->bsp0T[ bi*geom->num_domains + di] = bsp[0];
      species->bsp1T[ bi*geom->num_domains + di] = bsp[1];
      species->k_onT[ bi*geom->num_domains + di] = species->k_on[di][bi];
      species->k_offT[bi*geom->num_domains + di] = species->k_off[di][bi];
      species->totT[  bi*geom->num_domains + di] = species->tot[di][bi];
      
      // Flag the species as a buffer species
      species->all_buffers_b[bsp[0]] = 1;
    }
  }
  
  // Set default time for when last discrete open channeled closed
  species->last_opened_discrete_boundary = -1.0;

  // Calculate time step
  // FIXME: Move to a simulation struct?
  species->dt = arguments->dt > 0 ? arguments->dt : DT_SCALE*(1./6)*geom->h*geom->h/max_sigma;

  // Calculate all diffusive species
  species->num_all_diffusive = 0;
  for (si=0; si<species->num_species; si++)
    species->num_all_diffusive += all_diffusive[si];
  
  species->is_diffusive = mpi_malloc(comm, sizeof(domain_id)*species->num_species);
  memset(species->is_diffusive, 0, sizeof(domain_id)*species->num_species);
  species->all_diffusive = mpi_malloc(comm, sizeof(domain_id)*species->num_all_diffusive);
  species->species_sigma_dt = mpi_malloc(comm, sizeof(REAL)*species->num_all_diffusive);
  species->species_substeps = mpi_malloc(comm, sizeof(domain_id)*species->num_all_diffusive);
  REAL* max_sigma_per_diffusive_species = \
    mpi_malloc(comm, sizeof(REAL)*species->num_all_diffusive);
  memfill(max_sigma_per_diffusive_species, species->num_all_diffusive, 0.);

  dsi=0;
  for (si=0; si<species->num_species; si++)
  {
    if (all_diffusive[si])
    {
      species->all_diffusive[dsi] = si;
      species->species_sigma_dt[dsi] = fabs(max_sigma-max_sigma_species[si])<1e-12 ? \
        species->dt : DT_SCALE*(1./6)*geom->h*geom->h/max_sigma_species[si];
      species->species_substeps[dsi] = fmax(1.,floor(species->species_sigma_dt[dsi]/species->dt));
      species->species_sigma_dt[dsi] = species->dt*species->species_substeps[dsi];
      species->is_diffusive[si] = 1;
      dsi++;
    }
  }
  free(all_diffusive);
  free(max_sigma_species);

  // Calculate the max buffer value for all domains
  species->num_all_buffers = 0;
  for (si=0; si<species->num_species; si++)
    species->num_all_buffers += species->all_buffers_b[si];
  
  // Allocate information about all buffers
  bi=0;
  species->all_buffers = mpi_malloc(comm, sizeof(domain_id)*species->num_all_buffers);
  for (si=0; si<species->num_species; si++)
    if (species->all_buffers_b[si])
      species->all_buffers[bi++] = si;

  // Number of voxels
  for (dim=0; dim<NDIMS; dim++)
  {
    species->N[dim] = geom->N[dim]*geom->subdivisions;
    species->n[dim] = geom->n[dim]*geom->subdivisions;
  }

  // Allocate num_species dependent memory
  species->u1 = mpi_malloc(comm, sizeof(REAL*)*species->num_species);
  species->du = mpi_malloc(comm, sizeof(REAL*)*species->num_species);
  species->species_names = mpi_malloc(comm, sizeof(char*)*species->num_species);
  species->update_species = mpi_malloc(comm, sizeof(domain_id)*species->num_species);
  memset(species->update_species, 0, sizeof(domain_id)*species->num_species);

  // Calculate size of the species arrays
  size_t u_size = 1;
  for (dim=0; dim<NDIMS; dim++)
    u_size *= species->n[dim];

  for (si=0; si<species->num_species; si++)
  {

    char name[MAX_SPECIES_NAME];
    species->species_names[si] = mpi_malloc(comm, sizeof(char)*MAX_SPECIES_NAME);
    sprintf(name,"species_name_%zu", si);
    read_h5_attr(comm, file_id, "/", name, species->species_names[si]);
    
    // Allocate local memory for all species 
    species->u1[si] = mpi_malloc(comm, sizeof(REAL)*u_size);
    memfill(species->u1[si], u_size, 0.);
    species->du[si] = mpi_malloc(comm, sizeof(REAL)*u_size);
    memfill(species->du[si], u_size, 0.);

  }    

  // Read number of fixed domains
  read_h5_attr(comm, file_id, "/", "num_fixed_domain_species", \
               &species->num_fixed_domains);
  if (species->num_fixed_domains>0)
  {
    species->fixed_domains = mpi_malloc(comm, sizeof(domain_id)*\
                                        species->num_fixed_domains*2);
    read_h5_attr(comm, file_id, "/", "fixed_domain_species", \
                 species->fixed_domains);
    
    // Allocate memory for all species voxels 
    Species_init_fixed_domain_species(species, species->num_fixed_domains, 
                                      species->fixed_domains);
  }
  else
  {
    species->fixed_domains = NULL;
  }

  // Check save species
  species->ind_save_species = NULL;
  unsigned int ax;
  for (ax=0; ax<3; ax++)
    species->sheets_save[ax] = NULL;

  if (arguments->num_save_species>0)
  {
    
    // Allocate memory
    species->num_save_species = arguments->num_save_species;
    species->ind_save_species = mpi_malloc(comm, sizeof(domain_id)*\
             arguments->num_save_species);
    for (i=0; i<arguments->num_save_species; i++)
    {
      unsigned char found = 0;
      for (si=0; si<species->num_species; si++)
      {
        if (strcmp(arguments->species[i], species->species_names[si])==0)
        {
          found = 1;
          break;
        }
      }

      // Save the species index
      if (found)
      {
        species->ind_save_species[i] = si;
      }

      // If not found raise an error
      else
      {
        char all_species[MAX_FILE_NAME];
        all_species[0] = '\0';
        for (si=0; si<species->num_species; si++)
        {
          sprintf(all_species, si==0 ? "%s\"%s\"" : "%s, \"%s\"", all_species, \
              species->species_names[si]);
        }
        mpi_printf_error(comm, "*** ERROR: Expected given save species to be "
       "present in the model loaded from file. \n\"%s\" cannot "
       "be found in [%s].\n", arguments->species[i], all_species);
      }
    }

    // Allocate memory to save species in sheets
    species->sheets_save[X] = mpi_malloc(comm, sizeof(REAL)*species->n[Y]*species->n[Z]);
    species->sheets_save[Y] = mpi_malloc(comm, sizeof(REAL)*species->n[Z]*species->n[X]);
    species->sheets_save[Z] = mpi_malloc(comm, sizeof(REAL)*species->n[X]*species->n[Y]);
  }
  
  // Check all data save species
  species->all_data_ind_species = NULL;

  if (arguments->all_data_num_species>0)
  {
    
    // Allocate memory
    species->all_data_num_species = arguments->all_data_num_species;
    species->all_data_ind_species = mpi_malloc(comm, sizeof(domain_id)*\
             arguments->all_data_num_species);
    for (i=0; i<arguments->all_data_num_species; i++)
    {
      unsigned char found = 0;
      for (si=0; si<species->num_species; si++)
      {
        if (strcmp(arguments->all_data_species[i], species->species_names[si])==0)
        {
          found = 1;
          break;
        }
      }

      // Save the species index
      if (found)
        species->all_data_ind_species[i] = si;

      // If not found raise an error
      else
      {
        char all_species[MAX_FILE_NAME];
        all_species[0] = '\0';
        for (si=0; si<species->num_species; si++)
        {
          sprintf(all_species, si==0 ? "%s\"%s\"" : "%s, \"%s\"", all_species, \
              species->species_names[si]);
        }
        mpi_printf_error(comm, "*** ERROR: Expected given save species to be "
       "present in the model loaded from file. \n\"%s\" cannot "
       "be found in [%s].\n", arguments->species[i], all_species);
      }
    }
  }

#if defined(ALPHA) 
  // Calculate alpha values for each diffusive species for each x,y,z direction
  size_t i_offset_geom[NDIMS], i_offset_geom_p[NDIMS], i_offset[NDIMS];
  size_t xi, yi, zi, di_p;
  for (dim=0; dim<NDIMS; dim++)
  {
    
    // Set number of voxels in each direction. The last sheet of voxels in each 
    // direction will not be used, but we use the same number to easy handling
    for (dim1=0; dim1<NDIMS; dim1++)
      n[dim1] = species->n[dim1];
    n[dim] -= 1;
    
    // Allocate memory for diffusive species
    species->alpha[dim] = mpi_malloc(comm, sizeof(REAL*)*species->num_all_diffusive);

    // Iterate over all diffusive species
    for (dsi=0; dsi<species->num_all_diffusive; dsi++)
    {
      
      // Allocate memory for each diffusive species (reuse u_size. Note that this 
      // will allocate slightly more memory than we need, but it will make access 
      // more standardized with same offsets and so on.)
      species->alpha[dim][dsi] = mpi_malloc(comm, sizeof(REAL)*u_size);
      memfill(species->alpha[dim][dsi], u_size, 0.);

      // Get species index
      si = species->all_diffusive[dsi];
      assert(si<species->num_all_diffusive);

      // Iterate over all voxels and fill with alpha values
      // NOTE: If dim == X then n[X] == species->n[X]-1
      for (xi=0; xi<n[X]; xi++)
      {

        // Offsets into geom array
        i_offset_geom[X] = (xi/geom->subdivisions)*geom->n[Z]*geom->n[Y];

        // If creating values for the X direction we grab the next Y value 
        if (dim==X)
          i_offset_geom_p[X] = ((xi+1)/geom->subdivisions)*geom->n[Z]*geom->n[Y];
        else
          i_offset_geom_p[X] = i_offset_geom[X];
        
        // Offset into alpha array
        i_offset[X] = xi*species->n[Z]*species->n[Y];
        
        // NOTE: If dim == Y then n[Y] == species->n[Y]-1
        for (yi=0; yi<n[Y]; yi++)
        {
        
          // Offsets into geom array
          i_offset_geom[Y] = (yi/geom->subdivisions)*geom->n[Z];
          
          // If creating values for the Y direction we grab the next Y value 
          if (dim==Y)
            i_offset_geom_p[Y] = ((yi+1)/geom->subdivisions)*geom->n[Z];
          else
            i_offset_geom_p[Y] = i_offset_geom[Y];
          
          // Offset into alpha array
          i_offset[Y] = yi*species->n[Z];
          
          // NOTE: If dim == Z then n[Z] == species->n[Z]-1
          for (zi=0; zi<n[Z]; zi++)
          {
            i_offset_geom[Z] = zi/geom->subdivisions;

            // If creating values for the Z direction we grab the next Z value 
            if (dim==Z)
              i_offset_geom_p[Z] = ((zi+1)/geom->subdivisions);
            else
              i_offset_geom_p[Z] = (zi/geom->subdivisions);

            // Get domain index and domain index for the next voxel in the dim direction
            di = geom->domains[i_offset_geom[X]+i_offset_geom[Y]+i_offset_geom[Z]];
            di_p = geom->domains[i_offset_geom_p[X]+i_offset_geom_p[Y]+i_offset_geom_p[Z]];
            
            // If the same domain we set alpha based on diffusion constant
            if (di_p == di && dsi<species->num_diffusive[di])
            {
              // dt*D/h^2*num_substeps
              species->alpha[dim][dsi][i_offset[X]+i_offset[Y]+zi] = \
                  species->dt*species->sigma[di][dsi]/(geom->h*geom->h)*\
                  species->species_substeps[dsi];
            }

            // If the two domains are connected
            else if (geom->domain_connections[di_p*geom->num_domains+di] \
                           && dsi<species->num_diffusive[di])
            {

              // If one of the domains has zero diffusion
              if (dsi>=species->num_diffusive[di] || species->sigma[di][dsi]==0 || \
                  dsi>=species->num_diffusive[di_p] || species->sigma[di_p][dsi]==0)
              {
                species->alpha[dim][dsi][i_offset[X]+i_offset[Y]+zi] = 0.0;
              }
              else
              {
                REAL avg_sigma = (species->sigma[di][dsi]+species->sigma[di_p][dsi])/2;
                species->alpha[dim][dsi][i_offset[X]+i_offset[Y]+zi] =  \
                  species->dt*avg_sigma/(geom->h*geom->h)* species->species_substeps[dsi];
              }
            }

            // If no diffusive species in domain or we shift domain across boundary
            else
            {
              species->alpha[dim][dsi][i_offset[X]+i_offset[Y]+zi] = 0.0;
            }
          }
        }
      }
    }
  }
  #endif


  // New code ---------------------------------------------------------------
  if (geom->subdivisions==1) {  // just copy the respective array pointers
    species->voxels_domain_ids = geom->domains;
    for (dim2=0; dim2<NDIMS*2; dim2++)
      species->ghost_voxels_domain_ids[dim2] = geom->ghost_domains[dim2];
  }
  else {
    size_t xi, yi, zi;
    const size_t nx=species->n[X], ny=species->n[Y], nz=species->n[Z];
    const size_t gnx=geom->n[X], gny=geom->n[Y], gnz=geom->n[Z];
    
    species->voxels_domain_ids = mpi_malloc(comm, sizeof(domain_id)*nx*ny*nz);
    #pragma omp parallel for private(yi,zi) collapse(2)
    for (xi=0; xi<nx; xi++)
      for (yi=0; yi<ny; yi++)
  for (zi=0; zi<nz; zi++)
    species->voxels_domain_ids[xi*ny*nz+yi*nz+zi] = geom->domains[(xi/geom->subdivisions)*gny*gnz+(yi/geom->subdivisions)*gnz+(zi/geom->subdivisions)];

    for (dim2=0; dim2<NDIMS*2; dim2++)
      if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
  
  size_t odim, idim, oi, ii, gidim;
      
  if (dim2<2) {
    odim = ny; idim = nz; gidim = gnz;
  }
  else if (dim2<4) {
    odim = nx; idim = nz; gidim = gnz;
  }
  else {
    odim = nx; idim = ny; gidim = gny;
  }

  species->ghost_voxels_domain_ids[dim2] = mpi_malloc(comm, sizeof(domain_id)*odim*idim);

  for (oi=0; oi<odim; oi++)
    for (ii=0; ii<idim; ii++)
      species->ghost_voxels_domain_ids[dim2][oi*idim+ii]=geom->ghost_domains[dim2][(oi/geom->subdivisions)*gidim+(ii/geom->subdivisions)];
      }
  }
  MPI_Type_vector (species->n[X], species->n[Z], species->n[Y]*species->n[Z],
       MPIREAL, &(species->vector_xzplane_type));
  MPI_Type_commit (&(species->vector_xzplane_type));
  MPI_Type_vector (species->n[X]*species->n[Y], 1, species->n[Z],
       MPIREAL, &(species->vector_xyplane_type));
  MPI_Type_commit (&(species->vector_xyplane_type));

  init_stencil_test(species, comm);
  // New code ends here-------------------------------------------------------

  // Allocate ghost values
  size_t oi, ii, di_g;
  unsigned int domain_offsets, domain_offset_o, odim, idim;
  unsigned int domain_outer_offsets;
  unsigned int domain_inner_offsets;
  for (dim2=0; dim2<NDIMS*2; dim2++)
  {
    // X, Y, Z dimension
    size_t loc_dim = dim2/2;
    unsigned char left_right = dim2 % 2;

    // YZ sheet (every thing is contiguous)
    if (loc_dim == X)
    {

      // Sheet offset
      if (left_right==0)
      {
        domain_offsets = 0;
        species->u_offsets[dim2] = 0;
        species->u_inner_sheet_offsets[dim2] = species->n[Y]*species->n[Z];
      }
      else
      {
        domain_offsets = (geom->n[X]-1)*geom->n[Y]*geom->n[Z];
        species->u_offsets[dim2] = (species->n[X]-1)*species->n[Y]*species->n[Z];
        species->u_inner_sheet_offsets[dim2] = (species->n[X]-2)*species->n[Y]*species->n[Z];
      }

      // Each outer iteration is a stave
      domain_outer_offsets = geom->n[Z];
      species->u_outer_offsets[dim2] = species->n[Z];

      // Contiguous!
      domain_inner_offsets = 1;
      species->u_inner_offsets[dim2] = 1;
    }

    // XZ sheet
    else if (loc_dim == Y)
    {
      // Sheet offset
      if (left_right==0)
      {
        domain_offsets = 0;
        species->u_offsets[dim2] = 0;
        species->u_inner_sheet_offsets[dim2] = species->n[Z];
      }
      else
      {
        domain_offsets = (geom->n[Y]-1)*geom->n[Z];
        species->u_offsets[dim2] = (species->n[Y]-1)*species->n[Z];
        species->u_inner_sheet_offsets[dim2] = (species->n[Y]-2)*species->n[Z];
      }
      
      // Each outer iteration is a sheet
      domain_outer_offsets = geom->n[Y]*geom->n[Z];
      species->u_outer_offsets[dim2] = species->n[Y]*species->n[Z];

      // Contiguous!
      domain_inner_offsets = 1;
      species->u_inner_offsets[dim2] = 1;
    }

    // XY sheet
    else
    {

      // Sheet offset
      if (left_right==0)
      {
        domain_offsets = 0;
        species->u_offsets[dim2] = 0;
        species->u_inner_sheet_offsets[dim2] = 1;
      }
      else
      {
        domain_offsets = geom->n[Z]-1;
        species->u_offsets[dim2] = species->n[Z]-1;
        species->u_inner_sheet_offsets[dim2] = species->n[Z]-2;
      }
      
      // Each outer iteration is a sheet
      domain_outer_offsets = geom->n[Y]*geom->n[Z];
      species->u_outer_offsets[dim2] = species->n[Y]*species->n[Z];

      // Each inner iteration is a stave
      domain_inner_offsets = geom->n[Z];
      species->u_inner_offsets[dim2] = species->n[Z];
    }

    // For allocation of send/receive buffers
    /*
    species->ghost_values_send[dim2] = mpi_malloc(comm, sizeof(REAL*)* \
                                                  species->num_all_diffusive);
    */              
    species->ghost_values_receive[dim2] = mpi_malloc(comm, sizeof(REAL*)* \
                                                     species->num_all_diffusive);
    species->size_ghost_values[dim2] = 0;
    for (dsi=0; dsi<species->num_all_diffusive; dsi++)
    {
      species->ghost_values_receive[dim2][dsi] = NULL;
      /*
      species->ghost_values_send[dim2][dsi] = NULL;
      */
    }

    // If neighbor
    if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL)
    {
      
      // Allocate memory 
      size_t size_ghost_values = 1;
      for (dim=0; dim<NDIMS; dim++)
        if (loc_dim!=dim)
          size_ghost_values*=species->n[dim];

      species->size_ghost_values[dim2] = size_ghost_values;

      for (dsi=0; dsi<species->num_all_diffusive; dsi++)
      {
        /*
  if (dim2<2) { // for "negative" and "positve" x directions
    species->ghost_values_send[dim2][dsi] = species->u1[species->all_diffusive[dsi]]+species->u_offsets[dim2]; // pointing directly into the corresponding u array, no need to allocate a separate array!
  }
  else
    species->ghost_values_send[dim2][dsi] = mpi_malloc(comm, sizeof(REAL)* \
                                                           size_ghost_values);
  */
        species->ghost_values_receive[dim2][dsi] = mpi_malloc(comm, sizeof(REAL)* \
                                                              size_ghost_values);
  /*
        if (dim2>=2)
    memfill(species->ghost_values_send[dim2][dsi], size_ghost_values, -1.0);
  */
        memfill(species->ghost_values_receive[dim2][dsi], size_ghost_values, -1.0);
      }
    }
    else // nothing to be allocated if there's no MPI neighbour
    {
    }
  }

  // Init voxel indices for all boundaries
  Species_init_boundary_voxels(species);

  // FIXME: In future releases we need to make this more flexible...
  // Hard code everything about ryr and serca
  species->boundary_fluxes = BoundaryFluxes_construct(species, file_id, arguments);
  
  // Read in convolution constants
  read_h5_attr(comm, file_id, "/", "convolution_constants", species->conv);
  H5Fclose(file_id);

  // Init stochastic event timestepping
  species->stochastic_substep = fmax(1., floor(arguments->dt_update_stoch/species->dt));

  // Init reaction timestepping
  species->reaction_substep = fmax(1., floor(arguments->dt_update_react/species->dt));

  // If not given set default z coordinate and z indices for saving sheets
  // x and y coordinate will not be set.
        
  if (arguments->num_save_species)
  {
    // If no z point was given we prep it with the center coordinate
    if (arguments->num_ax_points[X] + arguments->num_ax_points[Y] + arguments->num_ax_points[Z] == 0)
    {
      arguments->num_ax_points[Z]++;
      arguments->ax_points[Z] = mpi_malloc(comm, sizeof(REAL));
      arguments->ax_points[Z][0] = (species->N[Z]/2-geom->subdivisions/2)*geom->h; // in nm
    }

    // Allocate memory for x,y,z indices
    unsigned int ax;
    for(ax=0; ax<3; ax++)
    { 
      species->sheets_per_species[ax] = arguments->num_ax_points[ax];
      species->indices[ax] = mpi_malloc(comm, sizeof(int)*arguments->num_ax_points[ax]);
    
      species->global_indices[ax] = mpi_malloc(comm, sizeof(unsigned int)*arguments->num_ax_points[ax]);
      species->coords[ax]  = mpi_malloc(comm, sizeof(REAL)*arguments->num_ax_points[ax]);
    
      for (i=0; i<arguments->num_ax_points[ax]; i++)
      {
        if (arguments->ax_points[ax][i]<=0 || arguments->ax_points[ax][i]>=species->N[ax]*geom->h)
          mpi_printf_error(comm, "*** ERROR: Expected given z coordinate for save "
                      "species to be within: (0,%f), got: %f.\n", 
                      species->N[ax]*geom->h, arguments->ax_points[ax][i]);

          // Get global index
          species->global_indices[ax][i] = floor(arguments->ax_points[ax][i]/geom->h);
        
          // Check if it is contained within this process
          if (geom->offsets[ax]*geom->subdivisions<=species->global_indices[ax][i] && 
              species->global_indices[ax][i]<(geom->offsets[ax]+geom->n[ax])*geom->subdivisions)
          {
            // Local index
            species->indices[ax][i] = species->global_indices[ax][i]-geom->offsets[ax]*geom->subdivisions;
          }
          else
            species->indices[ax][i] = -1;
            
          // Copy value
          species->coords[ax][i] = arguments->ax_points[ax][i];
      }
    }
  }
  
  // Populate information whether Dirichlet conditions should be enforced.
  species->force_dirichlet = arguments->force_dirichlet;

  // Populate information provided in command-line with --linescan option,
  // and compute all necessary data.
  species->linescan_data = NULL;
  if (arguments->linescan)
  {
    species->linescan_data = mpi_malloc(comm, sizeof(LinescanData));
    species->linescan_data->axis = arguments->linescan->axis;
    species->linescan_data->species_name = arguments->linescan->species;
    species->linescan_data->species_id = Species_get_species_id(species, species->linescan_data->species_name);
    
    // Get domains indices if domain names were provided in the command-line
    species->linescan_data->num_domains = arguments->linescan->num_domains;
    species->linescan_data->domains = malloc(sizeof(domain_id)*species->linescan_data->num_domains);
    for (i=0; i<species->linescan_data->num_domains; i++)
      species->linescan_data->domains[i] = Geometry_get_domain_id(species->geom, arguments->linescan->domains[i]);

    // Compute global indices
    int ax, limit_offsets[2];
    for (ax=0; ax<3; ++ax)
    {
      // Calculates limits the provided in command-line positional offset
      // must be within.
      limit_offsets[0] = species->N[ax]/2; 
      limit_offsets[1] = species->N[ax] - limit_offsets[0]-1;
      
      // Check if correct values are set
      if (ax != species->linescan_data->axis && (arguments->linescan->offsets[ax] < -limit_offsets[0] ||
                                                 arguments->linescan->offsets[ax] > limit_offsets[1]))
      {
          mpi_printf_error(comm, "*** ERROR: Expected given relative positions for convolution "
                      "to be within: [%d, %d], got: %d.\n", -limit_offsets[0], limit_offsets[1], arguments->linescan->offsets[ax]);
      }
      
      // Translate relative positions wrt the centre of the cube into coordinates.
      species->linescan_data->pos_offsets[ax] = arguments->linescan->offsets[ax];
      species->linescan_data->offsets[ax] = (limit_offsets[0] + arguments->linescan->offsets[ax])*geom->h;
    }
    // We are free to allocate memory for data
    species->linescan_data->sheet_save = mpi_malloc(comm, sizeof(REAL)*species->N[species->linescan_data->axis]);
    memfill(species->linescan_data->sheet_save, species->N[species->linescan_data->axis], 0.0);
  }

  return species;
}

//-----------------------------------------------------------------------------
void Species_init_values(Species_t* species)
{
  size_t di, si, xi, yi, zi, dim, xi_offset_geom;
  size_t xi_offset, yi_offset_geom, yi_offset, zi_geom;
  Geometry_t* geom = species->geom;

  size_t u_size = 1;
  for (dim=0; dim<NDIMS; dim++)
    u_size *= species->n[dim];

  for (si=0; si<species->num_species; si++)
  {
    // Initialize values
    #pragma omp parallel private(yi, zi, xi_offset, yi_offset, xi_offset_geom, yi_offset_geom, zi_geom, di)
    {
    long *tmp = calloc(geom->num_domains, sizeof(long));
    #pragma omp for collapse(2)
    for (xi=0; xi<species->n[X]; xi++) {
      for (yi=0; yi<species->n[Y]; yi++) {
  xi_offset_geom = (xi/geom->subdivisions)*geom->n[Y]*geom->n[Z];
  xi_offset = xi*species->n[Y]*species->n[Z];
        yi_offset_geom = (yi/geom->subdivisions)*geom->n[Z];
        yi_offset = yi*species->n[Z];
        for (zi=0; zi<species->n[Z]; zi++)
        {
          zi_geom = zi/geom->subdivisions;
          di = geom->domains[xi_offset_geom+yi_offset_geom+zi_geom];
          assert(di<geom->num_domains);
    tmp[di] += 1;
          species->u1[si][xi_offset+yi_offset+zi] = species->init[di][si];
        }
      }
    }
    #pragma omp critical
    for(size_t i=0; i<geom->num_domains; i++) species->local_domain_num[i] += tmp[i];
    free(tmp);
    }
  }

  // Sanity check!
  for (di=0; di<geom->num_domains; di++)
  {
    species->local_domain_num[di]/=species->num_species;
    if(fabs(species->local_domain_num[di]*species->dV-geom->local_domain_num[di]*geom->dV)>1e-16)
      printf("*** ERROR: [%d] The number of species domains is not the same: %f!=%f.\n", 
          geom->mpi_info.rank, species->local_domain_num[di]*species->dV,
          geom->local_domain_num[di]*geom->dV);
  }
}

//-----------------------------------------------------------------------------
void Species_init_boundary_voxels(Species_t* species)
{
  unsigned int bi, dim2, dim, left_right, odim, idim, oi, ii, from_ind, to_ind;
  unsigned int o_ind, i_ind, boundary_offset, boundary_id, subdivisions;
  unsigned short* loc_boundary;
  unsigned int* loc_boundary_voxels;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  
  // Allocate memory
  species->num_boundary_voxels = mpi_malloc(comm, sizeof(unsigned int)*geom->num_boundaries);
  species->boundary_voxels = mpi_malloc(comm, sizeof(unsigned int*)*geom->num_boundaries);

  // Iterate ovet all boundaries
  for (boundary_id=0; boundary_id<geom->num_boundaries; boundary_id++)
  {

    // Initiate voxel memory
    subdivisions = geom->boundary_types[boundary_id] == discrete ? 1 : geom->subdivisions;
    
    species->num_boundary_voxels[boundary_id] = subdivisions*subdivisions*\
          geom->local_boundary_size[boundary_id];
    species->boundary_voxels[boundary_id] = mpi_malloc(comm, sizeof(unsigned int)*\
                                        species->num_boundary_voxels[boundary_id]*2*NDIMS);

    // Iterate over geometry boundaries and initiate flux voxels
    for (bi=0; bi<geom->local_boundary_size[boundary_id]; bi++)
    {
      
      // Get local boundary offset
      loc_boundary = &geom->boundaries[boundary_id][bi*2*NDIMS];
      boundary_offset = bi*2*NDIMS*subdivisions*subdivisions;
    
      // Get direction of boundary
      dim2 = Geometry_get_boundary_dir(geom, boundary_id, bi);
      dim = dim2/2;
      left_right = dim2%2;
      odim = dim!=X ? X : Y;
      idim = dim!=Z ? Z : Y;

      // Get from and to ind in the direction of flux. This index should be the same 
      // for the inner and outer directions
    
      // Logic for from_ind. (Opposite logic for to_ind)
      // 1) offset basic voxel dim with subdivisions. 
      // 2) If voxel is left to right (left_right=1) we need to find the last 
      //    voxel row in this direction, hence subdivision-1. if right to left 
      //    (left_right=0) we add 0.
      //   
      from_ind = loc_boundary[dim]*geom->subdivisions + (left_right==0 ? \
                                                         geom->subdivisions-1 : 0);
      to_ind =   loc_boundary[dim+NDIMS]*geom->subdivisions + (left_right==0 ? \
                                                               0 : geom->subdivisions-1);
      
      /*
      printf("Boundary[%d|%d]: dim:[%d-%d][%d,%d](%d->%d)\n", boundary_id, 
             bi, dim, left_right, odim, idim, from_ind, to_ind);
      */
      // Iterate over the outer dimension
      for (oi=0; oi<subdivisions; oi++)
      {
        
        // Find outer index
        o_ind = geom->subdivisions*loc_boundary[odim]+oi;
        
        // If boundary type is discrete we need to offset the index
        if (geom->boundary_types[boundary_id] == discrete)
          o_ind += geom->subdivisions/2;
    
        // Iterate over the inner dimension
        for (ii=0; ii<subdivisions; ii++)
        {
          
          // Get local flux voxel
          loc_boundary_voxels = &species->boundary_voxels[boundary_id][\
                                        boundary_offset+(oi*subdivisions+ii)*2*NDIMS];

          // Find inner index
          i_ind = geom->subdivisions*loc_boundary[idim]+ii;
          
          // If boundary type is discrete we need to offset the index
          if (geom->boundary_types[boundary_id] == discrete)
            i_ind += geom->subdivisions/2;
    
          // Store the from and to voxels
          loc_boundary_voxels[dim]  = from_ind;
          loc_boundary_voxels[odim] = o_ind;
          loc_boundary_voxels[idim] = i_ind;
    
          loc_boundary_voxels[dim+NDIMS]  = to_ind;
          loc_boundary_voxels[odim+NDIMS] = o_ind;
          loc_boundary_voxels[idim+NDIMS] = i_ind;
          
        }
      }
    }
  }
}

void Species_init_fixed_domain_values(Species_t* species) {
  for (int si=0; si<species->num_species; si++) {
    for (int dsi=0; dsi<species->num_fixed_domain_species[si]; dsi++) {
      int di = species->voxels_domain_ids[species->fixed_domain_species[si][dsi]];
      species->u1[si][species->fixed_domain_species[si][dsi]] = species->init[di][si];
    }
  }
}

void Species_init_border_values(Species_t* species) {
  for (int dim2=0; dim2<NDIMS*2; dim2++) {
    if (species->geom->mpi_info.neighbors[dim2] == MPI_PROC_NULL) {
      // Get the directions of the outer and inner loops
      const int dim = dim2/2;
      const int odim = dim != X ? X : Y;
      const int idim = dim != Z ? Z : Y;

      #pragma omp parallel for
      for (int oi=0; oi < species->n[odim]; ++oi) {
        for (int ii=0; ii < species->n[idim]; ++ii) {
          const int pos = species->u_offsets[dim2] + oi*species->u_outer_offsets[dim2] + ii*species->u_inner_offsets[dim2];
          const int di = species->voxels_domain_ids[pos];
          species->u1[0][pos] = species->init[di][0];
        }
      }
    } 
  }
}

void Species_apply_fixed_values(Species_t* species) {
  push_time(FIXED);

  // Zero out du for Ca species on outer boundaries if enforced by special command 
  // line option.
  if (species->force_dirichlet) {
    Species_init_border_values(species);   
  }
  Species_init_fixed_domain_values(species);   
  pop_time(FIXED);

}

//-----------------------------------------------------------------------------
void Species_apply_du(Species_t* species) {
  push_time(APPLYDU);
  int total_n = species->n[X]*species->n[Y]*species->n[Z];
  #if defined(SWAP)
  /*
  for (int si=0; si<species->num_species; si++) {
    if (!species->update_species[si]) continue;
    REAL *u1 = species->u1[si];
    REAL *du = species->du[si];
    REAL **u1p = &u1;
    REAL **dup = &du;  
    REAL *tmp = *u1p;
    
    *u1p = *dup;
    *dup = tmp;
    species->update_species[si] = 0;    
  }
  
  REAL **tmp = species->u1;
  species->u1 = species->du;
  species->du = tmp;    
  */
  for (int si=0; si<species->num_species; si++) {
    species->update_species[si] = 0;    
  }
  REAL ***u1p = species->u1;
  REAL ***dup = species->du;
  REAL **t = *u1p;
  *u1p = *dup;
  *dup = t;
  #else
  for (int si=0; si<species->num_species; si++) {
    if (!species->update_species[si]) continue;
    REAL *u1 = species->u1[si];
    REAL *du = species->du[si];
    #pragma omp parallel for
    for (int i=0; i<total_n; i++) u1[i] = du[i];
    species->update_species[si] = 0;
  }
  #endif
  pop_time(APPLYDU);
}


void Species_update_ghost_one_species_start(Species_t* species, unsigned int dsi)
{
  // In case there is only one MPI process in total......
  if (!(species->geom->mpi_info.size>1))
    return;

  // Initiate MPI communication for one species  
  push_time(GHOST);
  Geometry_t* geom = species->geom;

  // Get the corresponding array pointer (of the target diffusive species)
  REAL* u = species->u1[species->all_diffusive[dsi]];

  const int nx = species->n[X], ny = species->n[Y], nz = species->n[Z];
  int dim2;

  dim2 = 1;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend((u+(nx-1)*ny*nz), ny*nz, MPIREAL,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }
  dim2 = 3;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend((u+(ny-1)*nz), 1, species->vector_xzplane_type,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }
  dim2 = 5;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend((u+nz-1), 1, species->vector_xyplane_type,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }
  dim2 = 4;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend(u, 1, species->vector_xyplane_type,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }
  dim2 = 2;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend(u, 1, species->vector_xzplane_type,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }
  dim2 = 0;
  if (geom->mpi_info.neighbors[dim2]!=MPI_PROC_NULL) {
    MPI_Isend(u, ny*nz, MPIREAL,
        geom->mpi_info.neighbors[dim2], 1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.send_req[dim2]);
    MPI_Irecv(species->ghost_values_receive[dim2][dsi],
        species->size_ghost_values[dim2], 
        MPIREAL, geom->mpi_info.neighbors[dim2], 
        1000*(dim2/2)*dsi,
        geom->mpi_info.comm, &geom->mpi_info.receive_req[dim2]);
  }

  pop_time(GHOST);
}

//-----------------------------------------------------------------------------
void Species_update_ghost_one_species_finish(Species_t* species, unsigned int dsi)
{
  // In case there is only one MPI process in total......
  if (!(species->geom->mpi_info.size>1))
    return;

  // "dsi" is strictly speaking not needed....
  
  // Wait for all send and receive calls (per diffusive species) to end communication
  push_time(GHOST);  
  MPI_Waitall(NDIMS*2, species->geom->mpi_info.send_req, MPI_STATUS_IGNORE);
  MPI_Waitall(NDIMS*2, species->geom->mpi_info.receive_req, MPI_STATUS_IGNORE);
  pop_time(GHOST);
}

//-----------------------------------------------------------------------------
static void sideZ_join(Species_t* species, unsigned int dsi, REAL* coef_table) {
  int offset, offset2;
  REAL u, *coef;

  const REAL* u_array = species->u1[species->all_diffusive[dsi]];
  REAL* du_array = species->du[species->all_diffusive[dsi]];
  const REAL* ghost_value_array4 = species->ghost_values_receive[4][dsi];
  const REAL* ghost_value_array5 = species->ghost_values_receive[5][dsi];
  const domain_id* dm_ids = species->voxels_domain_ids;
  domain_id** ghost_dm_ids = species->ghost_voxels_domain_ids;
  
  const int* neighbors = species->geom->mpi_info.neighbors;
  const unsigned char ghosted4 = neighbors[4]!=MPI_PROC_NULL;
  const unsigned char ghosted5 = neighbors[5]!=MPI_PROC_NULL;
  
  int nx = species->n[X], ny  = species->n[Y], nz = species->n[Z];
  int len1 = nx, incr1 = ny*nz;
  int len2 = ny, incr2 = nz;

  if (ghosted4 && ghosted5) {
    #pragma omp parallel for private(offset, offset2, u, coef)
    for (int idx1=1; idx1<len1-1; idx1++) {
      int idx2;
      idx2 = 1;
      offset  = idx1*incr1 + idx2*incr2;  
      offset2 = idx1*len2  + idx2;
      u = u_array[offset];
      coef = &coef_table[dm_ids[offset]*max_num_domains];
      du_array[offset] += coef[ghost_dm_ids[4][offset2]]*(ghost_value_array4[offset2]);

      innerZ_join_SIMD(u_array, du_array, coef_table, len2, idx1, incr1, incr2, ghost_value_array4, ghost_value_array5, dm_ids, ghost_dm_ids);

      idx2 = len2 - 2;
      offset  = idx1*incr1 + idx2*incr2 + nz - 1;
      offset2 = idx1*len2  + idx2;
      u = u_array[offset];
      coef = &coef_table[dm_ids[offset]*max_num_domains];
      du_array[offset] += coef[ghost_dm_ids[5][offset2]]*(ghost_value_array5[offset2]);
    }
  }
  else if (ghosted4) {
    #pragma omp parallel for
    for (int idx1=1; idx1<len1-1; idx1++) {
      innerZ_SIMD(u_array, du_array, coef_table, len2, idx1, incr1, incr2, idx1*incr1         , ghost_value_array4, dm_ids, ghost_dm_ids[4]);
    }
  }
  else if (ghosted5) {
    #pragma omp parallel for
    for (int idx1=1; idx1<len1-1; idx1++) {
      innerZ_SIMD(u_array, du_array, coef_table, len2, idx1, incr1, incr2, idx1*incr1 + nz - 1, ghost_value_array5, dm_ids, ghost_dm_ids[5]);
    }
  }
}

//-----------------------------------------------------------------------------
void Species_diffusion_one_species_one_side(Species_t* species, unsigned int dsi, unsigned int side_idx, REAL* coef_table)
{
  unsigned int offset, offset_m, offset_p;
  unsigned int idx1, idx2, len1, len2, incr1, incr2, boundary_incr;
  unsigned int fixed_idx, fixed_offset;
  REAL u, *coef;

  const REAL* u_array = species->u1[species->all_diffusive[dsi]];
  REAL* du_array = species->du[species->all_diffusive[dsi]];
  const REAL* ghost_value_array = species->ghost_values_receive[side_idx][dsi];
  
  const domain_id* dm_ids = species->voxels_domain_ids;
  domain_id** ghost_dm_ids = species->ghost_voxels_domain_ids;
  domain_id di, di_n;      // help variables
  
  const int* neighbors = species->geom->mpi_info.neighbors;
  const unsigned char ghosted = neighbors[side_idx]!=MPI_PROC_NULL;
  
  if (side_idx<2) {
    boundary_incr = (side_idx==0) ? -species->n[Y]*species->n[Z] : species->n[Y]*species->n[Z];
    fixed_idx = (side_idx==0) ? 0 : species->n[X]-1;
    fixed_offset = (side_idx==0) ? 0 : (species->n[X]-1)*species->n[Y]*species->n[Z];
    len1 = species->n[Y]; incr1 = species->n[Z];
    len2 = species->n[Z]; incr2 = 1;
  }
  else if (side_idx<4) {
    boundary_incr = (side_idx==2) ? -species->n[Z] : species->n[Z];
    fixed_idx = (side_idx==2) ? 0 : species->n[Y]-1;
    fixed_offset = (side_idx==2) ? 0 : (species->n[Y]-1)*species->n[Z];
    len1 = species->n[X]; incr1 = species->n[Y]*species->n[Z];
    len2 = species->n[Z]; incr2 = 1;
  }
  else {
    boundary_incr = (side_idx==4) ? -1 : 1;
    fixed_idx = (side_idx==4) ? 0 : species->n[Z]-1;
    fixed_offset = (side_idx==4) ? 0 : (species->n[Z]-1);
    len1 = species->n[X]; incr1 = species->n[Y]*species->n[Z];
    len2 = species->n[Y]; incr2 = species->n[Z];
  }

  if (side_idx<2) { // extra work specifically needed for xmin or xmax sides  ====
    // (yi==0,zi==0) corner
    offset = fixed_offset;
    u = u_array[offset];
    du_array[offset] = u;
    coef = coef_table+(dm_ids[offset]*max_num_domains);

    du_array[offset] += neighbors[4]!=MPI_PROC_NULL ? coef[ghost_dm_ids[4][fixed_idx*len1]]*(species->ghost_values_receive[4][dsi][fixed_idx*len1]-u) : 0.;
    offset_p = offset+1; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    
    du_array[offset] += neighbors[2]!=MPI_PROC_NULL ? coef[ghost_dm_ids[2][fixed_idx*len2]]*(species->ghost_values_receive[2][dsi][fixed_idx*len2]-u) : 0.;
    offset_p = offset+len2; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

    du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][0]]*(ghost_value_array[0]-u) : 0.;
    offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

    // (yi==0) edge
    #pragma omp parallel for private(offset, u, coef, offset_m, offset_p, di, di_n)
    for (idx2=1; idx2<len2-1; idx2++) {
      offset = fixed_offset+idx2;
      u = u_array[offset];
      du_array[offset] = u;
      coef = coef_table+(dm_ids[offset]*max_num_domains);

      offset_m = offset-1; di = dm_ids[offset_m];
      offset_p = offset+1; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di]*(u_array[offset_m]-u) + coef[di_n]*(u_array[offset_p]-u);

      du_array[offset] += neighbors[2]!=MPI_PROC_NULL ? coef[ghost_dm_ids[2][fixed_idx*len2+idx2]]*(species->ghost_values_receive[2][dsi][fixed_idx*len2+idx2]-u) : 0.;
      offset_p = offset+len2; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

      du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][idx2]]*(ghost_value_array[idx2]-u) : 0.;
      offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    }

    // (yi==0,zi==nz-1) corner
    offset = fixed_offset+len2-1;
    u = u_array[offset];
    du_array[offset] = u;
    coef = coef_table+(dm_ids[offset]*max_num_domains);

    offset_m = offset-1; di = dm_ids[offset_m];
    du_array[offset] += coef[di]*(u_array[offset_m]-u);
    du_array[offset] += neighbors[5]!=MPI_PROC_NULL ? coef[ghost_dm_ids[5][fixed_idx*len1]]*(species->ghost_values_receive[5][dsi][fixed_idx*len1]-u) : 0.;
    
    du_array[offset] += neighbors[2]!=MPI_PROC_NULL ? coef[ghost_dm_ids[2][(fixed_idx+1)*len2-1]]*(species->ghost_values_receive[2][dsi][(fixed_idx+1)*len2-1]-u) : 0.;
    offset_p = offset+len2; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

    du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][len2-1]]*(ghost_value_array[len2-1]-u) : 0.;
    offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
  }

#pragma omp parallel for private(idx2, offset, u, coef, offset_m, offset_p, di, di_n)
  for (idx1=1; idx1<len1-1; idx1++) {
    if (side_idx<4) { // extra work needed for xmin/ymin or xmax/ymax sides  =====
      // (idx2==0) 
      offset = fixed_offset + idx1*incr1;
      u = u_array[offset];
      du_array[offset] = u;
      coef = coef_table+(dm_ids[offset]*max_num_domains);

      if (neighbors[4]!=MPI_PROC_NULL) {
  offset_m = (side_idx<2) ? fixed_idx*species->n[Y]+idx1 : idx1*species->n[Y]+fixed_idx;
  du_array[offset] += coef[ghost_dm_ids[4][offset_m]]*(species->ghost_values_receive[4][dsi][offset_m]-u);
      }
      offset_p = offset+1; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

      offset_m = offset-incr1; di = dm_ids[offset_m];
      offset_p = offset+incr1; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di]*(u_array[offset_m]-u)+coef[di_n]*(u_array[offset_p]-u);

      du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][idx1*len2]]*(ghost_value_array[idx1*len2]-u) : 0.;
      offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    }
    if (ghosted) {
      offset = fixed_offset + idx1*incr1;
      int offset2 = idx1*len2;
      side_ghosted_SIMD(u_array, du_array, coef_table, ghost_value_array, dm_ids, ghost_dm_ids[side_idx], dsi, offset, offset2, incr1, incr2, boundary_incr, ghosted, len2);
    }
    else {
      offset = fixed_offset + idx1*incr1;
      int offset2 = idx1*len2;
      side_SIMD(u_array, du_array, coef_table, ghost_value_array, dm_ids, ghost_dm_ids[side_idx], dsi, offset, offset2, incr1, incr2, boundary_incr, ghosted, len2);
    }

    if (side_idx<4) { // extra work needed for xmin/ymin or xmax/ymax sides  =====
      // (idx2==len2-1)
      offset = fixed_offset + idx1*incr1 + (len2-1)*incr2;
      u = u_array[offset];
      du_array[offset] = u;
      coef = coef_table+(dm_ids[offset]*max_num_domains);

      offset_m = offset-1; di = dm_ids[offset_m];
      du_array[offset] += coef[di]*(u_array[offset_m]-u);
      if (neighbors[5]!=MPI_PROC_NULL) {
  offset_p = (side_idx<2) ? fixed_idx*species->n[Y]+idx1 : idx1*species->n[Y]+fixed_idx;
  du_array[offset] += coef[ghost_dm_ids[5][offset_p]]*(species->ghost_values_receive[5][dsi][offset_p]-u);
      }

      offset_m = offset-incr1; di = dm_ids[offset_m];
      offset_p = offset+incr1; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di]*(u_array[offset_m]-u)+coef[di_n]*(u_array[offset_p]-u);

      du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][(idx1+1)*len2-1]]*(ghost_value_array[(idx1+1)*len2-1]-u) : 0.;
      offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    }
  }

  if (side_idx<2) { // extra work specifically needed for xmin or xmax sides  ====
    // (yi==ny-1,zi==0) corner
    offset = fixed_offset+(len1-1)*len2;
    u = u_array[offset];
    du_array[offset] = u;
    coef = coef_table+(dm_ids[offset]*max_num_domains);

    du_array[offset] += neighbors[4]!=MPI_PROC_NULL ? coef[ghost_dm_ids[4][(fixed_idx+1)*len1-1]]*(species->ghost_values_receive[4][dsi][(fixed_idx+1)*len1-1]-u) : 0.;
    offset_p = offset+1; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    
    offset_m = offset-len2; di = dm_ids[offset_m];
    du_array[offset] += coef[di]*(u_array[offset_m]-u);
    du_array[offset] += neighbors[3]!=MPI_PROC_NULL ? coef[ghost_dm_ids[3][fixed_idx*len2]]*(species->ghost_values_receive[3][dsi][fixed_idx*len2]-u) : 0.;

    du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][(len1-1)*len2]]*(ghost_value_array[(len1-1)*len2]-u) : 0.;
    offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);

    // (yi==ny-1) edge
    #pragma omp parallel for private(offset, u, coef, offset_m, offset_p, di, di_n)
    for (idx2=1; idx2<len2-1; idx2++) {
      offset = fixed_offset+(len1-1)*len2+idx2;
      u = u_array[offset];
      du_array[offset] = u;
      coef = coef_table+(dm_ids[offset]*max_num_domains);

      offset_m = offset-1; di = dm_ids[offset_m];
      offset_p = offset+1; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di]*(u_array[offset_m]-u) + coef[di_n]*(u_array[offset_p]-u);

      offset_m = offset-len2; di = dm_ids[offset_m];
      du_array[offset] += coef[di]*(u_array[offset_m]-u);
      du_array[offset] += neighbors[3]!=MPI_PROC_NULL ? coef[ghost_dm_ids[3][fixed_idx*len2+idx2]]*(species->ghost_values_receive[3][dsi][fixed_idx*len2+idx2]-u) : 0.;

      du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][(len1-1)*len2+idx2]]*(ghost_value_array[(len1-1)*len2+idx2]-u) : 0.;
      offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
      du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
    }

    // (yi==ny-1,zi==nz-1) corner
    offset = fixed_offset+len1*len2-1;
    u = u_array[offset];
    du_array[offset] = u;
    coef = coef_table+(dm_ids[offset]*max_num_domains);

    offset_m = offset-1; di = dm_ids[offset_m];
    du_array[offset] += coef[di]*(u_array[offset_m]-u);
    du_array[offset] += neighbors[5]!=MPI_PROC_NULL ? coef[ghost_dm_ids[5][(fixed_idx+1)*len1-1]]*(species->ghost_values_receive[5][dsi][(fixed_idx+1)*len1-1]-u) : 0.;
    
    offset_m = offset-len2; di = dm_ids[offset_m];
    du_array[offset] += coef[di]*(u_array[offset_m]-u);
    du_array[offset] += neighbors[3]!=MPI_PROC_NULL ? coef[ghost_dm_ids[3][(fixed_idx+1)*len2-1]]*(species->ghost_values_receive[3][dsi][(fixed_idx+1)*len2-1]-u) : 0.;

    du_array[offset] += ghosted ? coef[ghost_dm_ids[side_idx][len1*len2-1]]*(ghost_value_array[len1*len2-1]-u) : 0.;
    offset_p = offset-boundary_incr; di_n = dm_ids[offset_p];
    du_array[offset] += coef[di_n]*(u_array[offset_p]-u);
  }
}

//-----------------------------------------------------------------------------
void Species_step_diffusion(Species_t* species, size_t time_ind)
{
  Geometry_t* geom = species->geom;
  unsigned int si, dsi, xi, yi, zi;
  unsigned int offset, offset_m, offset_p;
  REAL* u1;
  REAL* du;
  REAL* alpha;
  REAL u;

  const size_t nx = species->n[X], ny = species->n[Y], nz = species->n[Z];
  const size_t nx2 = nx - 2, ny2 = ny - 2, nz2 = nz - 2;
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1; // constants
  const unsigned int x_shift2  = ny2*nz2, y_shift2 = nz2;

  REAL coef_table[max_num_domains*max_num_domains], *coef; // maybe a 2D array???
  domain_id di, di_n;      // help variables
  const domain_id* dm_ids = species->voxels_domain_ids;

  const int* neighbors = species->geom->mpi_info.neighbors;
  const unsigned char ghosted4 = neighbors[4]!=MPI_PROC_NULL;
  const unsigned char ghosted5 = neighbors[5]!=MPI_PROC_NULL;
  domain_id** ghost_dm_ids = species->ghost_voxels_domain_ids;
    
  // Iterate over all diffusive species
  for (dsi=0; dsi<species->num_all_diffusive; dsi++) {
    #if defined(ALPHA) 
    REAL *alphax = species->alpha[X][dsi];
    REAL *alphay = species->alpha[Y][dsi];
    REAL *alphaz = species->alpha[Z][dsi];
    #endif
    // If substeping 
    if (time_ind % species->species_substeps[dsi] != 0)
      continue;

    // Initiate exchange of ghost values
    Species_update_ghost_one_species_start (species, dsi);

    push_time(DIFF);

    // Get species index
    si = species->all_diffusive[dsi];
    u1 = species->u1[si];
    du = species->du[si];
    species->update_species[si] = 1;
    
    /* Prepare the 5x5 coefficient table for this diffusive species */
    memfill (coef_table, max_num_domains*max_num_domains, 0.);
    for (di=0; di<geom->num_domains; di++)
      if (dsi<species->num_diffusive[di] && species->sigma[di][dsi]!=0) {
    
  for (di_n=0; di_n<di; di_n++)
    if (geom->domain_connections[di*geom->num_domains+di_n]
        && dsi<species->num_diffusive[di_n] && species->sigma[di_n][dsi]!=0)
      coef_table[di*max_num_domains+di_n] = coef_table[di_n*max_num_domains+di] = species->dt*0.5*(species->sigma[di][dsi]+species->sigma[di_n][dsi])
        /(geom->h*geom->h)*species->species_substeps[dsi];

  coef_table[di*max_num_domains+di] = species->dt*species->sigma[di][dsi]
    /(geom->h*geom->h)*species->species_substeps[dsi];
      }

    #if defined(MANUAL2)
    #if defined(ALPHA) 
    kernel_alpha_SIMD(u1, du, coef_table, dm_ids, ghost_dm_ids, species->voxels_stencil_test, nx, ny, nz, ghosted4, ghosted5, alphax, alphay, alphaz);
    #else
    kernel_SIMD(u1, du, coef_table, dm_ids, ghost_dm_ids, species->voxels_stencil_test, nx, ny, nz, ghosted4, ghosted5);
    #endif
    //kernel_AVX2(u1, du, coef_table, dm_ids, ghost_dm_ids, species->voxels_stencil_test, nx, ny, nz, ghosted4, ghosted5);
    //kernel_SSE4(u1, du, coef_table, dm_ids, ghost_dm_ids, species->voxels_stencil_test, nx, ny, nz, ghosted4, ghosted5);
    #else
    // Iterate over the interior 3D volume of voxels (overlapping with MPI comm)
#pragma omp parallel for private(yi,zi, offset, u, coef, offset_m, offset_p, di, di_n, alpha) collapse(2)
    for (xi=1; xi<nx-1; xi++) {
      for (yi=1; yi<ny-1; yi++) {
        #pragma ivdep
        for (zi=1;zi<nz-1; zi++) {
    offset = xi*x_shift + yi*y_shift + zi;
          u = u1[offset];
          #if !defined(ALPHA)
    coef = coef_table+(dm_ids[offset]*max_num_domains);
    #endif
          // Z direction
    offset_m = offset-1; di = dm_ids[offset_m];
    offset_p = offset+1; di_n = dm_ids[offset_p];
          #if defined(ALPHA)
          alpha = species->alpha[Z][dsi];
          du[offset] += alpha[offset_m]*(u1[offset_m]-u) + alpha[offset]*(u1[offset_p]-u);
    #else
          du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
    #endif
          // Y direction
    offset_m = offset-y_shift; di = dm_ids[offset_m];
    offset_p = offset+y_shift; di_n = dm_ids[offset_p];
          #if defined(ALPHA)
          alpha = species->alpha[Y][dsi];
          du[offset] += alpha[offset_m]*(u1[offset_m]-u) + alpha[offset]*(u1[offset_p]-u);
    #else
          du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
    #endif
          // X direction
    offset_m = offset-x_shift; di = dm_ids[offset_m];
    offset_p = offset+x_shift; di_n = dm_ids[offset_p];
          #if defined(ALPHA)
          alpha = species->alpha[X][dsi];
          du[offset] += alpha[offset_m]*(u1[offset_m]-u) + alpha[offset]*(u1[offset_p]-u);
    #else
          du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
    #endif
        }
      }
    }
    #endif

    pop_time(DIFF);

    // Conclude exchange of ghost values
    Species_update_ghost_one_species_finish (species, dsi);

    push_time(BORDERS);
    push_time(BORDERSX);    
    Species_diffusion_one_species_one_side (species, dsi, 1, coef_table);  // xmax
    pop_time(BORDERSX);

    push_time(BORDERSY);    
    Species_diffusion_one_species_one_side (species, dsi, 3, coef_table);  // ymax
    pop_time(BORDERSY);
#if defined(MANUAL2)
    push_time(BORDERSZ);    
    sideZ_join(species, dsi, coef_table);  // zmax
    pop_time(BORDERSZ);
#else
    push_time(BORDERSZ);
    Species_diffusion_one_species_one_side (species, dsi, 4, coef_table);  // zmin
    pop_time(BORDERSZ);

    push_time(BORDERSZ);
    Species_diffusion_one_species_one_side (species, dsi, 5, coef_table);  // zmin
    pop_time(BORDERSZ);
#endif
    push_time(BORDERSY);    
    Species_diffusion_one_species_one_side (species, dsi, 2, coef_table);  // ymin
    pop_time(BORDERSY);

    push_time(BORDERSX);    
    Species_diffusion_one_species_one_side (species, dsi, 0, coef_table);  // xmin
    pop_time(BORDERSX);
    pop_time(BORDERS);

  }
}

//-----------------------------------------------------------------------------
void Species_step_reaction(Species_t* species, size_t time_ind)
{
  // If substeping
  if (species->num_all_buffers==0 || time_ind % species->reaction_substep != 0)
    return;

  push_time(REACT);

  Geometry_t* geom = species->geom;
  domain_id bsi0, bsi1;
  unsigned int bi, di, xi, yi, zi;
  REAL J, b0, b1;
  // Scale time step with number of substeps
  const REAL dt = species->dt*species->reaction_substep;
  
  // Needed for vectorization
  const size_t nz = species->n[Z];

  const domain_id* dm_ids = species->voxels_domain_ids;
  const unsigned int incr_x = species->n[Y]*species->n[Z];
  const unsigned int incr_y = species->n[Z];

  // Flag all the species that are to be updated during reaction calculation


  int nspecies = species->num_species;
  int ndomains = geom->num_domains;

  // Iterate over the interior X indices
  #pragma omp parallel for private(yi, bi, zi, di, bsi0, bsi1, b0, b1, J) collapse(2)
  for (xi=0; xi<species->n[X]; xi++)
  {
    // Iterate over the interior Y indices
    for (yi=0; yi<species->n[Y]; yi++)
    {
      // Iterate over all buffer species
      for (bi=0; bi<species->num_all_buffers; bi++)
      {       
  unsigned int offset2 = xi*incr_x + yi*incr_y;
  reaction_SIMD(species->u1, species->du, species->k_onT, species->k_offT, species->totT, species->bsp0T, species->bsp1T, bi, offset2, dt, dm_ids, ndomains, nz, species->update_species);
      }
    }
  }
  for (di=0; di<geom->num_domains; di++)
    for (bi=0; bi<species->num_all_buffers; bi++) {
      species->update_species[species->bsp0[di][bi]] = 1;
      species->update_species[species->bsp1[di][bi]] = 1;
    }
  pop_time(REACT);
}

void Species_step_boundary_fluxes(Species_t* species, size_t time_ind)
{
  unsigned int* loc_boundary_voxels;
  unsigned int yz_offset, y_offset, xi, yi, zi, u0_offset=0, u1_offset=0;
  unsigned int ghost_offset=0, outer_i, inner_i, z, ind, g_ind, subdivisions;
  unsigned int open_state_ind;
  Geometry_t* geom = species->geom;
  unsigned int bi, bvi, si=0, dim2, dim, gi, dsi;
  unsigned short ghost_first_coord, ghost_second_coord;
  size_t u_offset, u_outer_offset, u_inner_offset;
  const Flux_t* flux_info;
  
  //unsigned short ghost_type;
  REAL J, flux_sign;
  
  // Species values on both side of the boundaries
  REAL u0;
  REAL u1;

  // Get offsets 
  yz_offset = species->n[Y]*species->n[Z];
  y_offset  = species->n[Z];

  push_time(BOUNDARIES);
  
  // Iterate over all boundaries
  for (bi=0; bi<geom->num_boundaries; bi++)
  {

    // Get flux function and params. If there is not flux connected to the
    // boundary, then go to the next one
    flux_info = BoundaryFluxes_get_flux_info(species->boundary_fluxes, bi);
    if (!flux_info)
      continue;
      
    // Get diffusive species index
    dsi = flux_info->flux_dsi;
    
    // If sub-stepping 
    if (time_ind % species->species_substeps[dsi] != 0)
      continue;
      
    // Get corresponding species index
    si = species->all_diffusive[dsi];
    species->update_species[si] = 1;
    

    // Iterate over all boundary voxels

    #pragma omp parallel for private(xi, yi, zi, u0_offset, u0, u1_offset, u1, J, loc_boundary_voxels)
    for (bvi=0; bvi<species->num_boundary_voxels[bi]; bvi++)
    {
      // If discrete and not open continue
      if (geom->boundary_types[bi]==discrete &&                 \
          !geom->open_local_discrete_boundaries[geom->from_bi_to_dbi_mapping[bi]][bvi])
      {
        continue;
      }

      // Get start of indices
      loc_boundary_voxels = &species->boundary_voxels[bi][bvi*2*NDIMS];
      
      // Collect from voxel indices
      xi = loc_boundary_voxels[X];
      yi = loc_boundary_voxels[Y];
      zi = loc_boundary_voxels[Z];
#if DEBUG
      printf("rank[%d] %s FROM: %d:[%d,%d,%d]=%.2f\n", geom->mpi_info.rank, 
             geom->boundary_names[bi], bvi, xi, yi, zi,                   \
             species->u1[si][xi*yz_offset+yi*y_offset+zi]);
#endif
      u0_offset = xi*yz_offset+yi*y_offset+zi;
      u0 = species->u1[si][u0_offset];

      // Increase pointer
      loc_boundary_voxels += NDIMS;
      
      // Collect to voxel indices
      xi = loc_boundary_voxels[X];
      yi = loc_boundary_voxels[Y];
      zi = loc_boundary_voxels[Z];
#if DEBUG
      printf("rank[%d] %s   To: %d:[%d,%d,%d]=%.2f\n", geom->mpi_info.rank, 
             geom->boundary_names[bi], bvi, xi, yi, zi,                   \
             species->u1[si][xi*yz_offset+yi*y_offset+zi]);
#endif

      u1_offset = xi*yz_offset+yi*y_offset+zi;
      u1 = species->u1[si][u1_offset];

      // Call flux function
      J = flux_info->flux_function(species->dt, geom->h, u0, u1, flux_info->flux_params);
      //printf("++++ J= %f\n", J);

      // Apply flux from u0 to u1. For outer boundaries u0_offset and u1_offset
      // are the same.

      #pragma omp atomic
      species->du[si][u1_offset] += J;
      if (geom->boundary_positions[bi] == inner)
  #pragma omp atomic
        species->du[si][u0_offset] -= J;

    }
    // Set subdivision to one if discrete boundary. Otherwise, keep
    // geometry subdivision value.
    subdivisions = (geom->boundary_types[bi]==discrete ? 1 : geom->subdivisions);
    
    // Offset index for open state information
    open_state_ind = geom->local_boundary_size[bi];
    
    // Calculate flux at ghost boundaries.
    for(dim2=0; dim2 < NDIMS*2; dim2++)
    {
      // Extract the correct dimension
      dim = dim2/2;
      if (dim == X)
        ghost_offset = species->n[Z];
      else if (dim == Y)
        ghost_offset = species->n[Z];
      else
        ghost_offset = species->n[Y];
        
      // Get u offsets
      u_offset = species->u_offsets[dim2];
      u_inner_offset = species->u_inner_offsets[dim2];
      u_outer_offset = species->u_outer_offsets[dim2];
      
      // There should be no ghost boundaries for outer boundaries.
      if (geom->boundary_positions[bi] == outer)
        assert(geom->num_local_ghost_boundaries[bi][dim2] == 0);
      
      // Iterate over all ghost boundaries in dim2 direction
      for (gi=0; gi<geom->num_local_ghost_boundaries[bi][dim2]; gi++)
      {
        // For discrete boundaries, check if a state is open.
        // If not, skip the calculations.
        if (geom->boundary_types[bi] == discrete && \
              !geom->open_local_discrete_boundaries[geom->from_bi_to_dbi_mapping[bi]][open_state_ind++])
        {
          continue;
        }

        // Save its 2D coordinates - ghost_first_coord is an outer coordinate for species coordinates.
        // Conversely, ghost_second_coord is an inner coordinate.
        ghost_first_coord = geom->local_ghost_boundaries[bi][dim2][3*gi+1]*geom->subdivisions;
        ghost_second_coord = geom->local_ghost_boundaries[bi][dim2][3*gi+2]*geom->subdivisions;
        
        z = ghost_first_coord*ghost_offset + ghost_second_coord;
        
        // Notice that if goem->subdivision > 1, then every geometry voxel is divided into
        // goem->subdivision**3 smaller voxels in species coordinates. Therefore, for N ghost
        // boundaries in geometry coordinates, we need to take care of goem->subdivision**2*N
        // ghost boundaries in species coordinates
        for(outer_i=0; outer_i < subdivisions; outer_i++)
        {
          for (inner_i=0; inner_i < subdivisions; inner_i++)
          {
            // Shift the indices if a boundary is discrete
            if (geom->boundary_types[bi] == discrete)
            {
              outer_i += geom->subdivisions/2;
              inner_i += geom->subdivisions/2;
            }
            // calculate the index
            ind = z + outer_i*ghost_offset + inner_i;
      g_ind = u_offset + (ghost_first_coord+outer_i)*u_outer_offset
        + (ghost_second_coord+inner_i)*u_inner_offset;
            
            // Extract value of a ghost boundary from this process
            // and from a neighbour process
            //
            // Use ghost_values_send to access to "inner" values
            // Conversely, ghost_values_receive to "outer" values
            if (geom->local_ghost_boundaries[bi][dim2][3*gi]) // ghost-to-inner
            {
              //u1 = species->ghost_values_send[dim2][dsi][ind];
        u1 = species->u1[si][g_ind];
              u0 = species->ghost_values_receive[dim2][dsi][ind];
              flux_sign = -1.0;
            }
            else // inner-to-ghost
            {
              //u0 = species->ghost_values_send[dim2][dsi][ind];
        u0 = species->u1[si][g_ind];
              u1 = species->ghost_values_receive[dim2][dsi][ind];
              flux_sign = 1.0;
            }
        
            // Calculate flux
            J = flux_info->flux_function(species->dt, geom->h, u0, u1, flux_info->flux_params);
            J *= flux_sign;
       
            // Apply flux from u0 to u1 only to the inner voxel
            species->du[si][u_offset + u_outer_offset*(ghost_first_coord+outer_i) \
                            + u_inner_offset*(ghost_second_coord+inner_i)] -= J;
          } // end inner loop
        } // end outer loop
      } // end ghost boundaries loop
    } // end dim2 loop
  
  } // end of boundary loop
  pop_time(BOUNDARIES);
}

//-----------------------------------------------------------------------------
void Species_evaluate_stochastic_events(Species_t* species, size_t time_ind)
{

  Geometry_t* geom = species->geom;
  unsigned int dbi, bi;
  const Flux_t* flux_info;

  // If substeping 
  if (geom->num_global_discrete_boundaries==0 || \
      time_ind % species->stochastic_substep != 0)
    return;

  // Communicate species values at discrete boundaries to processor 0
  push_time(STOCH);
  Species_communicate_values_at_discrete_boundaries(species);

  // Iterate over discrete boundaries
  for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
  {
    // Get the boundary index
    bi = geom->discrete_boundary_indices[dbi];

    // Get flux information, i.e. flux states and evaluation function. 
    // If there is not flux connected to the proccessed boundary, 
    // then go to the next one
    flux_info = BoundaryFluxes_get_flux_info(species->boundary_fluxes, bi);
    if (!flux_info)
      continue;
    
    // On discrete boundaries only fluxes with additional states
    // variables can be used. Make sure that it happens.
    assert(flux_info->storef);
    
    // Iterate over discrete bondaries on processor 0 and evaluate stochastic events
    if (geom->mpi_info.rank == 0)
    {
      // Evaluate stochastic state updates
      flux_info->storef->evaluate(flux_info->flux_params, flux_info->storef->states,
        time_ind*species->dt, species->dt*species->stochastic_substep,
        geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[dbi]);
    }
  }

  // Communicate the openness of the discrete boundaries from rank 0 to other processes
  Species_communicate_openness_of_discrete_boundaries(species);

  pop_time(STOCH);
}

//-----------------------------------------------------------------------------
void Species_compute_convolution_with_gaussian_function_locally(Species_t* species)
{
  if (!species->linescan_data)
    return;

  push_time(CONV);

  // Declare loop indices
  int di, axis_ind, x_ind, y_ind, z_ind, x_ind_geom, y_ind_geom, z_ind_geom;
  
  // Get the pointer where data for considered species start from.
  REAL* sp = species->u1[species->linescan_data->species_id];
  REAL* offsets = species->linescan_data->offsets;
  const int yz_offset = species->n[Y]*species->n[Z], y_offset = species->n[Z];
  const int geometry_yz_offset = species->geom->n[Y]*species->geom->n[Z], geometry_y_offset = species->geom->n[Z];
  
  // Get coordinate offsets to compute correct coordinates for each process
  hsize_t coord_offsets[NDIMS];
  for (axis_ind = 0; axis_ind<NDIMS; ++axis_ind)
    coord_offsets[axis_ind] = species->geom->offsets[axis_ind]*species->geom->subdivisions;
    
  // Every time the function is called clear the array that hold values
  // of the convolution
  memfill(species->linescan_data->sheet_save, species->N[species->linescan_data->axis], 0.0);
  
  // FIXME: Move the constants to double_parameters.h5
  //const REAL s_x = 20000/M_LN2, s_y = 20000/M_LN2, s_z = 80000/M_LN2;
  
  // First loop is over the axis we want to compute the average.
  for(axis_ind=0; axis_ind<species->N[species->linescan_data->axis]; ++axis_ind)
  {
    // Update the offset
    offsets[species->linescan_data->axis] = axis_ind*species->geom->h;
    // Sum over the axis X, Y and Z locally.
    REAL sheet_save = 0;
    #pragma omp parallel for private(y_ind, z_ind, x_ind_geom, y_ind_geom, z_ind_geom, di) reduction(+:sheet_save)
    for(x_ind=0; x_ind<species->n[X]; ++x_ind)
    {
      x_ind_geom = x_ind/species->geom->subdivisions;
      const REAL x = (x_ind + coord_offsets[X])*species->geom->h;
      for(y_ind=0; y_ind<species->n[Y]; ++y_ind)
      {
        y_ind_geom = y_ind/species->geom->subdivisions;
        const REAL y = (y_ind + coord_offsets[Y])*species->geom->h;
        for(z_ind=0; z_ind<species->n[Z]; ++z_ind)
        {
          z_ind_geom = z_ind/species->geom->subdivisions;
          const REAL z = (z_ind + coord_offsets[Z])*species->geom->h;
          
          // If domains were provided get domain index of the current species voxel
          // and verifies if the index is on the list of provided domains.
          if (species->linescan_data->num_domains)
          {
            const domain_id voxel_domain = species->geom->domains[x_ind_geom*geometry_yz_offset + y_ind_geom*geometry_y_offset + z_ind_geom];
            // Loop over all chosen domains
            for(di=0; di<species->linescan_data->num_domains; ++di)
            {
              if (voxel_domain == di)
              {
                sheet_save += sp[x_ind*yz_offset + y_ind*y_offset + z_ind]*gauss(offsets[X]-x, species->conv[X]) \
                                                                                                        *gauss(offsets[Y]-y, species->conv[Y]) \
                                                                                                        *gauss(offsets[Z]-z, species->conv[Z]);
                break;
              }
            }
          }
          // No specified domains. Compute everywhere.
          else
            sheet_save += sp[x_ind*yz_offset + y_ind*y_offset + z_ind]*gauss(offsets[X]-x, species->conv[X]) \
                                                                                                        *gauss(offsets[Y]-y, species->conv[Y]) \
                                                                                                        *gauss(offsets[Z]-z, species->conv[Z]);
        } // End of loop over z coordinate
      } // End of loop over y coordinate
    } // End of loop over x coordinate
    species->linescan_data->sheet_save[axis_ind] += sheet_save;
  }// End of outer loop
  pop_time(CONV);
}

//-----------------------------------------------------------------------------
void Species_write_convolution_to_file(Species_t* species, hid_t file_id, char* groupname)
{
  int di;
  const int rank = species->geom->mpi_info.rank;
  const int axis_size = species->N[species->linescan_data->axis];
  
  MPI_Comm comm = species->geom->mpi_info.comm;
  char datasetname[MAX_FILE_NAME];
  const char axis[3] = {'x', 'y', 'z'};
  char subname[4];
  char* domains_name;
  
  REAL* local_buffer = NULL;
  
  if (rank == 0)
    local_buffer = mpi_malloc(comm, axis_size*sizeof(REAL));
    
  // Collect partial convolution data such that the i-th elements from each
  // array are summed into the ith element in result array of process 0.
  MPI_Reduce(species->linescan_data->sheet_save, local_buffer, axis_size, MPIREAL, MPI_SUM, 0, comm);

  // Write the data to file
  if (rank == 0)
  {
    const int dir = species->linescan_data->axis;
    const int offset_one = mod(dir+(int)pow(-1, dir), 3), offset_two = mod(dir+(int)pow(-1, dir)*2, 3);
    
    domains_name = malloc(sizeof(char)*species->linescan_data->num_domains*4);
    domains_name[0] = '\0';
    for (di=0; di<species->linescan_data->num_domains; ++di)
    {
      sprintf(subname, "%d_", species->linescan_data->domains[di]);
      strcat(domains_name, subname);
    }
    
    sprintf(datasetname, "%s/linescan_%s_%s%c_%d_%c_%d", groupname, 
            species->linescan_data->species_name, domains_name,
            axis[offset_one], species->linescan_data->pos_offsets[offset_one],
            axis[offset_two], species->linescan_data->pos_offsets[offset_two] );

    free(domains_name);
    
    hsize_t size[1] = {(hsize_t)axis_size};
    hid_t filespace = H5Screate_simple(1, size, NULL); 
    hid_t memspace =  H5Screate_simple(1, size, NULL); 
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    hid_t dset_id = H5Dcreate(file_id, datasetname, H5REAL, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Pclose(plist_id);
    H5Sclose(filespace);
    
    filespace = H5Dget_space(dset_id);
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    
    H5Dwrite(dset_id, H5REAL, memspace, filespace, plist_id, local_buffer);
    
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dset_id);
    H5Sclose(memspace);
  
    free(local_buffer);
  }
}

//-----------------------------------------------------------------------------
void Species_output_init_values(Species_t* species)
{
  unsigned int i, j, k, bi, di, si;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;

  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "Model parameters:\n");
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");

  mpi_printf0(comm, "%8s", "Init:");
  for (i=0; i<species->num_species; i++)
    mpi_printf0(comm, "%13s", species->species_names[i]);
  mpi_printf0(comm, "\n");
  for (i=0; i<geom->num_domains; i++)
  {
    mpi_printf0(comm, "%7s:", geom->domain_names[i]);
    for (j=0; j<species->num_species; j++)
    {
      if (species->init[i][j]>0)
        mpi_printf0(comm, "%13.2e", species->init[i][j]);
      else
        mpi_printf0(comm, "%13s", "-");
    }
    mpi_printf0(comm, "\n");
  }

  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "%8s", "Sigma:");
  for (i=0; i<species->num_all_diffusive; i++)
    mpi_printf0(comm, "%13s", species->species_names[species->all_diffusive[i]]);
  mpi_printf0(comm, "\n");
  for (i=0; i<geom->num_domains; i++)
  {
    mpi_printf0(comm, "%7s:", geom->domain_names[i]);
    for (j=0; j<species->num_all_diffusive; j++)
    {

      unsigned int jj = species->all_diffusive[j];
      unsigned char diffusive = 0;
      for (k=0; k<species->num_diffusive[i]; k++)
      {
        if (species->diffusive[i][k]==jj)
        {
          mpi_printf0(comm, "%13.2e", species->sigma[i][k]);
          diffusive = 1;
        }
      }
      if (!diffusive)
        mpi_printf0(comm, "%13s", "-");
    }
    mpi_printf0(comm, "\n");
  }

  // Print buffer information if any
  if (species->num_all_buffers>0)
  {
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "%8s", "B_tot:");
    for (i=0; i<species->num_species; i++)
      mpi_printf0(comm, "%13s", species->all_buffers_b[i] ? species->species_names[i] : "");
    mpi_printf0(comm, "\n");
    for (i=0; i<geom->num_domains; i++)
    {
      mpi_printf0(comm, "%7s:", geom->domain_names[i]);
      bi = 0;
      for (j=0; j<species->num_species; j++)
      {
    
        if (species->all_buffers_b[j])
        {
          if (bi<species->num_buffers[i] && species->bsp0[i][bi]==j)
            mpi_printf0(comm, "%13.2e", species->tot[i][bi++]);
          else
            mpi_printf0(comm, "%13s", "-");
        }
        else
          mpi_printf0(comm, "%13s", "");
      }
      mpi_printf0(comm, "\n");
    }
    
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "%8s", "k_off:");
    for (i=0; i<species->num_species; i++)
      mpi_printf0(comm, "%13s", species->all_buffers_b[i] ? species->species_names[i] : "");
    mpi_printf0(comm, "\n");
    for (i=0; i<geom->num_domains; i++)
    {
      mpi_printf0(comm, "%7s:", geom->domain_names[i]);
      bi = 0;
      for (j=0; j<species->num_species; j++)
      {
    
        if (species->all_buffers_b[j])
        {
          if (bi<species->num_buffers[i] && species->bsp0[i][bi]==j)
            mpi_printf0(comm, "%13.2e", species->k_off[i][bi++]);
          else
            mpi_printf0(comm, "%13s", "-");
        }
        else
          mpi_printf0(comm, "%13s", "");
      }
      mpi_printf0(comm, "\n");
    }
    
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "%8s", "k_on:");
    for (i=0; i<species->num_species; i++)
      mpi_printf0(comm, "%13s", species->all_buffers_b[i] ? species->species_names[i] : "");
    mpi_printf0(comm, "\n");
    for (i=0; i<geom->num_domains; i++)
    {
      mpi_printf0(comm, "%7s:", geom->domain_names[i]);
      bi = 0;
      for (j=0; j<species->num_species; j++)
      {
    
        if (species->all_buffers_b[j])
        {
          if (bi<species->num_buffers[i] && species->bsp0[i][bi]==j)
            mpi_printf0(comm, "%13.2e", species->k_on[i][bi++]);
          else
            mpi_printf0(comm, "%13s", "-");
        }
        else
          mpi_printf0(comm, "%13s", "");
      }
      mpi_printf0(comm, "\n");
    }
  }

  if (species->num_fixed_domains>0)
  {
    
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "%8s\n", "Fixed domain species:");
    for (i=0; i<species->num_fixed_domains; i++)
    {
      di = species->fixed_domains[2*i];
      si = species->fixed_domains[2*i+1];
      mpi_printf0(comm, "%8s:%8s\n", geom->domain_names[di], species->species_names[si]);
    }
  }
}

//-----------------------------------------------------------------------------
void Species_output_data(Species_t* species, unsigned int output_ind, REAL t, 
       unsigned long time_ind, arguments_t* arguments)
{
  
  char filename[MAX_FILE_NAME];
  char groupname[MAX_SPECIES_NAME];
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  hid_t file_id=0, plist_id=0, group_id=0;
  
  sprintf(filename, "%s.h5", arguments->casename);
  sprintf(groupname, "/data_%07d", output_ind);

  if (geom->mpi_info.rank==0)
  {

    plist_id = H5Pcreate(H5P_FILE_ACCESS);

    // If first time create file
    if (output_ind==0)
    {
      file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      
      // Write argument information to file
      write_h5_attr(comm, file_id, "h", H5REAL, 1, &species->geom->h);
      write_h5_attr(comm, file_id, "N", H5T_STD_U64LE, NDIMS, species->N);
      write_h5_attr(comm, file_id, "geometry_file", H5T_STRING, 
              strlen(arguments->geometry_file), arguments->geometry_file);
      write_h5_attr(comm, file_id, "species_file", H5T_STRING, 
              strlen(arguments->model_file), arguments->model_file);
    }

    // If not first time open file
    else
    {
      file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);
    }

    // Create data group
    group_id = H5Gcreate(file_id, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write time and time index
    write_h5_attr(comm, group_id, "time", H5REAL, 1, &t);
    write_h5_attr(comm, group_id, "time_index", H5T_STD_U64LE, 1, &time_ind);
    
  }

  // Write scalar data to file
  Species_output_scalar_data(species, file_id, groupname, t, time_ind, output_ind, 
                             arguments->silent);

  // Write linescans data to file
  if (species->linescan_data)
    Species_write_convolution_to_file(species, file_id, groupname);
  
  if (geom->mpi_info.rank==0)
  {
    H5Gclose(group_id);
    H5Pclose(plist_id);
    H5Fclose(file_id);
  }

  // Check for saving 2D sheets
  if (arguments->num_save_species)
  {

    // Open the save file in collective mode
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
    file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);

    // Get center index
    Species_write_2D_sheet_to_file(species, file_id, groupname);
    
    H5Pclose(plist_id);
    H5Fclose(file_id);
  }
  
  if (arguments->all_data_num_species)
  {
    // Open an output file in collective mode
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
    file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);
    
    Species_write_all_data_to_file(species, file_id, groupname);
    
    H5Pclose(plist_id);
    H5Fclose(file_id);
  }
}

//-----------------------------------------------------------------------------
void Species_write_2D_sheet_to_file(Species_t* species, hid_t file_id, char* groupname)
{
  
  Geometry_t* geom = species->geom;
  unsigned int loc_X, loc_Y, dataset_ind;
  char datasetname[MAX_FILE_NAME];
  size_t ssi, si, loc_x_ind, loc_y_ind, sheet_ind, loc_x_offset;
  
  // Define temporary variables
  char ax_name[3] = {'x', 'y', 'z'};
  hsize_t dimsf[2], chunk_dims_full[2], offset[2];
  hsize_t chunk_dims_empty[2] = {0, 0};
  hsize_t count[2] = {1, 1};
  hsize_t stride[2] = {1,1};
  hsize_t* chunk_dims;
  
  // Iterate over the save species and save then using local chunks
  for (ssi=0; ssi<species->num_save_species; ssi++)
  {
    // Get species
    si = species->ind_save_species[ssi];
    
    unsigned int ax;
    for (ax=0; ax<3; ax++)
    {
      // Check if there are sheets to save in a certain direction
      if(species->sheets_per_species[ax] == 0)
        continue;
  
      // define local 2D projection in the following way:
      // x -> (loc_X, loc_Y) = (Y,Z)
      // y -> (loc_X, loc_Y) = (Z,X)
      // z -> (loc_X, loc_Y) = (X,Y)
      // They are provided in order to avoid copying parts of code
      loc_X = (ax+1)%3;
      loc_Y = (ax+2)%3;

      // Full data set dimension
      dimsf[0] = species->N[loc_X];
      dimsf[1] = species->N[loc_Y];

      // Local chunk dimension, count and offset into dataset
      chunk_dims_full[0] = species->n[loc_X];
      chunk_dims_full[1] = species->n[loc_Y];
      offset[0] = geom->offsets[loc_X]*geom->subdivisions; 
      offset[1] = geom->offsets[loc_Y]*geom->subdivisions;

      //printf("%d: chunk dims: [%zu,%zu]\n", geom->mpi_info.rank, chunk_dims[0], chunk_dims[1]);
      //printf("%d: offset: [%zu,%zu]\n", geom->mpi_info.rank, offset[0], offset[1]);

      for (sheet_ind=0; sheet_ind<species->sheets_per_species[ax]; sheet_ind++)
      {
        if (species->indices[ax][sheet_ind]>=0)
        {
          chunk_dims = chunk_dims_full;

          // Copy data to transfer memory
          for (loc_x_ind=0; loc_x_ind<species->n[loc_X]; loc_x_ind++)
          {
            loc_x_offset = loc_x_ind*species->n[loc_Y];
            for (loc_y_ind=0; loc_y_ind<species->n[loc_Y]; loc_y_ind++)
            {
              // Unfortunately I need to calculate dataset_ind every time
              // since the ax along which slices are saved changes
              // Otherwise, I would need to copy the whole for loop over axes
              if (ax == 0) // save a YZ slice 
                dataset_ind = species->indices[ax][sheet_ind]*species->n[Y]*species->n[Z]+ \
                              loc_x_ind*species->n[Z] + loc_y_ind;
              else if (ax == 1) //save a ZX slice
                dataset_ind = loc_y_ind*species->n[Y]*species->n[Z]+ \
                              species->indices[ax][sheet_ind]*species->n[Z] + loc_x_ind;
              else //save a XY slice
                dataset_ind = loc_x_ind*species->n[Y]*species->n[Z]+ \
                              loc_y_ind*species->n[Z] + species->indices[ax][sheet_ind];
                        
              species->sheets_save[ax][loc_x_offset+loc_y_ind] = \
                                  species->u1[si][dataset_ind];
            }
          }
        }
        else
          chunk_dims = chunk_dims_empty;

        // Create data field name
        sprintf(datasetname, "%s/%s_%c_sheet_%d", groupname, species->species_names[si],
              ax_name[ax], species->global_indices[ax][sheet_ind]);

        // file and dataset identifiers
        hid_t filespace = H5Screate_simple(2, dimsf, NULL); 
        //printf("created filespace:\n");
        
        hid_t memspace  = H5Screate_simple(2, chunk_dims, NULL); 
        //printf("created memspace:\n");
  
        // Create chunked dataset. 
        hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
        //H5Pset_chunk(plist_id, 2, chunk_dims);
        
        //printf("created chunked dataset0:\n");
  
        // Create compression filter (Not supported in parallel yet...)
        //unsigned int gzip_level = 9;
        //herr_t status = H5Pset_filter(plist_id, H5Z_FILTER_DEFLATE, 
        //          H5Z_FLAG_OPTIONAL, 1, &gzip_level);
  
        hid_t dset_id = H5Dcreate(file_id, datasetname, H5REAL, filespace, 
                    H5P_DEFAULT, plist_id, H5P_DEFAULT);
      
        //printf("created chunked dataset1:\n");
        H5Pclose(plist_id);
        H5Sclose(filespace);
  
        // Select hyperslab in the file.
        filespace = H5Dget_space(dset_id);
        //printf("get data space\n");
    
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, \
                            stride, count, chunk_dims);
  
        //printf("create hyper slab\n");
        // Create property list for collective dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  
        //printf("save data\n");
        H5Dwrite(dset_id, H5REAL, memspace, filespace, plist_id, \
                species->sheets_save[ax]);
  
        //printf("data saved\n");
  
        // Close/release resources.
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
        H5Sclose(memspace);
      }
    }
  }
}

//-----------------------------------------------------------------------------
void Species_write_all_data_to_file(Species_t* species, hid_t file_id, char* groupname)
{

  hsize_t dimsf[3] = {species->N[X], species->N[Y], species->N[Z]};
  hsize_t chunk_dims[3] = {species->n[X], species->n[Y], species->n[Z]};
  hsize_t count[3] = {1,1,1};
  hsize_t stride[3] = {1,1,1};
  hsize_t offset[3] = {species->geom->offsets[X]*species->geom->subdivisions,
                      species->geom->offsets[Y]*species->geom->subdivisions,
                      species->geom->offsets[Z]*species->geom->subdivisions};

  char datasetname[MAX_FILE_NAME];
  size_t ssi, si;
  
  // Iterate over the all data save species and save
  for (ssi=0; ssi<species->all_data_num_species; ssi++)
  {
    // Get species
    si = species->all_data_ind_species[ssi];

    sprintf(datasetname, "%s/%s_all_data", groupname, species->species_names[si]);

    // file and dataset identifiers
    hid_t filespace = H5Screate_simple(3, dimsf, NULL);
    hid_t memspace  = H5Screate_simple(3, chunk_dims, NULL);

    // Create chunked dataset. 
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 3, chunk_dims);

    // Create dataset id
    hid_t dset_id = H5Dcreate(file_id, datasetname, H5REAL, filespace, 
                    H5P_DEFAULT, plist_id, H5P_DEFAULT);
      
    //printf("created chunked dataset1:\n");
    H5Pclose(plist_id);
    H5Sclose(filespace);

    // Select hyperslab in the file.
    filespace = H5Dget_space(dset_id);

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, \
                    stride, count, chunk_dims);

    //printf("create hyper slab\n");
    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    //printf("save data\n");
    H5Dwrite(dset_id, H5REAL, memspace, filespace, plist_id, species->u1[si]);                              

    //printf("data saved\n");

    // Close/release resources.
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dset_id);
    H5Sclose(memspace);
  }
}

void Species_communicate_values_at_discrete_boundaries(Species_t* species)
{
  
  int dbi, dbj, bi, sendcount;
  unsigned int yz_offset, y_offset, xi, yi, zi, ghost_offset, ind, g_ind, dsi;
  unsigned short ghost_first_coord, ghost_second_coord;
  unsigned int* loc_boundary_voxels;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  REAL* send_buff;
  REAL* receive_buff = NULL;
  int* displs = NULL;
  int* recvcounts = NULL;
  const Flux_t* flux_info = NULL;

  // Only consider "Ca or the first species"
  REAL* u = species->u1[0];
  
  // Iterate over discrete boundaries
  yz_offset = species->n[Y]*species->n[Z];
  y_offset  = species->n[Z];

  for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
  {
    
    // Get the boundary index
    bi = geom->discrete_boundary_indices[dbi];

    // Iterate over the local boundaries and collect species values
    for (dbj=0; dbj<geom->local_boundary_size[bi]; dbj++)
    {
      // Get start of indices
      loc_boundary_voxels = &species->boundary_voxels[bi][dbj*2*NDIMS];
      
      // Collect from voxel indices
      xi = loc_boundary_voxels[X];
      yi = loc_boundary_voxels[Y];
      zi = loc_boundary_voxels[Z];
#if DEBUG
      printf("FROM: %d|%d:[%d,%d,%d]=%.2f\n", dbi, dbj, xi, yi, zi, \
             u[xi*yz_offset+yi*y_offset+zi]);
#endif
      // FIXME: Consider moving this to Species_t
      geom->species_values_at_local_discrete_boundaries[dbi][dbj*2] = \
        u[xi*yz_offset+yi*y_offset+zi];

      // Increase pointer
      loc_boundary_voxels += NDIMS;
      
      // Collect to voxel indices
      xi = loc_boundary_voxels[X];
      yi = loc_boundary_voxels[Y];
      zi = loc_boundary_voxels[Z];
#if DEBUG
      printf("TO:   %d|%d:[%d,%d,%d]=%.2f\n", dbi, dbj, xi, yi, zi, \
             u[xi*yz_offset+yi*y_offset+zi]);
#endif

      geom->species_values_at_local_discrete_boundaries[dbi][dbj*2+1] = \
        u[xi*yz_offset+yi*y_offset+zi];
    }
    
    
    // Gather information on species values at ghost discrete boundaries.
    // First we need to get diffusive species index, and then iterating
    // over ghost boundaries we can extract values from species->ghost_values_send
    // and species->ghost_values_receive variables.
    
    // Get flux function and params to retrieve diffusive species index.
    flux_info = BoundaryFluxes_get_flux_info(species->boundary_fluxes, bi);
    assert(flux_info);

    // Get diffusive species index
    dsi = flux_info->flux_dsi;
    
    // Iterate over ghost boundary voxels
    unsigned int gbi, si_offset = 2*geom->local_boundary_size[bi];
    unsigned int dim, dim2;
    for (dim2=0; dim2<2*NDIMS; dim2++)
    {
      // Extract the correct dimension
      dim = dim2/2;
      if (dim == X)
        ghost_offset = species->n[Z];
      else if (dim == Y)
        ghost_offset = species->n[Z];
      else
        ghost_offset = species->n[Y];
      
      for( gbi=0; gbi<geom->num_local_ghost_boundaries[bi][dim2]; gbi++)
      {
        // Retrieve ghost boundary voxel position and transform it into
        // species coordinates. Since we deal with discrete boundaries,
        // the obtained index need to be shifted.
        ghost_first_coord = geom->local_ghost_boundaries[bi][dim2][3*gbi+1]*geom->subdivisions + geom->subdivisions/2;
        ghost_second_coord = geom->local_ghost_boundaries[bi][dim2][3*gbi+2]*geom->subdivisions + geom->subdivisions/2;
        
        // Calculate the offset
        ind = ghost_first_coord*ghost_offset + ghost_second_coord;
  g_ind = species->u_offsets[dim2]
    + ghost_first_coord*species->u_outer_offsets[dim2]
    + ghost_second_coord*species->u_inner_offsets[dim2];
        
        // Check if the type of the ghost boundary is
        // ghost-to-inner
        if (geom->local_ghost_boundaries[bi][dim2][3*gbi])
        {
          geom->species_values_at_local_discrete_boundaries[dbi][si_offset+gbi*2] = 
              species->ghost_values_receive[dim2][dsi][ind];
          geom->species_values_at_local_discrete_boundaries[dbi][si_offset+gbi*2+1] =
        species->u1[species->all_diffusive[dsi]][g_ind];//species->ghost_values_send[dim2][dsi][ind];
        }
        else // inner-to-ghost
        {
          geom->species_values_at_local_discrete_boundaries[dbi][si_offset+gbi*2] = 
              species->u1[species->all_diffusive[dsi]][g_ind];//species->ghost_values_send[dim2][dsi][ind];
          geom->species_values_at_local_discrete_boundaries[dbi][si_offset+gbi*2+1] = 
              species->ghost_values_receive[dim2][dsi][ind];
        }
      }
      si_offset += 2*geom->num_local_ghost_boundaries[bi][dim2];
    }
    
    
    
    // Assign Gatherv arguments
    send_buff = geom->species_values_at_local_discrete_boundaries[dbi]; 
    sendcount = geom->num_local_discrete_boundaries[dbi]*2;
    recvcounts = geom->num_local_species_values_at_discrete_boundaries_rank_0[dbi];
    displs = geom->offset_num_species_values_discrete_boundaries_rank_0[dbi];
    receive_buff = geom->species_values_at_local_discrete_boundaries_rank_0[dbi];
    
    // Do the assymetric MPI call
    MPI_Gatherv(send_buff, sendcount, MPIREAL,
                receive_buff, recvcounts, displs, MPIREAL, 0, comm);

    // Transfer species values to globally correct order
    if (geom->mpi_info.rank==0)
    {
      
      int gdbi;
#if DEBUG
      printf("Receiving discrete values: ");
#endif
      for (dbj=0; dbj<geom->num_discrete_boundaries[dbi]; dbj++)
      {

        // Get global index number for discrete boundary
        gdbi = geom->local_distribution_of_global_discrete_boundaries_rank_0[dbi][dbj];

        // Transfer the two values 
        // FIXME: Make this more flexible in the future by transfering maybe all 
        // FIXME: species values?
        geom->species_values_at_local_discrete_boundaries_correct_order_rank_0\
          [dbi][gdbi*2] = geom->species_values_at_local_discrete_boundaries_rank_0[dbi][dbj*2];
        geom->species_values_at_local_discrete_boundaries_correct_order_rank_0\
          [dbi][gdbi*2+1] = geom->species_values_at_local_discrete_boundaries_rank_0[dbi][dbj*2+1];
#if DEBUG
        printf("%.2f, ", geom->species_values_at_local_discrete_boundaries_rank_0[dbi][dbj*2]);
        printf("%.2f, ", geom->species_values_at_local_discrete_boundaries_rank_0[dbi][dbj*2+1]);
#endif

      }      
#if DEBUG
      printf("\n\n");
#endif
    }
  }
}

//-----------------------------------------------------------------------------
void Species_communicate_openness_of_discrete_boundaries(Species_t* species)
{
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  unsigned int bi, dbi, i, gdbi;
  int* send_buff;
  int* send_counts;
  int* displs;
  int* receive_buff;
  int recv_count;
  
  const Flux_t* flux_info;

  // Iterate over discrete boundaries
  for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
  {
    // Get the boundary index
    bi = geom->discrete_boundary_indices[dbi];
    
    // Get flux information, i.e. flux states. If there is not flux 
    // connected to the proccessed boundary, then go to the next one
    flux_info = BoundaryFluxes_get_flux_info(species->boundary_fluxes, bi);
    if (!flux_info)
      continue;
      
    // On discrete boundaries only fluxes with additional states
    // variables can be used. Make sure that it happens.
    assert(flux_info->storef);
    
    // Iterate over discrete bondaries on processor 0 and gather openness
    if (geom->mpi_info.rank == 0)
    {
    
      // Iterate each discrete boundary
      for (i=0; i<geom->num_discrete_boundaries[dbi]; i++)
      {

        // Get global index number for discrete boundary
        gdbi = geom->local_distribution_of_global_discrete_boundaries_rank_0[dbi][i];

        // Evaluate the openess of currently processed flux
        geom->open_local_discrete_boundaries_rank_0[dbi][i] = flux_info->storef->open_states(flux_info->storef->states, gdbi);
          
      }
    }

    // Communicate the opennes of the discrete boundaries to the local processes
    send_buff = geom->open_local_discrete_boundaries_rank_0[dbi];
    send_counts = geom->num_local_discrete_boundaries_rank_0[dbi];
    displs = geom->offset_num_local_discrete_boundaries_rank_0[dbi];
    receive_buff = geom->open_local_discrete_boundaries[dbi];
    recv_count = geom->num_local_discrete_boundaries[dbi];

    // Communicate the opennes from process 0 to all other processes
    MPI_Scatterv(send_buff, send_counts, displs, MPI_INT, 
                 receive_buff, recv_count, MPI_INT,
                 0, comm);
  }
}

//-----------------------------------------------------------------------------
void Species_output_scalar_data(Species_t* species, hid_t file_id, char* groupname, 
        REAL t, unsigned long time_ind, unsigned long output_ind, 
                                domain_id silent)
{
  
  char domain_groupname[MAX_FILE_NAME];
  char species_groupname[MAX_FILE_NAME];
  char datasetname[MAX_FILE_NAME];
  unsigned int i, bi, dbi, xi, yi, zi, xi_offset, yi_offset, xi_offset_geom;
  unsigned int yi_offset_geom, zi_geom, di_si;

  hid_t domain_group_id, species_group_id;
  domain_id di, si;
  Geometry_t* geom = species->geom;
  int rank = geom->mpi_info.rank;
  MPI_Comm comm = geom->mpi_info.comm;
  MPIInfo* mpi_info = &geom->mpi_info;
  REAL* buff = NULL;
  REAL value;
  unsigned int num_domain_species = geom->num_domains*species->num_species;

  // Local buffers
  REAL* max_values = mpi_malloc(comm, sizeof(REAL)*num_domain_species);
  memfill(max_values, num_domain_species, FLT_MIN);

  REAL* min_values = mpi_malloc(comm, sizeof(REAL)*num_domain_species);
  memfill(min_values, num_domain_species, FLT_MAX);

  REAL* total_values = mpi_malloc(comm, sizeof(REAL)*num_domain_species);
  memfill(total_values, num_domain_species, 0.);

  // Allocate memory for communication buffer
  if (rank==0)
    buff = mpi_malloc(comm, sizeof(REAL)*num_domain_species*mpi_info->size);

  // Iterate over species and collect min, max and total values
  for (si=0; si<species->num_species; si++)
  {

    #pragma omp parallel private(yi, zi, xi_offset_geom, xi_offset, yi_offset_geom, yi_offset, zi_geom, di, value, di_si)
    {
    REAL* max_valuest = malloc(sizeof(REAL)*num_domain_species);
    memfill(max_valuest, num_domain_species, FLT_MIN);

    REAL* min_valuest = malloc(sizeof(REAL)*num_domain_species);
    memfill(min_valuest, num_domain_species, FLT_MAX);

    REAL* total_valuest = malloc(sizeof(REAL)*num_domain_species);
    memfill(total_valuest, num_domain_species, 0.);

    #pragma omp for collapse(2)
    for (xi=0; xi<species->n[X]; xi++) {
      for (yi=0; yi<species->n[Y]; yi++) {
        xi_offset_geom = (xi/geom->subdivisions)*geom->n[Y]*geom->n[Z];
        xi_offset = xi*species->n[Y]*species->n[Z];
        yi_offset_geom = (yi/geom->subdivisions)*geom->n[Z];
        yi_offset = yi*species->n[Z];
        for (zi=0; zi<species->n[Z]; zi++)
        {
          zi_geom = zi/geom->subdivisions;
          di = geom->domains[xi_offset_geom+yi_offset_geom+zi_geom];
          assert(di<geom->num_domains);
          value = species->u1[si][xi_offset+yi_offset+zi];
          di_si = di*species->num_species+si;
          min_valuest[di_si] = value < min_valuest[di_si] ? value : min_valuest[di_si];
          max_valuest[di_si] = value > max_valuest[di_si] ? value : max_valuest[di_si];
          total_valuest[di_si] += value*species->dV;
        }
      }
    }
    #pragma omp critical
    for(int i=0; i<num_domain_species; i++) {
      min_values[i] = min_valuest[i] < min_values[i] ? min_valuest[i] : min_values[i];
      max_values[i] = max_valuest[i] > max_values[i] ? max_valuest[i] : max_values[i];
      total_values[i] += total_valuest[i];
    }
    free(max_valuest), free(min_valuest), free(total_valuest);
    }
  }

  // Send all values to processor 0
  MPI_Gather(total_values, num_domain_species, MPIREAL, 
       buff, num_domain_species, MPIREAL, 0, mpi_info->comm);

  // Collect all values
  if (rank==0)
  {

    // Skip 0 rank
    for (i=1; i<mpi_info->size; i++)
    {
      for (di=0; di<geom->num_domains; di++)
      {
        for (si=0; si<species->num_species; si++)
        {
          total_values[di*species->num_species+si] +=     \
            buff[i*num_domain_species+di*species->num_species+si];
        }
      }
    }
  }
  
  // Send all values to processor 0
  MPI_Gather(min_values, num_domain_species, MPIREAL, 
       buff, num_domain_species, MPIREAL, 0, mpi_info->comm);

  // Collect all values
  if (rank==0)
  {

    // Skip 0 rank
    for (i=1; i<mpi_info->size; i++)
    {
      for (di=0; di<geom->num_domains; di++)
      {
        for (si=0; si<species->num_species; si++)
        {
          di_si = di*species->num_species+si;
          value = buff[i*num_domain_species+di*species->num_species+si];
          min_values[di_si] = value < min_values[di_si] ? value : min_values[di_si];
        }
      }
    }
  }

  // Send all values to processor 0
  MPI_Gather(max_values, num_domain_species, MPIREAL, 
       buff, num_domain_species, MPIREAL, 0, mpi_info->comm);

  // Collect all values
  if (rank==0)
  {

    // Skip 0 rank
    for (i=1; i<mpi_info->size; i++)
    {
      for (di=0; di<geom->num_domains; di++)
      {
        for (si=0; si<species->num_species; si++)
        {
          di_si = di*species->num_species+si;
          value = buff[i*num_domain_species+di*species->num_species+si];
          max_values[di_si] = value > max_values[di_si] ? value : max_values[di_si];
        }
      }
    }
    
    // Put discrete boundary values on rank 0 in correct order
    for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
    {

      for (bi=0; bi<geom->num_discrete_boundaries[dbi]; bi++)
      {
        unsigned int gdbi = geom->local_distribution_of_global_discrete_boundaries_rank_0[dbi][bi];
        geom->open_local_discrete_boundaries_correct_order_rank_0[dbi][gdbi] = \
          geom->open_local_discrete_boundaries_rank_0[dbi][bi];
        
      }
    }

    if (silent==0)
    {
      REAL conservation=0.;
      for (di=0; di<geom->num_domains; di++)
        for (si=0; si<species->num_species; si++)
        conservation += total_values[di*species->num_species+si];
      
      printf("\n%zu: time: %.2f ms, time_index: %zu, conservation: %g\n", \
             output_ind, t, time_ind, conservation);
      printf("-----------------------------------------------------------------------------\n");
      printf("%8s", "Average:");
      for (i=0; i<species->num_species; i++)
        printf("%13s", species->species_names[i]);
      printf("\n");
      for (di=0; di<geom->num_domains; di++)
      {
        printf("%7s:", geom->domain_names[di]);
        for (si=0; si<species->num_species; si++)
        {
        printf("%13.2e", total_values[di*species->num_species+si]/geom->volumes[di]);
        }
        printf("\n");
      }
      printf("\n");
      
      printf("%8s", "Min:");
      for (i=0; i<species->num_species; i++)
        printf("%13s", species->species_names[i]);
      printf("\n");
      for (di=0; di<geom->num_domains; di++)
      {
        printf("%7s:", geom->domain_names[di]);
        for (si=0; si<species->num_species; si++)
        {
        printf("%13.2e", min_values[di*species->num_species+si]);
        }
        printf("\n");
      }
      printf("\n");
      
      printf("%8s", "Max:");
      for (i=0; i<species->num_species; i++)
        printf("%13s", species->species_names[i]);
      printf("\n");
      for (di=0; di<geom->num_domains; di++)
      {
        printf("%7s:", geom->domain_names[di]);
        for (si=0; si<species->num_species; si++)
        {
        printf("%13.2e", max_values[di*species->num_species+si]);
        }
        printf("\n");
      }
      printf("\n");
      
      // Output openess of discrete boundaries
      if (geom->num_global_discrete_boundaries>0)
      {
        printf("%8s", "Channels:\n");
        for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
        {
        
          printf("%7s: ", geom->boundary_names[geom->discrete_boundary_indices[dbi]]);
          int sum = 0;
          for (bi=0; bi<geom->boundary_size[geom->discrete_boundary_indices[dbi]]; bi++)
          {
            sum += geom->open_local_discrete_boundaries_correct_order_rank_0[dbi][bi];
            printf("%d:%d", bi, geom->open_local_discrete_boundaries_correct_order_rank_0[dbi][bi]);
        
            if (bi<geom->boundary_size[dbi]-1)
            {
              printf(", ");
              if (bi%14 == 0 && bi!=0)
                printf("\n         ");
            }
          }
          printf(" | %d/%d\n", sum, (int)geom->boundary_size[geom->discrete_boundary_indices[dbi]]);
        }
      printf("\n");
      }
    }
    
    for (di=0; di<geom->num_domains; di++)
    {
    
      sprintf(domain_groupname, "%s/%s", groupname, geom->domain_names[di]);
      domain_group_id = H5Gcreate(file_id, domain_groupname, H5P_DEFAULT, \
              H5P_DEFAULT, H5P_DEFAULT);
      for (si=0; si<species->num_species; si++)
      {
      sprintf(species_groupname, "%s/%s", domain_groupname, species->species_names[si]);
      species_group_id = H5Gcreate(file_id, species_groupname, H5P_DEFAULT, 
                 H5P_DEFAULT, H5P_DEFAULT);
      
      // Write max value to file
      write_h5_attr(comm, species_group_id, "max", H5REAL, 1, \
              &max_values[di*species->num_species+si]);
      
      // Write min value to file
      write_h5_attr(comm, species_group_id, "min", H5REAL, 1,   \
              &min_values[di*species->num_species+si]);
      
      // Write average value to file
      REAL average = total_values[di*species->num_species+si]/geom->volumes[di];
      write_h5_attr(comm, species_group_id, "average", H5REAL, 1,   \
              &average);
    
      H5Gclose(species_group_id);
      
      }
      H5Gclose(domain_group_id);
    }
    
    for (dbi=0; dbi<geom->num_global_discrete_boundaries; dbi++)
    {
      sprintf(datasetname, "%s/discrete_%s", groupname, \
              geom->boundary_names[geom->discrete_boundary_indices[dbi]]);
    
      hsize_t size[1] = {(hsize_t)geom->boundary_size[geom->discrete_boundary_indices[dbi]]};
      hid_t filespace = H5Screate_simple(1, size, NULL); 
      hid_t memspace =  H5Screate_simple(1, size, NULL); 
      hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
      hid_t dset_id = H5Dcreate(file_id, datasetname, H5REAL, filespace, 
                    H5P_DEFAULT, plist_id, H5P_DEFAULT);
      H5Pclose(plist_id);
      H5Sclose(filespace);
      filespace = H5Dget_space(dset_id);
      plist_id = H5Pcreate(H5P_DATASET_XFER);
    
      H5Dwrite(dset_id, H5T_STD_I32LE, filespace, filespace, plist_id, 
               geom->open_local_discrete_boundaries_correct_order_rank_0[dbi]);

      H5Sclose(filespace);
      H5Pclose(plist_id);
      H5Dclose(dset_id);
      H5Sclose(memspace);
    }

    // Clean upp communication buffer
    free(buff);

  }

  free(total_values);
  free(max_values);
  free(min_values);

}

//-----------------------------------------------------------------------------
void Species_init_fixed_domain_species(Species_t* species, domain_id num_fixed_domains, 
                                       domain_id* fixed_domains)
{

  unsigned int i, si, xi, yi, zi, di, xii, yii, zii;
  unsigned int i_geom_offset[NDIMS], i_offset[NDIMS];
  size_t voxel_ind;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  size_t ind_fixed_single_species[species->num_species];
  memfill_size_t(ind_fixed_single_species, species->num_species, 0);

  // Iterate over fixed and collect the number of fixed voxels per species
  for (i=0; i<num_fixed_domains; i++)
  {
    
    // Get species index
    si = fixed_domains[2*i+1];

    // Iterate over the domains 
    for (xi=0; xi<geom->n[X]; xi++)
    {

      // Compute the X offsets 
      i_geom_offset[X] = xi*geom->n[Y]*geom->n[Z];

      // Iterate over the interior Y indices
      for (yi=0; yi<geom->n[Y]; yi++)
      {

        // Compute the Y offsets 
        i_geom_offset[Y] = yi*geom->n[Z];

        // Iterate over the interior Z indices
        for (zi=0; zi<geom->n[Z]; zi++)
        {
            
          // Get domain
          di = geom->domains[i_geom_offset[X]+i_geom_offset[Y]+zi];

          // If fixed domain increase the number of fixed voxels
          if (fixed_domains[2*i]==di)
            species->num_fixed_domain_species[si] += geom->subdivisions*geom->subdivisions* \
              geom->subdivisions;
        }
      }
    }
  }
  
  // Allocate memorty to hold index information for the fixed voxels
  for (si=0; si<species->num_species; si++)
    species->fixed_domain_species[si] = mpi_malloc(comm,                \
                     species->num_fixed_domain_species[si]*sizeof(size_t));

  // Iterate over given fixed domain species and store the voxel for each species
  for (i=0; i<num_fixed_domains; i++)
  {
    
    // Get species index
    si = fixed_domains[2*i+1]; 

    // Iterate over the domains 
    for (xi=0; xi<geom->n[X]; xi++)
    {

      // Compute the X offsets 
      i_geom_offset[X] = xi*geom->n[Y]*geom->n[Z];

      // Iterate over the interior Y indices
      for (yi=0; yi<geom->n[Y]; yi++)
      {

        // Compute the Y offsets 
        i_geom_offset[Y] = yi*geom->n[Z];

        // Iterate over the interior Z indices
        for (zi=0; zi<geom->n[Z]; zi++)
        {
            
          // Get domain
          di = geom->domains[i_geom_offset[X]+i_geom_offset[Y]+zi];

          // If domain is fixed for this species
          if (fixed_domains[2*i]==di)
          {
              
            // Iterate over all sub divisions
            for (xii=0; xii<geom->subdivisions; xii++)
            {
              
              // X offset in the species array
              i_offset[X] = (xi*geom->subdivisions+xii)*species->n[Z]*species->n[Y];

              for (yii=0; yii<geom->subdivisions; yii++)
              {

                // Y offset in the species array
                i_offset[Y] = (yi*geom->subdivisions+yii)*species->n[Z];

                for (zii=0; zii<geom->subdivisions; zii++)
                {
                  voxel_ind = i_offset[X]+i_offset[Y]+zi*geom->subdivisions+zii;
                  species->fixed_domain_species[si][\
                              ind_fixed_single_species[si]++] = voxel_ind;
                }
              }
            }
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
void Species_output_time_stepping_information(Species_t* species, REAL tstop, 
                size_t save_interval)
{
  unsigned int si, dsi;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;

  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "Time stepping:\n");
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");
  mpi_printf0(comm, "%13s %10.2g ms\n%13s%10.3g ms\n%13s%10zu\n%13s%10zu\n\n",
        "tstop", tstop, "dt", species->dt, "num dt", 
        (size_t)(tstop/species->dt), "save interval", save_interval);
  
  mpi_printf0(comm, "%13s%13s%10s\n", "Diffusion", "dt [ms]", "substeps");
  for (dsi=0; dsi<species->num_all_diffusive; dsi++)
  {
    si = species->all_diffusive[dsi];
    mpi_printf0(comm, "%13s%13.3g%10d\n", species->species_names[si], 
    species->species_sigma_dt[dsi],
    species->species_substeps[dsi]);
  }

  mpi_printf0(comm, "\n%13s%13s%10s\n", "", "dt [ms]", "substeps");
  mpi_printf0(comm, "%13s%13.3g%10d\n", "Discr bound", 
              species->stochastic_substep*species->dt,
              species->stochastic_substep);
  
  mpi_printf0(comm, "\n%13s%13s%10s\n", "", "dt [ms]", "substeps");
  mpi_printf0(comm, "%13s%13.3g%10d\n", "Reaction", 
              species->reaction_substep*species->dt,
              species->reaction_substep);
}

//-----------------------------------------------------------------------------
void Species_destruct(Species_t* species)
{
  unsigned int dim, dim2, dsi, di, si, bi;
  Geometry_t* geom = species->geom;

  for (di=0; di<geom->num_domains; di++)
  {

    if (species->init[di])
      free(species->init[di]);

    if (species->sigma[di])
      free(species->sigma[di]);
    
    if (species->diffusive[di])
      free(species->diffusive[di]);

    if (species->tot[di])
      free(species->tot[di]);

    if (species->k_on[di])
      free(species->k_on[di]);

    if (species->k_off[di])
      free(species->k_off[di]);

    if (species->bsp0[di])
      free(species->bsp0[di]);

    if (species->bsp1[di])
      free(species->bsp1[di]);

  }

  for (bi=0; bi<geom->num_boundaries; bi++)
  {
    free(species->boundary_voxels[bi]);
  }

  for (si=0; si<species->num_species; si++)
  {
    free(species->species_names[si]);
    free(species->u1[si]);
    free(species->du[si]);
    free(species->fixed_domain_species[si]);
  }
  
  for (dim2=0; dim2<NDIMS*2; dim2++)
  {

    for (dsi=0; dsi<species->num_all_diffusive; dsi++) 
    {
      if (species->ghost_values_receive[dim2][dsi])
        free(species->ghost_values_receive[dim2][dsi]);
      /*
      if (dim2>=2 && species->ghost_values_send[dim2][dsi])
        free(species->ghost_values_send[dim2][dsi]);
      */
    }
    /*
    free(species->ghost_values_send[dim2]);
    */
    free(species->ghost_values_receive[dim2]);
  }

  MPI_Type_free (&(species->vector_xzplane_type));
  MPI_Type_free (&(species->vector_xyplane_type));

  #if defined(ALPHA)
  for (dim=0; dim<NDIMS; dim++)
  {
    if (species->alpha[dim])
    {
      for (dsi=0; dsi<species->num_all_diffusive; dsi++) 
      {
        if (species->alpha[dim][dsi])
          free(species->alpha[dim][dsi]);
      }

      free(species->alpha[dim]);
    }
  }
  #endif

  // Free scratch space for saving sheets
  unsigned int ax;
  for(ax=0; ax<3; ax++)
    if (species->sheets_save[ax])
        free(species->sheets_save[ax]);

  // Free indices to save species
  if (species->ind_save_species)
    free(species->ind_save_species);
    
  if (species->linescan_data)
  {
    if (species->linescan_data->domains)
      free(species->linescan_data->domains);
    free(species->linescan_data->sheet_save);
    free(species->linescan_data);
  }


  // Free BoundaryFluxes
  BoundaryFluxes_destruct(species->boundary_fluxes);

  // FIXME: Add more clean ups
  free(species->fixed_domains);
  free(species->num_fixed_domain_species);
  free(species->num_boundary_voxels);
  free(species->boundary_voxels);
  free(species->local_domain_num);
  free(species->all_buffers);
  free(species->all_buffers_b);
  free(species->num_buffers);
  free(species->tot);
  free(species->k_on);
  free(species->k_off);
  free(species->bsp0);
  free(species->bsp1);

  free(species->bsp0T);
  free(species->bsp1T);
  free(species->k_onT);
  free(species->k_offT);
  free(species->totT);

  free(species->u1);
  free(species->num_diffusive);
  free(species->diffusive);
  free(species->sigma);
  free(species->species_sigma_dt);
  free(species->species_substeps);
  free(species->voxels_stencil_test); 
  free(species->init);
  free(species);
}

//-----------------------------------------------------------------------------
domain_id Species_check_open_discrete_boundaries(Species_t* species, REAL t, 
                                                 arguments_t* arguments)
{
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  int rank = geom->mpi_info.rank;
  int all_closed=0, sum_open = 0, sum_open_ryr = 0, sum_open_p_ryr = 0;
  int sp_ryr=0, sp_p_ryr=0;

  char * name_ryr = "ryr";
  char * name_p_ryr = "p_ryr";

  // If the abort option has not been chosen, do not proceed.
  if (!arguments->tabort || geom->num_global_discrete_boundaries==0 ){
    mpi_printf0(comm, "aborting");
    return 0;
    }

  // Only check for rank 0
  if (rank==0)
  {
    int di;
    for (di=0; di< geom->num_boundaries; di++)
    {
      if (strcmp(geom->boundary_names[di], name_ryr)==0){
        sp_ryr=1;
      }
      if (strcmp(geom->boundary_names[di], name_p_ryr)==0){
        sp_p_ryr=1;
      }
    }


    if (sp_ryr==1 && sp_p_ryr==0){
      const domain_id bi = Geometry_get_boundary_id(geom, name_ryr);
      const domain_id dbi = Geometry_get_discrete_boundary_id(geom, bi);

      int i;
      for (i=0; i<geom->boundary_size[geom->discrete_boundary_indices[dbi]]; i++)
        sum_open += geom->open_local_discrete_boundaries_correct_order_rank_0[dbi][i];

        // Verify if all channels for given boundary are closed
      if(sum_open == 0)
        all_closed = 1;
    }
    else if (sp_ryr==0 && sp_p_ryr==1){
      const domain_id bi = Geometry_get_boundary_id(geom, name_p_ryr);
      const domain_id dbi = Geometry_get_discrete_boundary_id(geom, bi);

      int i;
      for (i=0; i<geom->boundary_size[geom->discrete_boundary_indices[dbi]]; i++)
        sum_open += geom->open_local_discrete_boundaries_correct_order_rank_0[dbi][i];

        // Verify if all channels for given boundary are closed
      if(sum_open == 0)
        all_closed = 1;
    }
    else if (sp_ryr==1 && sp_p_ryr==1){
      const domain_id bi_ryr = Geometry_get_boundary_id(geom, name_ryr);
      const domain_id dbi_ryr = Geometry_get_discrete_boundary_id(geom, bi_ryr);

      int i;
      for (i=0; i<geom->boundary_size[geom->discrete_boundary_indices[dbi_ryr]]; i++)
        sum_open_ryr += geom->open_local_discrete_boundaries_correct_order_rank_0[dbi_ryr][i];

      const domain_id bi_p_ryr = Geometry_get_boundary_id(geom, name_p_ryr);
      const domain_id dbi_p_ryr = Geometry_get_discrete_boundary_id(geom, bi_p_ryr);

      for (i=0; i<geom->boundary_size[geom->discrete_boundary_indices[dbi_p_ryr]]; i++)
        sum_open_p_ryr += geom->open_local_discrete_boundaries_correct_order_rank_0[dbi_p_ryr][i];

        // Verify if all channels for given boundary are closed
      if(sum_open_ryr == 0 && sum_open_p_ryr == 0)
        all_closed = 1;
    }
  }

    // Send information about all_closed to all processes
    all_closed = mpi_max_int(comm, all_closed);
  
    // If not all closed we return false
    if (!all_closed)
    {
     // Reset flag for being opened
      species->last_opened_discrete_boundary = -1.0;
      return 0;
    }
  
 
  // All channels of chosen type are closed. Depending on the sign of the
  // provided argument with the --abort option choose the correct way of
  // handling the situation.
  if (arguments->tabort >= 0)
  {
    // If all closed for the first time
    if (species->last_opened_discrete_boundary<0)
      species->last_opened_discrete_boundary = t;
      
    // If the difference between present time and when it was last
    // opened is smaller than
    if (t-species->last_opened_discrete_boundary >= arguments->tabort)
      return 1;
  } 
  return 0;
}

//-----------------------------------------------------------------------------
domain_id Species_get_species_id(Species_t* species, char* species_name)
{
  domain_id si;
  for (si=0; si<species->num_species; ++si)
    if (strcmp(species->species_names[si], species_name) == 0)
      return si;
  
  mpi_printf_error(species->geom->mpi_info.comm, "*** ERROR: \"%s\" is not a valid "\
       "species name.\n", species_name);
  return 0;
}

//-----------------------------------------------------------------------------
domain_id Species_get_diffusive_species_id(Species_t* species, char* species_name)
{
  // First, find a species index
  domain_id dsi, si = Species_get_species_id(species, species_name);
      
  // Having species index find its diffusive counterpart
  for (dsi=0; dsi<species->num_all_diffusive; dsi++)
    if (species->all_diffusive[dsi] == si)
      return dsi;
      
  mpi_printf_error(species->geom->mpi_info.comm, "*** ERROR: \"%s\" is not a valid "\
       "diffusive species name.\n", species_name);
  return 0;
}

//-----------------------------------------------------------------------------