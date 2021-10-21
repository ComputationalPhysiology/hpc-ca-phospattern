#ifndef __SPECIES_H
#define __SPECIES_H

#include "types.h"
#include "options.h"
#include "geometry.h"

// Forward declaration
struct BoundaryFluxes;
typedef struct BoundaryFluxes BoundaryFluxes_t;

typedef struct LinescanData_
{
  int axis; // Axis along the convolution is computed: 0-X, 1-Y, 2-Z according to types.h
  char* species_name; // Species name. One of the listed ones in the parameters file
  domain_id species_id; // Species identifier
  domain_id* domains; // Indices the considered species lives on
  domain_id num_domains; // Number of chosen domains;
  int pos_offsets[3]; // Exact copy of command-line arguments
  REAL offsets[3]; // Offsets to shift the 3D Gaussian function
  REAL* sheet_save; // To store computed data
 
} LinescanData;

// Struct keeping the species information
struct Species {
  
  // The geometry
  Geometry_t* geom;
  
  // Two new MPI vector types for the communication of xz- and xy-planes
  MPI_Datatype vector_xzplane_type, vector_xyplane_type;
  
  // Struct keeping information about boundary fluxes
  BoundaryFluxes_t* boundary_fluxes;

  // Num species
  domain_id num_species;

  // Volume of each voxel
  REAL dV;
  
  // Global sizes species
  hsize_t N[NDIMS];
  
  // Local sizes species
  hsize_t n[NDIMS];

  // Species names
  char** species_names;

  // Species
  REAL** u1;
  REAL** du;

  // An array with indices for updating a species 
  domain_id* update_species;

  // What domain species should be fixed
  domain_id num_fixed_domains;
  domain_id* fixed_domains;

  size_t* num_fixed_domain_species;
  size_t** fixed_domain_species;
  
  // Memory to save sheets
  REAL* sheets_save[3];

  // Local number of voxels in each domain
  long* local_domain_num;

  domain_id* voxels_domain_ids; // domain id for each local computational voxel
  char * voxels_stencil_test;
  domain_id* ghost_voxels_domain_ids[NDIMS*2];

  // Sheet offset for ghost value extractions
  size_t u_offsets[NDIMS*2];
  size_t u_outer_offsets[NDIMS*2];
  size_t u_inner_offsets[NDIMS*2];
  size_t u_inner_sheet_offsets[NDIMS*2];

  // Num all diffusive species
  domain_id num_all_diffusive;

  // All diffusive species 
  domain_id* all_diffusive;

  // All diffusive species 
  domain_id* is_diffusive;

  // Num diffusive species per domain
  domain_id* num_diffusive;

  // Diffusive species per domain per species
  domain_id** diffusive;

  // Sigmas per domain per species
  REAL** sigma;

  // alpha values for the x,y,z directions for each diffusive species.
  REAL** alpha[NDIMS];

  // Init values per domain per species
  REAL** init;

  // Ghost values and send receive buffers
  // ghost_values_receive[dim2][species]
  REAL** ghost_values_receive[NDIMS*2];
  //REAL** ghost_values_send[NDIMS*2];

  // Size of each diffusive species in each ghost_value buffer
  size_t size_ghost_values[NDIMS*2];

  // Num all buffers
  domain_id num_all_buffers;
  
  // Species indices of all buffers
  domain_id* all_buffers;
  
  // Species indices of all buffers 
  unsigned char* all_buffers_b;
  
  // Num buffer reactions per domain 
  domain_id* num_buffers;

  // Buffer information per domain per buffer
  REAL** k_off;
  REAL** k_on;
  REAL** tot;

  REAL* k_offT;
  REAL* k_onT;
  REAL* totT;

  // Indices to the involved species
  domain_id** bsp0;
  domain_id** bsp1;
  //transpose
  domain_id* bsp0T;
  domain_id* bsp1T;

  
  // Time step
  // FIXME: Consider move to some simulation struct...
  REAL dt;

  // Flag to keep track of when the last opened discrete boundary closed
  REAL last_opened_discrete_boundary;

  // Time step dependent on species
  // FIXME: Consider move to some simulation struct...
  REAL* species_sigma_dt;
  domain_id* species_substeps;
  unsigned int stochastic_substep;
  unsigned int reaction_substep;

  // The number of save species
  domain_id num_save_species, all_data_num_species;

  // Indicies to species to save
  domain_id* ind_save_species;
  domain_id* all_data_ind_species;

  // The number of sheets saved for each species in x,y,z directions
  domain_id sheets_per_species[3];
  
  // The indices for each sheet that is going to be saved in every dimension
  // One can use it for instance by typing indices[X] to get indices in
  // x direction
  int* indices[3];
  int* global_indices[3];

  // The coordinates of the sheets that are going to be saved
  REAL* coords[3];

  // Boundary information
  unsigned int* num_boundary_voxels;
  unsigned int** boundary_voxels;
  
  // Flag indicating to force Dirichlet boundary conditions
  int force_dirichlet;
  
  // Species index and species name offsets coordinates and offsets indices
  // used to calculate the convolution.
  LinescanData* linescan_data;
  // Convolution constants
  REAL conv[NDIMS]; 
};

typedef struct Species Species_t;

// Construct a Species from a geometry and an h5 model file
Species_t* Species_construct(Geometry_t* geom, arguments_t* arguments);

// Output species information
void Species_output_init_values(Species_t* species);

void Species_init_du(Species_t* species);

// Destroy a Species
void Species_destruct(Species_t* species);

// Apply initial values to species
void Species_init_values(Species_t* species);

void Species_init_du(Species_t* species);

// Init voxel index informations for all boundaries
void Species_init_boundary_voxels(Species_t* species);

// Init memory and indices for voxels for fixed domain species
void Species_init_fixed_domain_species(Species_t* species, domain_id num_fixed_domains, 
                                       domain_id* fixed_domains);

// Switch species between u0 and u1
void Species_switch_u0_u1(Species_t* species, size_t time_ind);

// Apply increment
void Species_apply_du(Species_t* species);

void Species_apply_fixed_values(Species_t* species);

// Step diffusion pde
void Species_step_diffusion(Species_t* species, size_t time_ind);

// Step reaction pde
void Species_step_reaction(Species_t* species, size_t time_ind);

// Evaluate stochastic events
void Species_evaluate_stochastic_events(Species_t* species, size_t time_ind);

// Step prescribed fluxes over boundaries
void Species_step_boundary_fluxes(Species_t* species, size_t time_ind);

// Communicate species values for discrete boundaries
void Species_communicate_values_at_discrete_boundaries(Species_t* species);

// Communicate openness of discrete boundaries to all processes
// FIXME: Should be placed in geometry.h but now we need Species_t
void Species_communicate_openness_of_discrete_boundaries(Species_t* species);

// Compute species convolution with 3-D Gaussian function on each process
// This means that only partial sums are used. Later processes need to
// communicate to sum it up and store.
void Species_compute_convolution_with_gaussian_function_locally(Species_t* species);

// Write computed convolution to file. This
void Species_write_convolution_to_file(Species_t* species, hid_t file_id, char* groupname);

// Write 2D data to file
void Species_write_2D_sheet_to_file(Species_t* species, hid_t file_id, char* groupname);

//Write all data to file
void Species_write_all_data_to_file(Species_t* species, hid_t file_id, char* groupname);

// Output scalar values for each species and domain
void Species_output_scalar_data(Species_t* species, hid_t file_id, char* groupname, 
				REAL t, unsigned long time_ind, unsigned long output_ind, 
                                domain_id silent);

// Output function
void Species_output_data(Species_t* species, unsigned int output_ind, REAL t, 
			 size_t time_ind, arguments_t* arguments);

// Check if there are any open discrete channels
domain_id Species_check_open_discrete_boundaries(Species_t* species, REAL t, 
                                                 arguments_t* arguments);

// Output time steping information
void Species_output_time_stepping_information(Species_t* species, REAL tstop, 
					      size_t save_interval);

// Return species index for given species name
domain_id Species_get_species_id(Species_t* species, char* species_name);

// Return diffusive species index 
domain_id Species_get_diffusive_species_id(Species_t* species, char* species_name);

#endif
