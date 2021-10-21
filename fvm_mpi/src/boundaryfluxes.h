#ifndef __BOUNDARYFLUXES_H
#define __BOUNDARYFLUXES_H

#include <mpi.h>

#include "types.h"
#include "basefluxinfo.h"

// Struct keeping parameters for fluxes
struct SERCA_model {

  REAL v_cyt_to_A_sr;
  REAL density;
  REAL scale;

  // The name of the boundary this flux exists on
  char boundary_name[MAX_SPECIES_NAME];
  
  // The name of the species this flux is applied to
  char species_name[MAX_SPECIES_NAME];

  // Parameters
  REAL params[3];
  
};

struct RyR_model {
  // State values
  ModelStates states;

  // Parameters
  REAL Kd_open;
  REAL k_min_open;
  REAL k_max_open;
  REAL n_open;

  REAL Kd_close;
  REAL k_min_close;
  REAL k_max_close;
  REAL n_close;

  REAL i_unitary;

  REAL t_close_ryrs;

  // The name of the boundary this flux exists on
  char boundary_name[MAX_SPECIES_NAME];
  
  // The name of the species this flux is applied to
  char species_name[MAX_SPECIES_NAME];

  // Parameters
  REAL params[10];
};

typedef struct RyR_model RyR_model_t;
typedef struct SERCA_model SERCA_model_t;

// Boundary flux struct
// FIXME: Everything in boundary fluxes are pretty hardcoded to serca and RyR 
// FXIME: fluxes. It would be nice to make this more flexible. With for example 
// FIXME: code generation...
struct BoundaryFluxes {
  
  // Flags for using ryr and serca in the model
  domain_id use_ryr, use_p_ryr, use_serca;
  
  // The RyR model
  RyR_model_t ryr;

    // The pRyR model
  RyR_model_t p_ryr;

  // The SERCA model
  SERCA_model_t serca;
  
  // The number of used boundary fluxes, i.e. the sum of all 
  // this->use_{model name}
  domain_id num_of_used_fluxes;
  
  // An array mapping boundary index into Flux_t structure pointer.
  // The length of that map is equal to the number of used fluxes.
  // Obviously, its length can be less then the number of models above.
  Flux_t** map;
  
  // An array of all boundary indices on which fluxes are applied
  unsigned int* boundary_map;
};

typedef struct BoundaryFluxes BoundaryFluxes_t;

// Flux accomodated by the serca pump
REAL serca_flux(REAL dt, REAL h, REAL u0, REAL u1, REAL* params);

// Flux through a single open channel
REAL ryr_flux(REAL dt, REAL h, REAL u0, REAL u1, REAL* params);

// Initialize a RyR_model
void init_RyR_model(RyR_model_t* ryr);

// Initialize a SERCA_model
void init_SERCA_model(SERCA_model_t* serca);

// Evaluate the RyR states stochastically
void evaluate_RyR_stochastically(REAL* params, ModelStates* states, 
                            REAL t, REAL dt, REAL* species_at_boundary);

// Initialize RyR states
void BoundaryFluxes_init_stochastic_boundaries(Species_t* species, arguments_t* arguments);

// Initialize RyR states using initial Ca concentration
void init_RyR_model_states_stochastically(RyR_model_t* ryr, REAL* species_at_boundary);

// Initialize RyR states using command line arguments
void init_RyR_model_states_deterministically(RyR_model_t* ryr, 
					     unsigned int number_init_ryrs,
					     int* open_ryrs);

// Construct a BoundaryFlux struct
BoundaryFluxes_t* BoundaryFluxes_construct(Species_t* species, hid_t file_id, 
                                           arguments_t* arguments);

// Output initial data of BoundaryFluxes
void BoundaryFluxes_output_init_data(Species_t* species, arguments_t* arguments);

// Destroy BoundaryFlux struct
void BoundaryFluxes_destruct(BoundaryFluxes_t* fluxes);

// Init voxels for each flux // unsued ??
void BoundaryFluxes_init_flux_voxels(BoundaryFluxes_t* fluxes, Geometry_t* geom, 
				     domain_id flux_id, domain_id boundary_id);

// Map provided boundary index into the flux information structure
Flux_t* BoundaryFluxes_get_flux_info(BoundaryFluxes_t* fluxes, domain_id boundary_ind);

// Return 1 if the i-th flux is open. Otherwise 0.
domain_id open_states_RyR(const ModelStates* states, int i);

#endif
