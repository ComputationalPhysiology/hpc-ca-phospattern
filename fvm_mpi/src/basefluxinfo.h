#ifndef __FLUX_H
#define __FLUX_H

#include "types.h"

// More general structure holding states variables. Provided in order to
// have a full access to those variables in an easy way. Could be easily
// passed to functions.
typedef struct ModelStates{
  // Number of states
  unsigned int N;
  // Pointer of states with 0-1 values
  domain_id* s0;
}ModelStates;


// Helpful structure that holds pointers to states and to significant 
// functions
typedef struct StochasticReferences{
  // Pointer to states
  ModelStates* states;
  
  // Pointer to a function that evaluates model states
  void (* evaluate) (REAL* params, ModelStates* states, REAL t, REAL dt, REAL* species_at_boundary);
  
  // Pointer to a function that solves whether the state we are in is to
  // be considered as open or close
  domain_id (* open_states) (const ModelStates* states, int i);
}StochasticReferences;


// Additional structure that holds different fluxes information as pointers
// and is used in the BoundaryFluxes mapping
typedef struct Flux{
  // Pointer to Flux parameters
  REAL* flux_params;
  // Diffusive species index of an appropriate model
  unsigned int flux_dsi;
  
  // Pointer to function that evaluates flux value
  REAL (* flux_function) (REAL dt, REAL h, REAL u0, REAL u1, REAL* params);
  
  // Pointer to stochastic parts of a model if such provided. Otherwise,
  // the pointer is set to null
  StochasticReferences* storef;
}Flux_t;

#endif
