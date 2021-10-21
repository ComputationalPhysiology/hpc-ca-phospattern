#ifndef __OPTIONS_H
#define __OPTIONS_H

#include <stdlib.h>
#include <argp.h>
#include <stdint.h>
#include <mpi.h>

#include "types.h"

// Program documentation
const char* argp_program_version;
const char* argp_program_bug_address;
static char args_doc[];


// Discrete boundary open states info structure provided to collect all
// information on open states from command line arguments and use them
// in BoundaryFluxes_init_stochastic_boundaries function.
typedef struct OS_info{
  char* name; // boundary name
  int number_open_states;
  int max_ind;
  int* open_states;
} OS_info;

// Structure to store pairs: boundary name and time for certain action
// like update or close. This holds only for stochastic currents.
typedef struct ActionTime_
{
  char* name; //boundary name
  REAL time;
} ActionTime;

// Structure to store information used in calculating species convoluted
// with 3-D Gaussian function.
typedef struct LineScan_
{
  int axis;
  char* species;
  char** domains;
  domain_id num_domains;
  int offsets[3];
} LineScan;

// Used by main to communicate with parse_opt
struct arguments_
{
  // Simulation time
  double tstop;

  // Geometry resolution
  double h;

  // Output options
  int silent, verbose, force_dirichlet;

  // Output time
  REAL dt_save;

  // Random seed
  unsigned char seed_given;
  uint32_t seed;

  // Process splitting direction
  domain_id split_plane;

  // In deterministic runs close times can be forced
  ActionTime *t_close;
  int num_t_close;

  // Update interval for reactions
  REAL dt_update_react;
  // Update interval for stochastics
  REAL dt_update_stoch;

  // Time step
  REAL dt;
  
  // Abort simulations after X ms all channels of provided discrete boundary
  // are closed.
  double tabort;

  // List of species that should be saved
  char** species;
  int num_save_species;
  
  // List of species for which all data is save. Could be different then above ones
  char** all_data_species;
  int all_data_num_species;

  // List of x,y,z center points which should be used to save species in sheets
  int num_ax_points[3];
  REAL* ax_points[3];

  // File name options
  char *geometry_file;
  char *model_file;

  // A casename
  char *casename;
  
  // At that moment provide two ways of defining open states.
  // Later, erease the previous method.
  OS_info* states_info;
  int states_info_length;
  
  // Store linescan arguments
  LineScan* linescan;
  
  // Vector of processor splitting in XYZ directions.
  unsigned int* mpi_dist;
  unsigned int mpi_prod;
};


// Typedef for the argument
typedef struct arguments_ arguments_t;

// Parse a single option
error_t parse_opt (int key, char *arg, struct argp_state *state);

// Output parsed arguments
void arguments_output(MPI_Comm comm, arguments_t* arguments);

// Construct and initialize an arguments struct
arguments_t* arguments_construct(MPI_Comm comm);

// Destruct arguments struct
void arguments_destruct(arguments_t* arguments);

// Return an info structure for given flux name
const OS_info* arguments_get_open_states_info(const arguments_t* arguments,
                                               const char* boundary_name);

// Return closing time for given flux name
REAL arguments_get_t_close(const arguments_t* arguments, const char* boundary_name);
                                               
#endif
