#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#include "mtwist.h"
#include "geometry.h"
#include "species.h"
#include "boundaryfluxes.h"
#include "options.h"
#include "utils.h"

#define OUTPUT

//-----------------------------------------------------------------------------
static char args_doc[] =
  "calcium_spark -- A finite volume program modeling a calcium spark simulation";
//-----------------------------------------------------------------------------
static struct argp_option options[] = {
  {"casename",        'c', "CASENAME", 0,  "Sets the name of the run to CASENAME. Default (\"casename\")" },
  {"dt",              'd', "TIME", 0,  "Forced minimal timestep. If negative time step will be deducted from diffusion constants. Default(-1.)" },
  {"force_dirichlet", 'f', 0, 0, "Force to use Dirichlet boundary condition."},
  {"geometry_file",   'g', "FILE", 0,  "Read the geometry from FILE. Exprects a .h5 file." },
  {"resolution",      'h', "RES",  0,  "Run simulation with RES nm as the spatial resolution. Default(6.)" },
  {"species_file",    'm', "FILE", 0,  "Read the species model from FILE. Expects a .h5 file." },
  {"dt_save",         'o', "TIME", 0,  "Save solution for each dt_save. Default(.5)" },
  {"split_processes", 'p', "PLANE", 0, "Split processes in given directions {XY, YZ, XZ}. Default(XY)."},
  {"quiet",           'q', 0,      0,  "Don't produce any output" },
  {"save_species",    's', "SPECIES", 0,  "A list of species that should be saved in 2D sheets at given z-coordinates." },
  {"tstop",           't', "TIME", 0,  "Run simulation for TIME number of ms. Default(10.)" },
  {"verbose",         'v', 0,      0,  "Produce verbose output" },
  {"save_x_coords",   'x', "COORD", 0,  "X coordinates for the 2D sheet that will be saved." },
  {"save_y_coords",   'y', "COORD", 0,  "Y coordinates for the 2D sheet that will be saved." },
  {"save_z_coords",   'z', "COORD", 0,  "Z coordinates for the 2D sheet that will be saved." },
  {"linescan",        'l', "DIR_SPCECIES_COORD", 0, "Computes convolution of the species with 3-D Gaussian function."},
  {"t_close",         'C', "TIME", 0,  "Time for closing currents. If negative, they are ""closed stochastically. Default (-1)"},
  {"dt_update_stoch", 'D', "TIME", 0,  "Time step for updating the stochastics. If negative it will be the same as global timestep. Default (-1)"},
  {"open",            'O', "OPEN_STATES", 0, "Initialize given discrete fluxes to open. Otherwise they are randomly initialized."},
  {"seed",            'S', "SEED", 0,  "Provide a seed for the random generator. If not given the seed will be random each run." },
  {"dt_update_react", 'T', "TIME", 0,  "Time step for updating the reactions. If negative it will be the same as global timestep. Default (-1)"},
  {"abort",           'X', "TYPE TIME", 0, "Abort simulations after TIME ms after the last RyR channel has been closed. By default it is inactive."},
  {"all_data",        777, "SPECIES", 0, "Save species values of all points (3D)"},
  {"mpi_dist",        777, "MPI_DIST", 0, "An mpi distribution for processor splitting in XYZ directions. It overwrites the -p option. By default it is switched off."},
  { 0 }
};
//-----------------------------------------------------------------------------
static struct argp argp = { options, parse_opt, 0, args_doc };
//-----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  push_time(SIM);
  push_time(SETUP);
  size_t time_ind, save_interval;
  REAL t, dt;
  unsigned int output_ind;

  // Init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // Check number of processes (could be made more flexible in the future)
  int size;
  MPI_Comm_size(comm, &size);

  // Init the arguments with default values
  arguments_t* arguments = arguments_construct(comm);

  // Parse arguments
  argp_parse(&argp, argc, argv, 0, 0, arguments);

  // Check the number of processes. If the -p option was used, then the
  // number of processes should be either a quadratic or cubic number
  // depending on the -p option argument.
  // On the other hand, if the --mpi_dist option was used, then the number
  // of processes should be equal to the product of the numbers following
  // the --mpi_dist option.
  if (arguments->mpi_dist)
  {
    if (size!=arguments->mpi_prod)
      mpi_printf_error(comm, "*** Error: --mpi_dist option was used. Expected "\
                    "the number of processes to be equal to %d. Instead got %d.\n",
                    arguments->mpi_prod, size);
  }
  else // --mpi_dist was not provided
  {
    if (arguments->split_plane == XYZ) // -p XYZ
    {
      if (fmod(pow((double)size, 1/3.), 1.0) != 0.)
        mpi_printf_error(comm, "*** Error: The option -p XYZ was used. Thus "
                    "expected a cubic number for the number of processes: "\
                    "0, 1, 8, 27, 64 aso, got %d.\n", size);
    }
    else // -p XY (YZ, XZ) used
      if (fmod(sqrt((double)size), 1.0) != 0.)
        mpi_printf_error(comm, "***Error: The option -p XY or YZ or XZ was "\
                    "used. Expected a quadratic number for the number of "\
                     "processes: 0, 1, 4, 9, 16, 25 aso, got %d.\n", size);
  }


  if (!arguments->geometry_file)
    mpi_printf_error(comm, "*** ERROR: Expected a geometry file passed as argument:"\
                     " --geometry_file or -g\n");

  if (!arguments->model_file)
    mpi_printf_error(comm, "*** ERROR: Expected a species file passed as argument: "\
                     "--species_file or -m\n");

  // Force silent off if verbose is on
  arguments->silent = arguments->verbose==0 ? arguments->silent : 0;

  // Check for given seed
  if (arguments->seed_given)
    mt_seed32new(arguments->seed);
  else
    mt_seed();

  // Output arguments
  arguments_output(comm, arguments);
  pop_time(SETUP);
  //FIXME: remove barrier when not profiling
  //MPI_Barrier(comm);

  // Read in the geometry
  push_time(GEOM);
  Geometry_t* geom = Geometry_construct(comm, arguments);
  pop_time(GEOM);
  push_time(SETUP);

  // Check given open state indices
  unsigned int os_info_id;
  int num_boundaries;

  for (os_info_id=0; os_info_id<arguments->states_info_length; os_info_id++)
  {
    // Get boundary size for appropriate boundary name
    num_boundaries = geom->boundary_size[Geometry_get_boundary_id(geom, arguments->states_info[os_info_id].name)];

    // Check if given open state number from command line is less then
    // the boundary size for given boudary name
    if (arguments->states_info[os_info_id].max_ind >= num_boundaries)
      mpi_printf_error(comm, "*** ERROR: Expected max %s indices given by option "\
                      "--force_discrete_boundaries_open: %d to be lower than "\
                     "the number of %s boundaries: %d\n",
                     arguments->states_info[os_info_id].name,
                     arguments->states_info[os_info_id].max_ind,
                     arguments->states_info[os_info_id].name,
                     num_boundaries);
  }
  pop_time(SETUP);

  // Read in model file
  push_time(SPECIES);
  Species_t* species = Species_construct(geom, arguments);
  pop_time(SPECIES);

  push_time(INIT);
  // Apply initial values
  Species_init_values(species);
  pop_time(INIT);
  push_time(SETUP);
  // Init discrete boundaries
  BoundaryFluxes_init_stochastic_boundaries(species, arguments);

  // Output species info
  output_ind = 0;
  dt = species->dt;
  save_interval = max_int(1, floor(arguments->dt_save/dt));

  if(arguments->verbose)
  {
    Species_output_init_values(species);
    BoundaryFluxes_output_init_data(species, arguments);
    Species_output_time_stepping_information(species, arguments->tstop, save_interval);
  }
  pop_time(SETUP);

  Species_init_du(species);

  // Output init values to file
  #ifdef OUTPUT
  Species_compute_convolution_with_gaussian_function_locally(species);
  push_time(WRITE);
  Species_output_data(species, 0, 0.0, 0, arguments);
  pop_time(WRITE);
  #endif

  push_time(MAINLOOP);
  // Time loop
  for (t=0., time_ind=1; t<arguments->tstop; t+=dt, time_ind++)
  {
    //Species_init_du(species);
    // Do one diffusion step after communicating ghost values

    push_time(DIFFTOT);
    Species_step_diffusion(species, time_ind);
    pop_time(DIFFTOT);

    // Evaluate and broadcast all stochastic events
    Species_evaluate_stochastic_events(species, time_ind);

    // Step all boundary fluxes
    //this function is causing apply-du swap to fail
    Species_step_boundary_fluxes(species, time_ind);

    // Do one reaction step
    Species_step_reaction(species, time_ind);

    // Apply increment
    Species_apply_du(species);
    Species_apply_fixed_values(species);

    // If output

    if (time_ind % save_interval == 0) {
      #ifdef OUTPUT
      Species_compute_convolution_with_gaussian_function_locally(species);
      push_time(WRITE);
      Species_output_data(species, ++output_ind, t+dt, time_ind, arguments);     
      pop_time(WRITE);
      #endif
    }

    // Check if we should stop simulation because no discrete boundaries are open
    
    if (Species_check_open_discrete_boundaries(species, t+dt, arguments))
      break;
  }
  pop_time(MAINLOOP);
  // Cleanup
  Species_destruct(species);
  Geometry_destruct(geom);
  arguments_destruct(arguments);

  // Output timings
  pop_time(SIM);
  if(arguments->verbose)
  {
    output_timings(comm);
  }
  // Finalize MPI
  MPI_Finalize();

  return 0;
}