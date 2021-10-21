#include "options.h"
#include "utils.h"
#include "string.h"
#include "math.h"

//-----------------------------------------------------------------------------
const char *argp_program_version =
  "calcium_spark 0.1";
//-----------------------------------------------------------------------------
const char *argp_program_bug_address =
  "<hake@simula.no>";
//-----------------------------------------------------------------------------
error_t parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  int offset, i, index, num_args;
  unsigned int ax;
  arguments_t *arguments = state->input;
  REAL r_read_number;
  
  switch (key)
  {
  case 'c':
    arguments->casename = arg;
    break;
    
  case 'd':
    arguments->dt = atof(arg);
    break;
    
  case 'f':
    arguments->force_dirichlet = 1;
    break;

  case 'g':
    arguments->geometry_file = arg;
    break;
    
  case 'h':
    arguments->h = atof(arg);
    break;
    
  case 'm':
    arguments->model_file = arg;
    break;
    
  case 'o':
    arguments->dt_save = atof(arg);
    break;    
  
  case 'p':
    // Check the command -p argument
    if (strcmp(arg, "XY") == 0)
      arguments->split_plane = XY;
    else if (strcmp(arg, "YZ") == 0)
      arguments->split_plane = YZ;
    else if (strcmp(arg, "XZ") == 0)
      arguments->split_plane = XZ;
    else if (strcmp(arg, "XYZ") == 0)
      arguments->split_plane = XYZ;
    else
      mpi_printf_error(MPI_COMM_WORLD, "***Error: Expected argument for "\
                "option -p is one of {XY, XZ, YZ}. Instead got %s\n", arg);
    break;
    
  case 'q': 
    arguments->silent = 1;
    break;

  case 's':
    offset = 0;

    // Get where to start in argv
    while (state->argv[state->next+offset] != NULL &&
          state->argv[state->next+offset][0] != '-')
      offset++;

    // Get species from argv
    arguments->species = &state->argv[state->next-1];
    arguments->num_save_species = offset+1;
    
    // Update next counter
    state->next += offset;
    break;
    
  case 't':
    arguments->tstop = atof(arg);
    break;

  case 'v':
    arguments->verbose = 1;
    break;

  case 'x': case 'y': case 'z':
    ax = (key == 'x' ? X : (key == 'y' ? Y : Z));
    offset = 0;

    // Get where to start in argv
    while (state->argv[state->next+offset] != NULL &&
     state->argv[state->next+offset][0] != '-')
      offset++;

    arguments->num_ax_points[ax] = offset + 1;
    arguments->ax_points[ax] = mpi_malloc(MPI_COMM_WORLD, sizeof(REAL)*arguments->num_ax_points[ax]);
    
    for (i=0; i<arguments->num_ax_points[ax]; i++)
      arguments->ax_points[ax][i] = atof(state->argv[state->next-1+i]);

    // Update next counter
    state->next += offset;
    break;
    
  case 'l':
    offset = 0;
    // Calculate how many arguments have been provided with -l option
    while (state->argv[state->next+offset] != NULL && (state->argv[state->next+offset][0] != '-' || 
           (state->argv[state->next+offset][0] == '-' && (state->argv[state->next+offset][1] >= '0' ||  state->argv[state->next+offset][1] <= '9'))))
      offset++;    
    
    // Store number of arguments and check if at least 4 have been provided
    num_args = offset + 1;
    if (num_args < 4)
      mpi_printf_error(MPI_COMM_WORLD, "*** Error: Incorrect number of "\
            "provided arguments for --linescan option. Expects at least 4, but"
            "instead found %d.\n", num_args);
    
    // A helper pointer that points to the first argument provided
    // with --linescan option
    char** args = &state->argv[state->next-1];
      
    // First argument should be x,y or z
    if (strlen(args[0]) != 1 || (args[0][0] != 'x' && args[0][0] != 'y' && args[0][0] != 'z'))
      mpi_printf_error(MPI_COMM_WORLD, "*** Error: Expects axis x,y or z as "\
            "the first argument passed to --linescan (-l). Instead got: %s\n", args[0]);
    
    arguments->linescan = mpi_malloc(MPI_COMM_WORLD, sizeof(LineScan));
    arguments->linescan->domains = NULL;
    arguments->linescan->axis = args[0][0] - 'x';
    arguments->linescan->species = args[1];
    
    // Allocate memory for potentially provided subdomains and store their names
    arguments->linescan->num_domains = num_args - 4;
    if (arguments->linescan->num_domains)
      arguments->linescan->domains = mpi_malloc(MPI_COMM_WORLD, sizeof(char*)*arguments->linescan->num_domains);
    
    for(i=0; i<arguments->linescan->num_domains; ++i)
      arguments->linescan->domains[i] = args[2+i];

    // Get offsets. First set maximum values at every direction, and then
    // overwrite some of them with appropriate values
    const int dir = arguments->linescan->axis;
    
    arguments->linescan->offsets[mod(dir+(int)pow(-1, dir), 3)] = atoi(args[2+arguments->linescan->num_domains]);
    arguments->linescan->offsets[mod(dir+(int)pow(-1, dir)*2, 3)] = atoi(args[3+arguments->linescan->num_domains]);
    
    // Update next counter
    state->next += offset;
    break;

  case 'C':
    offset = 0, num_args = 1;
    
    // First loop is to detect for how many fluxes we define close times
    while (state->argv[state->next+offset] != NULL &&
      state->argv[state->next+offset][0] != '-')
        offset++;
    num_args += offset;
    
    // Check that the user provided a even number of arguments.
    if (num_args < 2 && num_args%2 != 0)
      mpi_printf_error(MPI_COMM_WORLD, "*** Error: Incorrect number of "\
            "provided arguments for --t_close option. Expects an even "
            "positive number. Instead found %d provided arguments.\n", num_args);
            
    // Number of provided fluxes
    arguments->num_t_close = num_args/2;
    // Allocate enough memory for flux open states information
    arguments->t_close = mpi_malloc(MPI_COMM_WORLD, sizeof(ActionTime)*arguments->num_t_close);
    
    // Loop over all command line arguments once again and retrieve
    // information on open states
    for(index=0; index<=offset; index++)
    {
      // For even index expect a string
      if (index%2 == 0) 
      {
        if (!(atoi(&state->argv[state->next-1][0]) == 0 &&
            state->argv[state->next-1][0] != '0'))
              mpi_printf_error(MPI_COMM_WORLD, "Error: Expected a string "\
                "on the  %d arguments for --t_close option. Instead got: "\
                "%s.\n", index+1, state->argv[state->next+index-1]);

        // Memorize a boundary name
        arguments->t_close[index/2].name = state->argv[state->next+index-1];
      }
      // For odd index expect a real number
      else if (index%2 == 1)
      {
        r_read_number = atof(state->argv[state->next+index-1]);
        r_read_number += strcmp(state->argv[state->next+index-1], "0") == 0;
        r_read_number += strcmp(state->argv[state->next+index-1], "0.0") == 0;
        
        if (!r_read_number)
          mpi_printf_error(MPI_COMM_WORLD, "Error: Expected a REAL number "\
            "on the %d arguments for --t_close option. Instead got: %s.\n",
            index+1, state->argv[state->next+index-1]);

        arguments->t_close[index/2].time = atof(state->argv[state->next+index-1]);
      }
    }
    
    state->next += offset;
    break;
  
  case 'D':
    arguments->dt_update_stoch = atof(arg);
    break;


  case 'O':
    offset = 0;
    int current_flux_id = -1, num_states=0, read_number, id;
    
    // Check first if the first argument on the arguments list is
    // a string
    if (!(atoi(&state->argv[state->next-1][0]) == 0 &&
        state->argv[state->next-1][0] != '0'))
        mpi_printf_error(MPI_COMM_WORLD, "*** Error: Expected a string as "\
          "the first argument for --open option. Instead got: %s.\n", 
          state->argv[state->next-1]);
    
    // First loop is to detect for how many fluxes we define open states
    while (state->argv[state->next+offset] != NULL &&
      state->argv[state->next+offset][0] != '-')
    {
      if (atoi(&state->argv[state->next+offset][0]) == 0 &&
        state->argv[state->next+offset][0] != '0')
          arguments->states_info_length++;
      offset++;
    }
    arguments->states_info_length++;
    
    // Check if the last argument on the argument list is a number
    if (atoi(state->argv[state->next+offset-1]) == 0 &&
        strcmp(state->argv[state->next+offset-1], "0") != 0)
        mpi_printf_error(MPI_COMM_WORLD, "*** Error: Expected an integer as "\
          "the last argument for --force_discrete_boundaries_open option."\
          "Instead got: %s.\n", state->argv[state->next+offset-1]);
          
    // Allocate enough memory for flux open states information
    arguments->states_info = mpi_malloc(MPI_COMM_WORLD, sizeof(OS_info)*arguments->states_info_length);
    
    // Loop over all command line arguments once again and retrieve
    // information on open states
    for(index=0; index<=offset; index++)
    {
      // Check if we can read a number
      if (state->argv[state->next+index] == NULL)
        read_number = 0;
      else
      {
        read_number = atoi(state->argv[state->next+index]);
        read_number += strcmp(state->argv[state->next+index], "0") == 0;
      }
      
      if (read_number)
        num_states++;
      else
      {
        // Save information about open states for currently processed flux
        arguments->states_info[++current_flux_id].name = state->argv[state->next+index-num_states-1];
        arguments->states_info[current_flux_id].number_open_states = num_states;
        arguments->states_info[current_flux_id].open_states = mpi_malloc(MPI_COMM_WORLD, sizeof(int)*num_states);
        arguments->states_info[current_flux_id].max_ind = 0;
        for(id=0; id<num_states; id++)
        {
          // Read the number
          arguments->states_info[current_flux_id].open_states[id] = atoi(state->argv[state->next+index-num_states+id]);
          
          // Check if the number is a non-negative number
          if (arguments->states_info[current_flux_id].open_states[id] < 0)
            mpi_printf_error(MPI_COMM_WORLD, "***Error: Expected a positive number for"\
                       " the open indices given by --force_discrete_boundaries_open option. "\
                       "For %s got %d.\n", 
                       arguments->states_info[current_flux_id].name,
                       arguments->states_info[current_flux_id].open_states[id]);
    
          // Collect the maximal number
          arguments->states_info[current_flux_id].max_ind = \
                max_int(arguments->states_info[current_flux_id].max_ind, 
                arguments->states_info[current_flux_id].open_states[id]);
          
          // Check if the numbers are given in increasing oreder
          if(id>0 && arguments->states_info[current_flux_id].open_states[id]
              <= arguments->states_info[current_flux_id].open_states[id-1])
            mpi_printf_error(MPI_COMM_WORLD, "***Error: Expected a monotonicly "\
                "increasing numbers for the open states indices given by "\
                "--force_discrete_boundaries_open option. For %s got %d and %d.\n",
                arguments->states_info[current_flux_id].name,
                arguments->states_info[current_flux_id].open_states[id-1],
                arguments->states_info[current_flux_id].open_states[id]);
        }

        // Cancel out the number of states
        num_states = 0;
      }
    }

    state->next += offset;
    break;  
  
  case 'S':
    arguments->seed = atoi(arg);
    arguments->seed_given = 1;
    break;

  case 'T':
    arguments->dt_update_react = atof(arg);
    break;
        
  case 'X':
    arguments->tabort = atof(arg);
    break;
  
  case 777:
    // Use this case number for long-name options only. They do not have
    // an equivalent shortcut
    
    // If I have more then one long options, need to compare which one
    // in fact is being used
    if (strcmp(state->argv[state->next-2], "--all_data") == 0)
    {
      offset = 0;

      // Get where to start in argv
      while (state->argv[state->next+offset] != NULL &&
       state->argv[state->next+offset][0] != '-')
        offset++;

      // Get species from argv
      arguments->all_data_species = &state->argv[state->next-1];
      arguments->all_data_num_species = offset+1;
      
      // Update next counter
      state->next += offset;
    }
    else if (strcmp(state->argv[state->next-2], "--mpi_dist") == 0)
    {
      // Alocate memory for mpi distribution numbers
      arguments->mpi_dist = mpi_malloc(MPI_COMM_WORLD, sizeof(unsigned int)*3);
      arguments->mpi_prod = 1;
      
      // Iterate over arguments and translate them into integer.
      // Check whether the obtained number is strictly bigger than one,
      // and whether we detected 3 numbers.
      int offset;
      for (offset=0; offset<3; offset++)
      {
        if (state->argv[state->next-1+offset] == NULL ||
                  state->argv[state->next-1+offset][0] == '-')
          mpi_printf_error(MPI_COMM_WORLD, "*** Error: Excpected three positive "\
              "integers after the --mpi_dist option. Found only %d\n", offset);
        else
        {
          arguments->mpi_dist[offset] = atoi(state->argv[state->next-1+offset]);
          if (arguments->mpi_dist[offset] < 1)
            mpi_printf_error(MPI_COMM_WORLD, "*** Error: Encountered a wrong number "\
                "at position %d for the --mpi_dist option. Expected an integer > 0.\n",
                offset+1);
          arguments->mpi_prod *= arguments->mpi_dist[offset];
        }
      } 
      
      // Update the next counter
      state->next += 2;
    }
    break;
    
  case ARGP_KEY_ARG:
    
    if (state->arg_num >= 0)
      /* Too many arguments. */
      argp_usage (state);
    break;
     
  default:
    return ARGP_ERR_UNKNOWN;
  
  }
  
  return 0;
}
//-----------------------------------------------------------------------------
void arguments_output(MPI_Comm comm, arguments_t* arguments)
{
  
  if (!arguments->silent)
  {
    const char coord[3] = {'x', 'y', 'z'};
    unsigned int i,j;
    
    mpi_printf0(comm, "\n");
    mpi_printf0(comm, "Arguments:\n");
    mpi_printf0(comm, "-----------------------------------------------------------------------------\n");
    mpi_printf0(comm, "  casename:          \"%s\"\n", arguments->casename);
    mpi_printf0(comm, "  geometry_file:     \"%s\"\n", arguments->geometry_file);
    mpi_printf0(comm, "  species_file:      \"%s\"\n", arguments->model_file);
    mpi_printf0(comm, "  tstop:             %.6f ms\n", arguments->tstop);
    mpi_printf0(comm, "  h:                 %.1f nm\n", arguments->h);
    mpi_printf0(comm, "  dt:                %.3g\n", arguments->dt);
    mpi_printf0(comm, "  dt_save:           %.3g\n", arguments->dt_save);
    mpi_printf0(comm, "  dt_update_stoch:   %.3g\n", arguments->dt_update_stoch);
    mpi_printf0(comm, "  dt_update_react:   %.3g\n", arguments->dt_update_react);
    mpi_printf0(comm, "  t_close:           %s", arguments->num_t_close ? "" : "-");
    
    for (i=0; i<arguments->num_t_close; i++)
    {
      mpi_printf0(comm, "%s %.3g", arguments->t_close[i].name, arguments->t_close[i].time);
      if (i < arguments->num_t_close-1)
        mpi_printf0(comm, ", ");
    }
    mpi_printf0(comm, "\n");
    
    mpi_printf0(comm, "  linescan:          %s", arguments->linescan ? "" : "-\n");
    if (arguments->linescan)
    {
      const int dir = arguments->linescan->axis; 
      const int offset_one = mod(dir+ (int)pow(-1, dir), 3), offset_two = mod(dir+ (int)pow(-1, dir)*2, 3);
      mpi_printf0(comm, "%c, %s, %d, %d\n", coord[arguments->linescan->axis], arguments->linescan->species, 
                                            arguments->linescan->offsets[offset_one], arguments->linescan->offsets[offset_two]);
     }
    
    mpi_printf0(comm, "  abort simulations:%s", arguments->tabort ? "" : "-\n");
    if (arguments->tabort){
      mpi_printf0(comm, "%.3g\n", arguments->tabort);
    }

    mpi_printf0(comm, "  save_2D_species:   %s", arguments->num_save_species ? "" : "-");
    for (i=0; i<arguments->num_save_species; i++)
    {
      mpi_printf0(comm, "%s", arguments->species[i]);
      if (i<arguments->num_save_species-1)
        mpi_printf0(comm, ", ");
    }
    mpi_printf0(comm, "\n");
    if (arguments->num_save_species)
    {
      // Check if no coord was given, neither z, nor y, nor z
      // If so, use by default z coord with default value
      if(arguments->num_ax_points[0] + arguments->num_ax_points[1] + arguments->num_ax_points[2] == 0)
         mpi_printf0(comm, "   z-coords (nm):   center\n");
      else
      {
         // Print x,y,z coordinates
         unsigned int ax;
         for(ax=0; ax<3; ax++)
           if(arguments->num_ax_points[ax])
           {
              mpi_printf0(comm, "    %c-coords (nm):   ", coord[ax]);
              for (i=0; i<arguments->num_ax_points[ax]; i++)
              {
                mpi_printf0(comm, "%.2f", arguments->ax_points[ax][i]);
                if (i<arguments->num_ax_points[ax]-1)
                  mpi_printf0(comm, ", ");
              }
              mpi_printf0(comm, "\n");
           }
      }
    }
    mpi_printf0(comm, "  save_all_data:     %s", arguments->all_data_num_species ? "" : "-");
    for (i=0; i<arguments->all_data_num_species; i++)
    {
      mpi_printf0(comm, "%s", arguments->all_data_species[i]);
      if (i<arguments->all_data_num_species-1)
        mpi_printf0(comm, ", ");
    }
    mpi_printf0(comm, "\n");
    
    mpi_printf0(comm, "  open states:       %s", arguments->states_info_length ? "" : "-");
    for(i=0; i<arguments->states_info_length; i++)
    {
      if (i>0)
        mpi_printf0(comm, "                   ");
      
      mpi_printf0(comm, "%s: ", arguments->states_info[i].name);
      for(j=0; j<arguments->states_info[i].number_open_states; j++)
      {
        mpi_printf0(comm, "%d", arguments->states_info[i].open_states[j]);
        if (j<arguments->states_info[i].number_open_states-1)
          mpi_printf0(comm, ", ");
      }
      mpi_printf0(comm, "\n");
    }
     
    mpi_printf0(comm, "\n\n");
    MPI_Barrier(comm);
  }
  
}
//-----------------------------------------------------------------------------
arguments_t* arguments_construct(MPI_Comm comm)
{
  // Our arguments with default values
  arguments_t* arguments = mpi_malloc(comm, sizeof(arguments_t));
  arguments->geometry_file = NULL;
  arguments->model_file = NULL;
  arguments->casename = "casename";
  arguments->silent = 0;
  arguments->verbose = 0;
  arguments->force_dirichlet = 0;
  arguments->tstop = 10.;
  arguments->h = 6.;
  arguments->dt_save = .5;
  arguments->dt = -1.;
  arguments->num_save_species = 0;
  arguments->all_data_num_species = 0;
  unsigned int ax;
  for(ax=0; ax<3; ax++)
  {
     arguments->num_ax_points[ax] = 0;
     arguments->ax_points[ax] = NULL;
  }
  arguments->dt_update_react = -1.;
  arguments->dt_update_stoch = -1.;
  
  arguments->t_close = NULL;
  arguments->num_t_close = 0;

  arguments->seed_given = 0;
  arguments->seed = 0;
  arguments->split_plane = XY;
  
  arguments->states_info = NULL;
  arguments->states_info_length = 0;
  
  arguments->linescan = NULL;
  
  arguments->mpi_dist = NULL;
  
  return arguments;
}
//-----------------------------------------------------------------------------
void arguments_destruct(arguments_t* arguments)
{
//  free(arguments->open_ryrs);
//  int i;
//  for(i=0; i<arguments->num_save_species; i++)
//    free(arguments->species[i]);
//  free(arguments->species);
  unsigned int i;
  for(i=0; i<3; i++)
    if(arguments->ax_points[i])
      free(arguments->ax_points[i]);

//  free(arguments->geometry_file);
//  free(arguments->model_file);
//  free(arguments->casename);

  if (arguments->t_close)
    free(arguments->t_close);

  if (arguments->states_info)
  {
    for(i=0; i<arguments->states_info_length; i++)
      free(arguments->states_info[i].open_states);
    free(arguments->states_info);
  }
  
  if (arguments->linescan)
  {
    free(arguments->linescan->domains);
    free(arguments->linescan);
  }
  
  if (arguments->mpi_dist)
    free(arguments->mpi_dist);
      
  free(arguments);
}
//-----------------------------------------------------------------------------
const OS_info* arguments_get_open_states_info(const arguments_t* arguments,
                                               const char* boundary_name)
{
  unsigned int i;
  for (i=0; i<arguments->states_info_length; i++)
    if (strcmp(arguments->states_info[i].name,boundary_name) == 0)
      return &arguments->states_info[i];
      
  return NULL;
}
//-----------------------------------------------------------------------------
REAL arguments_get_t_close(const arguments_t* arguments, const char* boundary_name)
{
  unsigned int i;
  for (i=0; i<arguments->num_t_close; i++)
    if (strcmp(arguments->t_close[i].name, boundary_name) == 0)
      return arguments->t_close[i].time;
      
  return -1;
}
