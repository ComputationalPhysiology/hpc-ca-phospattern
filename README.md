# hpc-ca-phospattern
Code and geometries for running simulations. The model is described in "Nanoscale organization of ryanodine receptor distribution and phosphorylation pattern determines the dynamics of calcium sparks".
Calcium_spark -- A finite volume program modelling a calcium spark simulation
Please note that in the code G6 refers to geometry G4 as described in the paper and G8 refers to geometry G5 as described in the paper.
Additionally inner 10, refers to inner 20% and inner 25, to inner 50%. 

COMPILATION:
------------
Go to the folder fvm_mpi/ and run 'make -j ARCH=3'. HDF5 modules are needed to compile the code. 
 
The script will compile and place the software in fvm_mpi/bin/calcium_sparks.


USAGE:
------
Basic run instruction is based on providing all necessary information
using possible command line options

    in serial:   calcium_sparks [OPTION...]
    in parallel: mpirun -np N calcium_sparks [OPTION...]
    
There are two arguments that need to be provided:
  
  --geometry_file=G1_nonP_0.h5
  --species_file=parameters_sr1100_ryr55.h5
  
These .h5 files describes the entire problem, and can be changed or re-generated
by different python scripts provided in "tests" folder.


Other optional arguments related to the simulation are the following ones:
  -h, --resolution=RES              Spatial simulation resolution with RES nm. By default is set to 6.0 nm. This value can be 
                                    anything. However it will be adjusted to the resolution of the geometry. For example: if the 
                                    geometry resolution is 10 nm, then a given simulation resolution: -h X, will be adjusted so 
                                    that h_geom % h_adjust = 0. So again assuming the resolution of the geometry is 10 nm, we 
                                    will have the following simulation resolution:
                                        -h 12  -> simulation resolution = 10
                                        -h 3   -> simulation resolution = 3.333
                                        -h 2.5 -> simulation resolution = 2.5

  -c, --casename=CASENAME           Sets the name of the run to CASENAME. Default ("casename")

  -d, --dt=TIME                     Forces a minimal time step in ms. If negative, the time step will be deduced from diffusion 
                                    constants. By default -1 is set.
                  
  -t, --tstop=TIME                  Controls the end time of the simulations in ms. By default is set to 10.0.
  
  -X, --abort=TYPE TIME             Abort simulations TIME ms after all discrete channels of provided TYPE are closed. For negative
                                    integers, abort simulations as soon as all channels are closed, provided that fewer than abs(TIME)
                                    channels were initially open (with -O option). By default the option is inactive, and the
                                    simulations end as soon as the final time is reached. Example:
                                        -X ryr 1.5
                                        -X ryr -4

  -C, --t_close=TIME                Time for closing currents. If negative, they are closed stochastically. Default (-1)

  -T, --dt_update_react=TIME        Evaluating reaction parts of the simulation each time step may be expensive and not necessary.
                                    To overwrite this behaviour provide a coarser time step. If negative, then it will be the same 
                                    as the global time step. On the other hand, if less than the provided time step, the larger one 
                                    will be used. By default is set to -1.0.

  -D, --dt_update_stoch=TIME        Evaluation of the stochastic Markov model for the RyRs each time step can be expensive (at least 
                                    the scatter and gather communication for the parallel run can be). To overwrite this behaviour
                                    provide a coarser time step. If negative, then it will be the same as the global time step. On the 
                                    other, hand if less than the provided time step, the larger one is used. By default is set to -1.0.
  
  -O, --open=open_states            Force discrete fluxes to be open at start. Define whether a non phosphorylated (RyR) or a phosphorylated RyR (pRyR)          
                                    should be open ar start. Example: to have 5 non phosphotylated RyRs and 1 phosphorylated RyR to be open at start
                                          -O ryr 0 1 2 3 4 p_ryr 3. 
                                    In general, however, there can be more than one type of fluxes needed to be initialised to be open
                                    at start:
                                          -O name_1 num11 num12 num13 ... name_2 num21 num22 num23 ... ..., 
                                    where name_N is the flux name (e.g. ryr, p_ryr, lcc) and num_NM is the flux number. If not provided, they 
                                    are initialised randomly.

  -S, --seed=SEED                   Provide a seed for the random generator. If not given the seed will be random each run.
                            
  
MPI related arguments:
  -p, --split_processes=PLANE       Split processes in one of the possible directions XY, YZ or XZ. If one of these three options is used,
                                    the number of processes must be a quadratic number. Splitting processes in all three XYZ directions
                                    is also possible. Make sure that in this case the number of processes is be a cubic number. By default
                                    direction XY is set.

      --mpi_dist=mpi_distribution   An mpi distribution for processor splitting in all three XYZ directions with different number of processes. 
                                    It overwrites the -p option and by default is switched off. For example providing --mpi_dist 4 5 2
                                    forces to use a total number of processes equal to 40 = 4*5*2, and thus this number must be provided
                                    with -np option to mpirun.

Saving data
  -o, --dt_save=TIME                Save solution for each dt_save. By default is 0.5 ms.

  -s, --save_species=SPECIES        Provide a list of species that should be saved in 2D sheets i.e. only a slice of the provided species
                                    will be saved. 
  
  -x, --save_x_coords=COORD         X coordinates in nm for the 2D sheet that will be saved. For instance, for a cubic mesh 1um x 1um x 1um,
                                    the COORD should be a positive number less then 1000.0 nm. The value will be rounded to fit the geometry
                                    resolution. Example: for COORD = 26.7 and geometry's resolution 12 nm, the COORD will re rounded to 24nm.
                                    The option should be provided along with --save_species.

  -y, --save_y_coords=COORD         Y coordinates for the 2D sheet that will be saved. See above for further information.
                            
  -z, --save_z_coords=COORD         Z coordinates for the 2D sheet that will be saved. See above for further information.

      --all_data=SPECIES            Save species values of all points (3D). Be aware that this might be memory expensive because all data for
                                    provided species will be dumped to file each time step 'dt_save'.
                                    
  -l, --linescan=DIR_SPCECIES_COORD Along given axes (x,y or z) computes and stores the convolution of a provided species with 3-D Gaussian function. 
                                    The linescan can be controlled by: "-l x Ca offset offset". The first two arguments are the axis and the species
                                    for which the convolution is computed. The last two arguments represent the domain's offset in the last two directions,
                                    i.e. for x,y,z directions there are (y_offset, z_offset), (x_offset, z_offset), (x_offset, y_offset) respectively. 

                                    Mathematically the linescan option (--linescan x SPECIES Y_offset Z_offset ) can be expressed as:
                                    SPECIES_AVG(x, Y_offset, Z_offset) = \sum_x' \sum_y' \sum_z'  SPECIES(x',y',z') * G(x-x',Y_offset-y', Z_offset-z')
                                    
                                    with G(x,y,z) = g(x, \sigma_x)*g(y, \sigma_y)*g(z, \sigma_z), where g(w,s) =  (2\pi s)^(-1/2) exp(-w^2/2s).
