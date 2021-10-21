#include <assert.h>
#include <math.h>

#include "mtwist.h"
#include "utils.h"
#include "geometry.h"
#include "species.h"

#include "boundaryfluxes.h"

//-----------------------------------------------------------------------------
REAL serca_flux(REAL dt, REAL h, REAL u0, REAL u1, REAL* params)
{

  REAL   k_p1,  k_p2,  k_p3,  k_m1,  k_m2,  k_m3,  kdcai,  kdcasr,
kdh1,  kdhi, kdhsr,  kdh,  H_i,  ATP,  ADP,  P_i,  T_Cai,  T_Casr,
T_H1,  T_Hi,  T_Hsr, T_H,  a_p1,  a_p2,  a_p3,  a_m1,  a_m2,  a_m3,
s1,  s2,  s3, v_cycle, Ca_sr, Ca_i, T_Cai2,  T_Casr2,  density,
scale, v_cyt_to_A_sr;

  REAL Q, S,I, A, V, J, Flux;




  Ca_sr = u1/1000; // The serca model is formulated in mM, we use uM
  Ca_i = u0/1000;

  k_p1 = 25900;
  k_p2 = 2540;
  k_p3 = 20.5;
  k_m1 = 16;
  k_m2 = 67200;
  k_m3 = 149;
  kdcai = 0.9;
  kdcasr = 2.24;
  kdh1 = 1.09e-5;
  kdhi = 3.54e-3;
  kdhsr = 1.05e-8;
  kdh = 7.24e-5;
  H_i = 1e-4;
  ATP = 5;
  ADP = 36.3e-3;
  P_i = 1;

  T_Cai = Ca_i/kdcai;
  T_Casr = Ca_sr/kdcasr;

  T_Cai2 = T_Cai*T_Cai;
  T_Casr2 = T_Casr*T_Casr;

  T_H1 = H_i/kdh1;
  T_Hi = (H_i*H_i)/kdhi;
  T_Hsr = (H_i*H_i)/kdhsr;
  T_H = H_i/kdh;
  a_p1 = k_p1*ATP;
  a_p2 = k_p2*T_Cai2/(T_Cai2 + T_Cai2*T_Hi + T_Hi*(1 + T_H1));
  a_p3 = k_p3*T_Hsr/(T_Casr2*T_H + T_H + T_Hsr*(1 + T_H));
  a_m1 = k_m1*T_Hi/(T_Cai2 + T_Cai2*T_Hi + T_Hi*(1 + T_H1));
  a_m2 = k_m2*ADP*T_Casr2*T_H/(T_Casr2*T_H + T_H + T_Hsr*(1 + T_H));
  a_m3 = k_m3*P_i;
  s1 = a_p2*a_p3 + a_m1*a_p3 + a_m1*a_m2;
  s2 = a_p1*a_p3 + a_m2*a_p1 + a_m2*a_m3;
  s3 = a_p1*a_p2 + a_m3*a_m1 + a_m3*a_p2;
  v_cycle = (a_p1*a_p2*a_p3 - a_m1*a_m2*a_m3)/(s1 + s2 + s3); // #/s


  // Get parameters
  v_cyt_to_A_sr = params[0]; // not used in this formulation
  density = params[1]; // number of SERCAs pr nm^2
  scale = params[2]; // not this either

  // converts from #/s to uM/ms
  Q = 1e-3/6e17; // conversion factor from #/s to umol/ms
  S = v_cycle*Q;  // mass flux per serca (umol/ms)

  I = density*S; // Flux density (umol/ms)/(nm^2)
  A = h*h;  // Area nm^2
  J = A*I;  // mass flux over the whole surface (umol/ms)
  V = 1e-24*h*h*h;  // volume in liter = (l/nm^3)*nm^3
  Flux = J/V; // concentration flux, uM/ms

  return dt*Flux;  // flux uM/ms

}
//-----------------------------------------------------------------------------
REAL ryr_flux(REAL dt, REAL h, REAL u0, REAL u1, REAL* params)
{

  // Flux is described as:
  //
  //   J_ryr = g_c/h^3(u0-u1)
  //
  // where:
  //
  //   g_c = K*i_c/(U0-U1)
  //
  // Here i_c is the unitary current through an open channel in pA, when it opens,
  // U1 and U0 is the initial concentration when this current is measured, and K 
  // a constant:
  //
  //   K = 10^15/(z*F)
  //
  // Here z is the valence of the current (2) and F Faraday's constant. 
  //
  // So with explicit Euler we have:
  // 
  //   J_ryr = g_c/h^3*(u0-u1)
  //   u0 -= dt*J_RyR;
  //   u1 += dt*J_RyR
  //
  // With K so large this scheme will not be stable so we need to solve it 
  // analytically with the following scheme:
  //
  //   c0 = (u1 + u0)/2
  //   c1 = (u1 - u0)/2
  //   u1 = c0 + c1*K_0 
  //   u0 = c0 - c1*K_0
  //
  // where K_0 is given by:
  //
  //   K_0 = exp(-2*dt*g_c/h^3)

  //REAL c0, c1;
  REAL i_c, K_0;

  // g_c = 10^15/(z*F)*i_c/(U1-U0)
  // K_0 = exp(-2*dt*g_c/h**3)
  i_c = params[6];
  K_0 = exp(-2*dt*1e15/(2*96485.)*i_c/(1300.-0.1)/(h*h*h));
  
  // The resulting analytic flux
  //printf("u0: %f; u1:%f\n", u0, u1);
  return (u0-u1)*(1-K_0)/2;

  //c0 = (*u1 + *u0)/2;
  //c1 = (*u1 - *u0)/2;
  //*u0 = c0 - c1*K_0;
  //*u1 = c0 + c1*K_0;

}
//-----------------------------------------------------------------------------
void init_RyR_model(RyR_model_t* ryr)
{

  ryr->states.N = 0;

  ryr->Kd_close = 62.5;
  ryr->params[0] = ryr->Kd_close;

  ryr->k_max_close = 10.0;
  ryr->params[1] = ryr->k_max_close;

  ryr->n_open = 2.8;
  ryr->params[2] = ryr->n_open;

  ryr->k_max_open = 0.7;
  ryr->params[3] = ryr->k_max_open;

  ryr->k_min_open = 0.0001;
  ryr->params[4] = ryr->k_min_open;

  ryr->n_close = -0.5;
  ryr->params[5] = ryr->n_close;

  ryr->i_unitary = 0.5;
  ryr->params[6] = ryr->i_unitary;

  ryr->k_min_close = 0.9;
  ryr->params[7] = ryr->k_min_close;

  ryr->t_close_ryrs = -1.0;
  ryr->params[8] = ryr->t_close_ryrs;

  ryr->Kd_open = 127.92;
  ryr->params[9] = ryr->Kd_open;

}
//-----------------------------------------------------------------------------
void init_pRyR_model(RyR_model_t* p_ryr)
{

  p_ryr->states.N = 0;

  p_ryr->Kd_close = 62.5;
  p_ryr->params[0] = p_ryr->Kd_close;

  p_ryr->k_max_close = 10.0;
  p_ryr->params[1] = p_ryr->k_max_close;

  p_ryr->n_open = 2.8;
  p_ryr->params[2] = p_ryr->n_open;

  p_ryr->k_max_open = 0.7;
  p_ryr->params[3] = p_ryr->k_max_open;

  p_ryr->k_min_open = 0.0001;
  p_ryr->params[4] = p_ryr->k_min_open;

  p_ryr->n_close = -0.5;
  p_ryr->params[5] = p_ryr->n_close;

  p_ryr->i_unitary = 0.5;
  p_ryr->params[6] = p_ryr->i_unitary;

  p_ryr->k_min_close = 0.9;
  p_ryr->params[7] = p_ryr->k_min_close;

  p_ryr->t_close_ryrs = -1.0;
  p_ryr->params[8] = p_ryr->t_close_ryrs;

  p_ryr->Kd_open = 127.92;
  p_ryr->params[9] = p_ryr->Kd_open;

}
//-----------------------------------------------------------------------------
void init_SERCA_model(SERCA_model_t* serca)
{
  serca->v_cyt_to_A_sr = 307.;
  serca->density = 75.;
  serca->scale = 1.0;

  serca->params[0] = serca->v_cyt_to_A_sr;
  serca->params[1] = serca->density;
  serca->params[2] = serca->scale;

}
//-----------------------------------------------------------------------------
void init_RyR_model_states_stochastically(RyR_model_t* ryr, REAL* species_at_boundary)
{

  unsigned int i;
  REAL ko, kc, c0, _r_var;
  
  REAL Kd_close, k_max_close, n_open, k_max_open, k_min_open,
    n_close, k_min_close, Kd_open;
    
  REAL* params = ryr->params;

  //Get parameters
  Kd_close = params[0];
  k_max_close = params[1];
  n_open = params[2];
  k_max_open = params[3];
  k_min_open = params[4];
  n_close = params[5];
  k_min_close = params[7];
  Kd_open = params[9];

  if (ryr->states.N==0)
    return;

  // Assume the state arrays has been constructed
  assert(ryr->states.s0);
  
  for (i=0; i<ryr->states.N; i++)
  {
    // Get species at outlet of the boundary
    c0 = species_at_boundary[i*2+1];

    // Compute intermediates for the stochastic evaluation
    ko = fmax(fmin(pow(c0/Kd_open, n_open), k_max_open), k_min_open);
    kc = fmax(fmin(pow(c0/Kd_close, n_close), k_max_close), k_min_close);

    // Stochastic evaluation
    _r_var = mt_ldrand();
    if (_r_var <= kc/(ko + kc))
      ryr->states.s0[i] = 0;
    else
      ryr->states.s0[i] = 1;
  }
}
//-----------------------------------------------------------------------------
void init_RyR_model_states_deterministically(RyR_model_t* ryr, 
					     unsigned int number_init_ryrs,
					     int* open_ryrs)
{
  
  if (ryr->states.N==0)
    return;

  unsigned int i;
  
  // Assume the state arrays has been constructed
  assert(ryr->states.s0);
  
  for (i=0; i<ryr->states.N; i++)
    ryr->states.s0[i] = 0;

  for (i=0; i<number_init_ryrs; i++)
    ryr->states.s0[open_ryrs[i]] = 1;

}
//-----------------------------------------------------------------------------
void evaluate_RyR_stochastically(REAL* params, ModelStates* states, 
                            REAL t, REAL dt, REAL* species_at_boundary)
{
  
  unsigned int i;
  REAL ko, kc, c0, _r_var;

  if (states->N==0)
    return;
    
  REAL Kd_close, k_max_close, n_open, k_max_open, k_min_open,
    n_close, k_min_close, t_close_ryrs, Kd_open;

  //Get parameters
  Kd_close = params[0];
  k_max_close = params[1];
  n_open = params[2];
  k_max_open = params[3];
  k_min_open = params[4];
  n_close = params[5];
  k_min_close = params[7];
  t_close_ryrs = params[8];
  Kd_open = params[9];

  // Assume the state arrays has been constructed
  assert(states->s0);
  
  for (i=0; i<states->N; i++)
  {
    //FIXME: Need to deal with that part also. Figure out how to define it
    //FIXME: in the .sm file
    
    // Check for forcing ryr to close (or be open)
    if (t_close_ryrs>0)
    {

      // Check for close time
      if (t_close_ryrs<=t+dt)
        states->s0[i] = 0;
      
      // Do not continue to stochastic evaluation
      continue;
    }

    // Get species at outlet of the boundary
    c0 = species_at_boundary[i*2+1];
    kc = fmax(fmin(pow(c0/Kd_close, n_close), k_max_close), k_min_close);
    ko = fmax(fmin(pow(c0/Kd_open, n_open), k_max_open), k_min_open);


    // Change state if necessary
    _r_var = mt_ldrand();
    switch (states->s0[i])
    {
      case 0:
        if (_r_var <= dt*ko)
          states->s0[i] = 1;
        break;
      case 1:
        if (_r_var <= dt*kc)
          states->s0[i] = 0;
        break;
    }
  }
}
//-----------------------------------------------------------------------------
BoundaryFluxes_t* BoundaryFluxes_construct(Species_t* species, hid_t file_id, 
                                           arguments_t* arguments)
{
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;

  BoundaryFluxes_t* fluxes = mpi_malloc(comm, sizeof(BoundaryFluxes_t));
  fluxes->ryr.states.s0 = NULL;
  fluxes->p_ryr.states.s0 = NULL;

  // Initalize ryr and serca models with default parameters
  init_RyR_model(&fluxes->ryr);
  init_pRyR_model(&fluxes->p_ryr);
  init_SERCA_model(&fluxes->serca);

  // Read usage flags
  read_h5_attr(comm, file_id, "/", "use_ryr", &fluxes->use_ryr);
  read_h5_attr(comm, file_id, "/", "use_p_ryr", &fluxes->use_p_ryr);
  read_h5_attr(comm, file_id, "/", "use_serca", &fluxes->use_serca);
  
  fluxes->map = NULL;
  fluxes->boundary_map = NULL;
  
  // Calculate the number of fluxes
  fluxes->num_of_used_fluxes = 0;
  fluxes->num_of_used_fluxes += fluxes->use_ryr;
  fluxes->num_of_used_fluxes += fluxes->use_p_ryr;
  fluxes->num_of_used_fluxes += fluxes->use_serca;

  // Return struct if no boundary fluxes
  if (fluxes->num_of_used_fluxes==0)
    return fluxes;

  // Allocate memory for mappings
  fluxes->map = mpi_malloc(comm, sizeof(Flux_t*)*fluxes->num_of_used_fluxes);
  fluxes->boundary_map = mpi_malloc(comm, sizeof(unsigned int)*fluxes->num_of_used_fluxes);
  
  unsigned int map_ind = 0;
  
  // Initialize ryr data
  if (fluxes->use_ryr)
  {

    // Read RyR attributes from file
    read_h5_attr(comm, file_id, "/ryr", "Kd_open", &fluxes->ryr.Kd_open);
    read_h5_attr(comm, file_id, "/ryr", "k_min_open", &fluxes->ryr.k_min_open);
    read_h5_attr(comm, file_id, "/ryr", "k_max_open", &fluxes->ryr.k_max_open);
    read_h5_attr(comm, file_id, "/ryr", "n_open", &fluxes->ryr.n_open);

    read_h5_attr(comm, file_id, "/ryr", "Kd_close", &fluxes->ryr.Kd_close);
    read_h5_attr(comm, file_id, "/ryr", "k_min_close", &fluxes->ryr.k_min_close);
    read_h5_attr(comm, file_id, "/ryr", "k_max_close", &fluxes->ryr.k_max_close);
    read_h5_attr(comm, file_id, "/ryr", "n_close", &fluxes->ryr.n_close);

    read_h5_attr(comm, file_id, "/ryr", "i_unitary", &fluxes->ryr.i_unitary);
    read_h5_attr(comm, file_id, "/ryr", "boundary", &fluxes->ryr.boundary_name);
    read_h5_attr(comm, file_id, "/ryr", "species", &fluxes->ryr.species_name);
    
    fluxes->ryr.t_close_ryrs = arguments_get_t_close(arguments, "ryr");
    
    // Update flux mapping
    fluxes->boundary_map[map_ind] = Geometry_get_boundary_id(geom, fluxes->ryr.boundary_name);
    fluxes->map[map_ind] = mpi_malloc(comm, sizeof(Flux_t));
    fluxes->map[map_ind]->flux_params = fluxes->ryr.params;
    fluxes->map[map_ind]->flux_function = &ryr_flux;
    fluxes->map[map_ind]->flux_dsi = Species_get_diffusive_species_id(species, fluxes->ryr.species_name);
    
    fluxes->map[map_ind]->storef = mpi_malloc(comm, sizeof(StochasticReferences));
    fluxes->map[map_ind]->storef->states = &fluxes->ryr.states;
    fluxes->map[map_ind]->storef->evaluate = &evaluate_RyR_stochastically;
    fluxes->map[map_ind]->storef->open_states = &open_states_RyR; // ++
    map_ind++;
    
    // Update flux parameters
    fluxes->ryr.params[0] = fluxes->ryr.Kd_close;
    fluxes->ryr.params[1] = fluxes->ryr.k_max_close;
    fluxes->ryr.params[2] = fluxes->ryr.n_open;
    fluxes->ryr.params[3] = fluxes->ryr.k_max_open;
    fluxes->ryr.params[4] = fluxes->ryr.k_min_open;
    fluxes->ryr.params[5] = fluxes->ryr.n_close;
    fluxes->ryr.params[6] = fluxes->ryr.i_unitary;
    fluxes->ryr.params[7] = fluxes->ryr.k_min_close;
    fluxes->ryr.params[8] = fluxes->ryr.t_close_ryrs;
    fluxes->ryr.params[9] = fluxes->ryr.Kd_open;

    // Allocate states only on rank 0 process
    if (geom->mpi_info.rank == 0)
    {
      // Allocate global RyR states
      fluxes->ryr.states.N  = geom->boundary_size[Geometry_get_boundary_id(geom, "ryr")];
      fluxes->ryr.states.s0 = mpi_malloc(comm, sizeof(domain_id)*fluxes->ryr.states.N);
    }
  }

    if (fluxes->use_p_ryr)
  {

    // Read pRyR attributes from file
    read_h5_attr(comm, file_id, "/p_ryr", "Kd_open", &fluxes->p_ryr.Kd_open);
    read_h5_attr(comm, file_id, "/p_ryr", "k_min_open", &fluxes->p_ryr.k_min_open);
    read_h5_attr(comm, file_id, "/p_ryr", "k_max_open", &fluxes->p_ryr.k_max_open);
    read_h5_attr(comm, file_id, "/p_ryr", "n_open", &fluxes->p_ryr.n_open);

    read_h5_attr(comm, file_id, "/p_ryr", "Kd_close", &fluxes->p_ryr.Kd_close);
    read_h5_attr(comm, file_id, "/p_ryr", "k_min_close", &fluxes->p_ryr.k_min_close);
    read_h5_attr(comm, file_id, "/p_ryr", "k_max_close", &fluxes->p_ryr.k_max_close);
    read_h5_attr(comm, file_id, "/p_ryr", "n_close", &fluxes->p_ryr.n_close);

    read_h5_attr(comm, file_id, "/p_ryr", "i_unitary", &fluxes->p_ryr.i_unitary);
    read_h5_attr(comm, file_id, "/p_ryr", "boundary", &fluxes->p_ryr.boundary_name);
    read_h5_attr(comm, file_id, "/p_ryr", "species", &fluxes->p_ryr.species_name);
    
    fluxes->p_ryr.t_close_ryrs = arguments_get_t_close(arguments, "p_ryr");
    
    // Update flux mapping
    fluxes->boundary_map[map_ind] = Geometry_get_boundary_id(geom, fluxes->p_ryr.boundary_name);
    fluxes->map[map_ind] = mpi_malloc(comm, sizeof(Flux_t));
    fluxes->map[map_ind]->flux_params = fluxes->p_ryr.params;
    fluxes->map[map_ind]->flux_function = &ryr_flux;
    fluxes->map[map_ind]->flux_dsi = Species_get_diffusive_species_id(species, fluxes->p_ryr.species_name);
    
    fluxes->map[map_ind]->storef = mpi_malloc(comm, sizeof(StochasticReferences));
    fluxes->map[map_ind]->storef->states = &fluxes->p_ryr.states;
    fluxes->map[map_ind]->storef->evaluate = &evaluate_RyR_stochastically;
    fluxes->map[map_ind]->storef->open_states = &open_states_RyR; // ++
    map_ind++;
    
    // Update flux parameters
    fluxes->p_ryr.params[0] = fluxes->p_ryr.Kd_close;
    fluxes->p_ryr.params[1] = fluxes->p_ryr.k_max_close;
    fluxes->p_ryr.params[2] = fluxes->p_ryr.n_open;
    fluxes->p_ryr.params[3] = fluxes->p_ryr.k_max_open;
    fluxes->p_ryr.params[4] = fluxes->p_ryr.k_min_open;
    fluxes->p_ryr.params[5] = fluxes->p_ryr.n_close;
    fluxes->p_ryr.params[6] = fluxes->p_ryr.i_unitary;
    fluxes->p_ryr.params[7] = fluxes->p_ryr.k_min_close;
    fluxes->p_ryr.params[8] = fluxes->p_ryr.t_close_ryrs;
    fluxes->p_ryr.params[9] = fluxes->p_ryr.Kd_open;

    // Allocate states only on rank 0 process
    if (geom->mpi_info.rank == 0)
    {
      // Allocate global RyR states
      fluxes->p_ryr.states.N  = geom->boundary_size[Geometry_get_boundary_id(geom, "p_ryr")];
      fluxes->p_ryr.states.s0 = mpi_malloc(comm, sizeof(domain_id)*fluxes->p_ryr.states.N);
    }
  }

  // Initialize serca data
  if (fluxes->use_serca)
  {

    // Read SERCA attributes from file
    read_h5_attr(comm, file_id, "/serca", "v_cyt_to_A_sr", &fluxes->serca.v_cyt_to_A_sr);
    read_h5_attr(comm, file_id, "/serca", "density", &fluxes->serca.density);
    read_h5_attr(comm, file_id, "/serca", "scale", &fluxes->serca.scale);
    read_h5_attr(comm, file_id, "/serca", "boundary", &fluxes->serca.boundary_name);
    read_h5_attr(comm, file_id, "/ryr", "species", &fluxes->serca.species_name);
    
    // update flux mapping
    fluxes->boundary_map[map_ind] = Geometry_get_boundary_id(geom, fluxes->serca.boundary_name);
    fluxes->map[map_ind] = mpi_malloc(comm, sizeof(Flux_t));
    fluxes->map[map_ind]->flux_params = fluxes->serca.params;
    fluxes->map[map_ind]->flux_function = &serca_flux;
    fluxes->map[map_ind]->flux_dsi = Species_get_diffusive_species_id(species, fluxes->serca.species_name);
    
    // This model has no extra states variables
    fluxes->map[map_ind]->storef = 0;
    map_ind++;
    
    // Update flux parameters
    fluxes->serca.params[0] = fluxes->serca.v_cyt_to_A_sr;
    fluxes->serca.params[1] = fluxes->serca.density;
    fluxes->serca.params[2] = fluxes->serca.scale;

  }
  
  // Sanity check
  assert(map_ind == fluxes->num_of_used_fluxes);

  return fluxes;
}
//-----------------------------------------------------------------------------
void BoundaryFluxes_init_stochastic_boundaries(Species_t* species, arguments_t* arguments)
{
  unsigned int dbi;
  Geometry_t* geom = species->geom;
  BoundaryFluxes_t* fluxes = species->boundary_fluxes;
  const OS_info* info;
  const OS_info* info_p_ryr;
  MPI_Comm comm = geom->mpi_info.comm;
  
  // Communicate species values at discrete boundaries to processor 0
  Species_communicate_values_at_discrete_boundaries(species);
   
  // Code generator will generate as many if statement as number of
  // discrete fluxes that need stochastic evaluations
   
  // Initialize ryr data
  if (fluxes->use_ryr && geom->num_global_discrete_boundaries>0)
  {
    if (geom->mpi_info.rank == 0)
    {
      // Initiate state values
      info = arguments_get_open_states_info(arguments, fluxes->ryr.boundary_name);
      info_p_ryr = arguments_get_open_states_info(arguments, fluxes->p_ryr.boundary_name);
 
      if (info)
      {
        // The flux->ryr.name will be generated by the code generator
        init_RyR_model_states_deterministically(&fluxes->ryr, info->number_open_states, 
                                                info->open_states);
      }
      else
      {
        // The value 0 will be generated by the code generator, and it is
        // equivalent to the order of definining fluxes in .flux file
        // In the example file serca_ryr.flux ryr is a second flux so the index is 1.
        dbi = Geometry_get_discrete_boundary_id(geom, fluxes->boundary_map[0]);
        init_RyR_model_states_stochastically(&fluxes->ryr, \
            geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[dbi]);
      }

      if (info_p_ryr)
      {
        // The flux->ryr.name will be generated by the code generator
        init_RyR_model_states_deterministically(&fluxes->p_ryr, info_p_ryr->number_open_states, 
                                                info_p_ryr->open_states);
      }
      else
      {
        // The value 0 will be generated by the code generator, and it is
        // equivalent to the order of definining fluxes in .flux file
        // In the example file serca_ryr.flux ryr is a second flux so the index is 1.
        dbi = Geometry_get_discrete_boundary_id(geom, fluxes->boundary_map[0]);
        init_RyR_model_states_stochastically(&fluxes->p_ryr, \
            geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[dbi]);
      }
    }
  }

  // Communicate the openness of the discrete boundaries from rank 0 to other processes
  Species_communicate_openness_of_discrete_boundaries(species);

}
//-----------------------------------------------------------------------------
void BoundaryFluxes_output_init_data(Species_t* species, arguments_t* arguments)
{
  BoundaryFluxes_t* fluxes = species->boundary_fluxes;
  Geometry_t* geom = species->geom;
  MPI_Comm comm = geom->mpi_info.comm;
  
  if (!fluxes->num_of_used_fluxes)
    return;

  mpi_printf0(comm, "\n");
  mpi_printf0(comm, "Boundary fluxes:\n");
  mpi_printf0(comm, "-----------------------------------------------------------------------------\n");

  if (fluxes->use_ryr)
  {
    mpi_printf0(comm, "  RyR:\n");
    mpi_printf0(comm, "  Open:  kd: %4.2f, N: %.2f, k_min: %.2f, k_max: %.2f\n", 
                fluxes->ryr.Kd_open, fluxes->ryr.n_open, fluxes->ryr.k_min_open, 
                fluxes->ryr.k_max_open);
    mpi_printf0(comm, "  Close:  kd: %4.2f, N: %.2f, k_min: %.2f, k_max: %.2f\n", 
                fluxes->ryr.Kd_close, fluxes->ryr.n_close, fluxes->ryr.k_min_close, 
                fluxes->ryr.k_max_close);

    mpi_printf0(comm, "  Flux:        i_c: %.2f pA\n", fluxes->ryr.i_unitary);

    if (geom->mpi_info.rank == 0)
    {
      int i, open = 0;
      for (i=0; i<fluxes->ryr.states.N; i++)
        open += open_states_RyR(&fluxes->ryr.states, i); //+++
      mpi_printf0(comm, "  Num open RyRs: %d/%d\n", open, fluxes->ryr.states.N);
    }
    mpi_printf0(comm, "\n");
  }

  if (fluxes->use_p_ryr)
  {
    mpi_printf0(comm, "  pRyR:\n");
    mpi_printf0(comm, "  Open:  kd: %4.2f, N: %.2f, k_min: %.2f, k_max: %.2f\n", 
                fluxes->p_ryr.Kd_open, fluxes->p_ryr.n_open, fluxes->p_ryr.k_min_open, 
                fluxes->p_ryr.k_max_open);
    mpi_printf0(comm, "  Close:  kd: %4.2f, N: %.2f, k_min: %.2f, k_max: %.2f\n", 
                fluxes->p_ryr.Kd_close, fluxes->p_ryr.n_close, fluxes->p_ryr.k_min_close, 
                fluxes->p_ryr.k_max_close);

    mpi_printf0(comm, "  Flux:        i_c: %.2f pA\n", fluxes->p_ryr.i_unitary);

    if (geom->mpi_info.rank == 0)
    {
      int i, open = 0;
      for (i=0; i<fluxes->p_ryr.states.N; i++)
        open += open_states_RyR(&fluxes->p_ryr.states, i); //+++
      mpi_printf0(comm, "  Num open pRyRs: %d/%d\n", open, fluxes->p_ryr.states.N);
    }
    mpi_printf0(comm, "\n");
  }
  
  
  if (fluxes->use_serca)
  {
    mpi_printf0(comm, "  SERCA:\n");
    mpi_printf0(comm, "  v_cyt_to_A_sr: %4.2g\n", fluxes->serca.v_cyt_to_A_sr);
    mpi_printf0(comm, "        density: %4.2g\n", fluxes->serca.density);
    mpi_printf0(comm, "          scale: %4.2g\n", fluxes->serca.scale);
  }
}
//-----------------------------------------------------------------------------
void BoundaryFluxes_destruct(BoundaryFluxes_t* fluxes)
{

  if (!fluxes->num_of_used_fluxes)
    return;
    
  domain_id internal_id;
  
  for(internal_id=0; internal_id<fluxes->num_of_used_fluxes; internal_id++)
  {
    if(fluxes->map[internal_id]->storef)
      free(fluxes->map[internal_id]->storef);
    free(fluxes->map[internal_id]);
  }
  free(fluxes->map);
  free(fluxes->boundary_map);

  free(fluxes->ryr.states.s0);
  free(fluxes);
  
}
//-----------------------------------------------------------------------------
Flux_t* BoundaryFluxes_get_flux_info(BoundaryFluxes_t* fluxes, domain_id boundary_ind)
{
  domain_id internal_id;
  
  for(internal_id=0; internal_id<fluxes->num_of_used_fluxes; internal_id++)
    if(fluxes->boundary_map[internal_id] == boundary_ind)
      return fluxes->map[internal_id];
      
  return NULL;
}
//-----------------------------------------------------------------------------
domain_id open_states_RyR(const ModelStates* states, int i)
{
  assert(i<states->N);
  return states->s0[i] == 1;
}

