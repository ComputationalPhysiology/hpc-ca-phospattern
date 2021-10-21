#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include "vectorclass.h"

typedef unsigned char domain_id;
typedef double REAL;

#include "vectorclass_extra.h"

#define max_num_domains 5   // assuming five domains (cyt,cleft,jsr,nsr,tt)
#define N2 max_num_domains*max_num_domains

void static inline stream(REAL *u1, REAL *du) {
#if INSTRSET > 8 
  _mm512_stream_pd(u1, _mm512_loadu_pd(du));
}
#elif INSTRSET > 6
  _mm256_stream_pd(&u1[0], _mm256_loadu_pd(&du[0]));
  _mm256_stream_pd(&u1[4], _mm256_loadu_pd(&du[4]));
}
#else
  _mm_stream_pd(&u1[0], _mm_loadu_pd(&du[0]));
  _mm_stream_pd(&u1[2], _mm_loadu_pd(&du[2]));
  _mm_stream_pd(&u1[4], _mm_loadu_pd(&du[4]));
  _mm_stream_pd(&u1[6], _mm_loadu_pd(&du[6]));
}
#endif

void apply_du_stream(double *u1, double *du, int n) {
  //AVX512 stream must be 64 byte aligned
  uintptr_t tp  = (uintptr_t)u1;
  tp = (tp+63)&-64;  
  double *u2 = (double*)tp;
  int n1 = u2-u1;
  int n2 = (n-n1)/8; //8 doubles is 64 bytes
  for(int i=0; i<n1; i++) {
    u1[i] = du[i];
  }
  #pragma omp parallel for
  for(int i=0; i<n2; i++) {
    stream(&u1[8*i+n1], &du[8*i+n1]);
  }
  for(int i=8*n2; i<n; i++) {
    u1[i] = du[i];
  }
}

/*
void foo() {
  for (bi=0; bi<geom->num_boundaries; bi++) {

    flux_info = BoundaryFluxes_get_flux_info(species->boundary_fluxes, bi);
    if (!flux_info)
      continue;
      
    dsi = flux_info->flux_dsi;
    
    if (time_ind % species->species_substeps[dsi] != 0)
      continue;
      
    si = species->all_diffusive[dsi];   

    for (bvi=0; bvi<species->num_boundary_voxels[bi]; bvi++) {
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
}
*/

extern "C" void apply_du(REAL *u1, REAL *du, int total_n, int nx, int ny, int nz) {
  int x_shift = ny*nz, y_shift = nz;
  REAL **u1p = &u1;
  REAL **dup = &du;  
  REAL *tmp = *u1p;
#if defined(SWAP)


  /*
  for (int xi=0; xi<nx; xi++) {
    for (int yi=0; yi<ny; yi++) {
      for (int zi=0;zi<nz; zi++) {
	int offset = xi*x_shift + yi*y_shift + zi;
	u1[offset] = du[offset];
      }
    }
  }
  */
  *u1p = *dup;
  *dup = tmp;
#else
#if defined(STREAM)
  apply_du_stream(u1, du, total_n);
#else
  #pragma omp parallel for
  for (int i=0; i<total_n; i++) u1[i] = du[i];
#endif  
#endif
}

#if INSTRSET > 8 
static int64_t broadcast_char(int64_t t) {
  return t = 0xff & t, t |= (t<<8), t |= (t<<16), t |= (t<<32);
}
#elif INSTRSET > 6
static int32_t broadcast_char(int32_t t) {
  return t = 0xff & t, t |= (t<<8), t |= (t<<16);
}
#else
static int16_t broadcast_char(int16_t t) {
  return t = 0xff & t, t |= (t<<8);
}
#endif

extern "C" void init_stencil_test2(const domain_id* dm_ids, char * voxels_stencil_test, int nx, int ny, int nz) {
  const unsigned int x_shift  = ny*nz, y_shift  = nz, z_shift = 1;

  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      int nz2 = (nz-1)&-VECSIZE;
      int dz = nz - nz2;
      for (int zi=0; zi<nz; zi+=VECSIZE) {
	int offset = xi*x_shift + yi*y_shift + zi;
	int offsetm = offset - z_shift, offsetp = offset + z_shift;  // Z direction
	int offsetu = offset - y_shift, offsetd = offset + y_shift;  // Y direction
	int offsetb = offset - x_shift, offsetf = offset + x_shift;  // X direction
        #if INSTRSET > 8 
        int64_t x = 0, xm = 0 , xp = 0, xu = 0 , xd = 0 , xf = 0 , xb = 0;
        #elif INSTRSET > 6
        int32_t x = 0, xm = 0 , xp = 0, xu = 0 , xd = 0 , xf = 0 , xb = 0;
        #else
        int16_t x = 0, xm = 0 , xp = 0, xu = 0 , xd = 0 , xf = 0 , xb = 0;
        #endif
	int bytes = VECSIZE;
	if (zi==nz2) bytes = dz;
	memcpy( &x, &dm_ids[offset ], bytes);
	memcpy(&xm, &dm_ids[offsetm], bytes);
	memcpy(&xp, &dm_ids[offsetp], bytes);
	memcpy(&xu, &dm_ids[offsetu], bytes);
	memcpy(&xd, &dm_ids[offsetd], bytes);
	memcpy(&xf, &dm_ids[offsetf], bytes);
	memcpy(&xb, &dm_ids[offsetb], bytes);
	int flag;
	if(zi==0) {
	  flag =            x == xp && x == xu && x == xd && x == xf && x == xb;
	}
	else if (zi==nz2) {
	  flag =            x == xm && x == xu && x == xd && x == xf && x == xb;
	}
	else {
	  flag = x == xm && x == xp && x == xu && x == xd && x == xf && x == xb;
	}
	if(!flag || x != broadcast_char(x)) {
	  voxels_stencil_test[offset] = 1;
	}
	else {
	  voxels_stencil_test[offset] = 0;
	}
      }
    }
  }
}

void side(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset, int incr1, int incr2, int incr3, int ghosted) {
  Vecnd u, um, up, uu, ud, ub, d;

  int offsetm = offset - incr2, offsetp = offset + incr2;
  int offsetu = offset - incr1, offsetd = offset + incr1;
  int offsetb = offset - incr3;

  u  = Vecnd().load(&u1[offset]);
  um = Vecnd().load(&u1[offsetm]);
  up = Vecnd().load(&u1[offsetp]);
  uu = Vecnd().load(&u1[offsetu]);
  ud = Vecnd().load(&u1[offsetd]);
  ub = Vecnd().load(&u1[offsetb]);

  Vecnd cm, cp, cu, cd, cf, cb;
  Vecni dicv, dirv;
  dirv = max_num_domains*get_index(&dm_ids[offset]);
  cm = lookupi(dirv + get_index(&dm_ids[offsetm]), coef_table);
  cp = lookupi(dirv + get_index(&dm_ids[offsetp]), coef_table);
  cu = lookupi(dirv + get_index(&dm_ids[offsetu]), coef_table);
  cd = lookupi(dirv + get_index(&dm_ids[offsetd]), coef_table);
  cb = lookupi(dirv + get_index(&dm_ids[offsetb]), coef_table);

  d = mul_add(-u, cm + cp + cu + cd + cb, cm*um + cp*up  + cu*uu + cd*ud + cb*ub + u);
  d.store(&du[offset]);
}

extern "C" void side_SIMD(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset0, int offset20, int incr1, int incr2, int incr3, int ghosted, int len2) {
  int idx2 = 1;
  for (; idx2<((len2-2)&-VECSIZE); idx2+=VECSIZE) {
    int offset = offset0 + idx2*incr2;
    int offset2 = offset20 + idx2;
    side(u1, du, coef_table, &ghost_value_array[offset2], dm_ids, &ghost_dm_ids[offset2], dsi, offset, incr1, incr2, incr3, ghosted);
  }
  for (; idx2<len2-1; idx2++) {
    int offset = offset0 + idx2*incr2;
    int offset_m, offset_p;
    domain_id di, di_n;
    double u = u1[offset];
    double *coef = coef_table+(dm_ids[offset]*max_num_domains);
    
    du[offset] = u;
    offset_m = offset-incr2; di = dm_ids[offset_m];
    offset_p = offset+incr2; di_n = dm_ids[offset_p];
    du[offset] += coef[di]*(u1[offset_m]-u)+coef[di_n]*(u1[offset_p]-u);
    
    offset_m = offset-incr1; di = dm_ids[offset_m];
    offset_p = offset+incr1; di_n = dm_ids[offset_p];
    du[offset] += coef[di]*(u1[offset_m]-u)+coef[di_n]*(u1[offset_p]-u);
    
    offset_p = offset-incr3; di_n = dm_ids[offset_p];
    du[offset] += coef[di_n]*(u1[offset_p]-u);
    
  }
}

extern "C" void side_ghosted(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset, int incr1, int incr2, int incr3, int ghosted) {
  Vecnd u, um, up, uu, ud, uf, ub, d;

  int offsetm = offset - incr2, offsetp = offset + incr2;
  int offsetu = offset - incr1, offsetd = offset + incr1;
  int offsetb = offset - incr3;

  u  = Vecnd().load(&u1[offset]);
  um = Vecnd().load(&u1[offsetm]);
  up = Vecnd().load(&u1[offsetp]);
  uu = Vecnd().load(&u1[offsetu]);
  ud = Vecnd().load(&u1[offsetd]);
  uf = Vecnd().load(ghost_value_array);
  ub = Vecnd().load(&u1[offsetb]);

  Vecnd cm, cp, cu, cd, cf, cb;
  Vecni dicv, dirv;
  dirv = max_num_domains*get_index(&dm_ids[offset]);
  cm = lookupi(dirv + get_index(&dm_ids[offsetm]), coef_table);
  cp = lookupi(dirv + get_index(&dm_ids[offsetp]), coef_table);
  cu = lookupi(dirv + get_index(&dm_ids[offsetu]), coef_table);
  cd = lookupi(dirv + get_index(&dm_ids[offsetd]), coef_table);
  cb = lookupi(dirv + get_index(&dm_ids[offsetb]), coef_table);
  cf = lookupi(dirv + get_index(ghost_dm_ids),     coef_table);

  d = mul_add(-u, cm + cp + cu + cd + cf + cb, cm*um + cp*up  + cu*uu + cd*ud + cf*uf  + cb*ub + u);
  d.store(&du[offset]);
}

extern "C" void side_ghosted_SIMD(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset0, int offset20, int incr1, int incr2, int incr3, int ghosted, int len2) {
  int idx2 = 1;
  for (; idx2<((len2-2)&-VECSIZE); idx2+=VECSIZE) {
    int offset = offset0 + idx2*incr2;
    int offset2 = offset20 + idx2;
    side_ghosted(u1, du, coef_table, &ghost_value_array[offset2], dm_ids, &ghost_dm_ids[offset2], dsi, offset, incr1, incr2, incr3, ghosted);
  }
  for (; idx2<len2-1; idx2++) {
    int offset = offset0 + idx2*incr2;
    int offset_m, offset_p;
    domain_id di, di_n;
    double u = u1[offset];
    double *coef = coef_table+(dm_ids[offset]*max_num_domains);
    du[offset] = u;
    
    offset_m = offset-incr2; di = dm_ids[offset_m];
    offset_p = offset+incr2; di_n = dm_ids[offset_p];
    du[offset] += coef[di]*(u1[offset_m]-u)+coef[di_n]*(u1[offset_p]-u);
    
    offset_m = offset-incr1; di = dm_ids[offset_m];
    offset_p = offset+incr1; di_n = dm_ids[offset_p];
    du[offset] += coef[di]*(u1[offset_m]-u)+coef[di_n]*(u1[offset_p]-u);
    
    offset_p = offset-incr3; di_n = dm_ids[offset_p];
    du[offset] += coef[di_n]*(u1[offset_p]-u);
  }
}

static void kernel_alpha(REAL *u1, REAL*du, REAL *alphax, REAL *alphay, REAL *alphaz, int offset, int x_shift, int y_shift, Vecndb maska, Vecndb maskm, Vecndb maskp) {
  Vecnd um, up, uu, ud, uf, ub, u, d;
  Vecnd cm, cp, cu, cd, cf, cb;
  
  int offsetm = offset - 1, offsetp = offset + 1;
  int offsetu = offset - y_shift, offsetd = offset + y_shift;
  int offsetb = offset - x_shift, offsetf = offset + x_shift;

  u  = Vecnd().load(&u1[offset]);
  um = load(&u1[offsetm], maskm);
  up = load(&u1[offsetp], maskp);
  uu = Vecnd().load(&u1[offsetu]);
  ud = Vecnd().load(&u1[offsetd]);
  uf = Vecnd().load(&u1[offsetf]);
  ub = Vecnd().load(&u1[offsetb]);

  cm = load(&alphaz[offsetm], maskm);
  cp = load(&alphaz[offset ], maskp);
  cu = Vecnd().load(&alphay[offsetu]);
  cd = Vecnd().load(&alphay[offset ]);
  cb = Vecnd().load(&alphax[offsetb]);
  cf = Vecnd().load(&alphax[offset ]);
  
  d = mul_add(-u, cm + cp + cu + cd + cf + cb, cm*um + cp*up  + cu*uu + cd*ud + cf*uf  + cb*ub + u);

  store(&du[offset], maska, d);
}

extern "C" void innerZ_SIMD(const REAL *u1, REAL *du, REAL *coef_table, int len2, int idx1, int incr1, int incr2, int offset2, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids) {
  
  int i=1;  
  
  #if INSTRSET > 7
  #if INSTRSET > 8
  Vecni incv = incr2*Vecni(0,1,2,3,4,5,6,7) + offset2;
  #else
  Vecni incv = incr2*Vecni(0,1,2,3) + offset2;
  #endif
  for (; i<((len2-3)&-VECSIZE); i+=VECSIZE) {    
    Vecni div, offsetv;
    Vecnd coefv, uf, u, d;
    int offset2 = idx1*len2 + i;

    offsetv = i*incr2  + incv;
     
    uf = Vecnd().load(&ghost_value_array[offset2]);

    div  = max_num_domains*get_index2(dm_ids, offsetv) + get_index(&ghost_dm_ids[offset2]);
    coefv = lookupi(div, coef_table);
    d = lookupi(offsetv, du) + coefv*uf;
    scatter(du, offsetv, d);
  }
  #endif
  
  for (; i<len2-1; i++) {
    int offset = offset2 + i*incr2;	
    int offset2 = idx1*len2 + i;
    REAL *coef = &coef_table[dm_ids[offset]*max_num_domains];
    REAL u = u1[offset];
    du[offset] += coef[ghost_dm_ids[offset2]]*(ghost_value_array[offset2]);
  }
}

extern "C" void innerZ_join_SIMD(const REAL *u1, REAL *du, REAL *coef_table, int len2, int idx1, int incr1, int incr2, const REAL* ghost_value_array4, const REAL* ghost_value_array5, const domain_id* dm_ids, domain_id** ghost_dm_ids) {
  int i = 2;

  #if INSTRSET > 7
  #if INSTRSET > 8
  Vecni incv = incr2*Vecni(0,1,2,3,4,5,6,7) + idx1*incr1 - 1;
  #else
  Vecni incv = incr2*Vecni(0,1,2,3) + idx1*incr1 - 1;
  #endif
  for (i=2; i<((len2-3)&-VECSIZE); i+=VECSIZE) {
    int offset2;
    Vecni div, offsetv;
    Vecnd coefv, uf, u, d;    

    offsetv = i*incr2  + incv;
    offset2 = idx1*len2  + i - 1;
      
    uf = Vecnd().load(&ghost_value_array5[offset2]);
    div  = max_num_domains*get_index2(dm_ids, offsetv) + get_index(&ghost_dm_ids[5][offset2]);
    coefv = lookupi(div, coef_table);
    d = lookupi(offsetv, du) + coefv*uf;
    scatter(du, offsetv, d);
    
    offsetv++, offset2++;

    uf = Vecnd().load(&ghost_value_array4[offset2]);
    div  = max_num_domains*get_index2(dm_ids, offsetv) + get_index(&ghost_dm_ids[4][offset2]);
    coefv = lookupi(div, coef_table);
    d = lookupi(offsetv, du) + coefv*uf;
    scatter(du, offsetv, d);
  }
  #endif

  for (; i<len2-1; i++) {
    REAL *coef;
    int offset  = idx1*incr1 + i*incr2 - 1;
    int offset2 = idx1*len2  + i - 1;

    coef = &coef_table[dm_ids[offset]*max_num_domains];
    du[offset] += coef[ghost_dm_ids[5][offset2]]*(ghost_value_array5[offset2]);
    
    coef = &coef_table[dm_ids[offset + 1]*max_num_domains];
    du[offset + 1] += coef[ghost_dm_ids[4][offset2 + 1]]*(ghost_value_array4[offset2 + 1]);
  }
}

static void reaction(double **u1, double **du, double * k_onT, double * k_offT, double *totT, domain_id* bsp0T, domain_id* bsp1T, int bi, int offset, REAL dt, const domain_id* dm_ids, int ndomains, domain_id* update_species) {
  #if INSTRSET > 8 
  int64_t x  = *(int64_t *)&dm_ids[offset];
  #elif INSTRSET > 6
  int32_t x  = *(int32_t*)&dm_ids[offset];
  #else
  int16_t x  = *(int16_t*)&dm_ids[offset];
  #endif
  if(x==0) {
    int di = dm_ids[offset];
    domain_id bsi0 = bsp0T[bi*ndomains + di];
    domain_id bsi1 = bsp1T[bi*ndomains + di];

    Vecnd b0v = Vecnd().load(&u1[bsi0][offset]);
    Vecnd b1v = Vecnd().load(&u1[bsi1][offset]);
    Vecnd J = k_onT[bi*ndomains + di]*(totT[bi*ndomains + di]-b0v)*b1v - k_offT[bi*ndomains + di]*b0v;

    if(update_species[bsi0]) {
      (Vecnd().load(&du[bsi0][offset]) + dt*J).store(&du[bsi0][offset]);
    }
    else {
      (b0v + dt*J).store(&du[bsi0][offset]);
    }
    if(update_species[bsi1]) {
      (Vecnd().load(&du[bsi1][offset]) - dt*J).store(&du[bsi1][offset]);
    }
    else {
      (b1v - dt*J).store(&du[bsi1][offset]);
    }
    
    //(Vecnd().load(&du[bsi0][offset]) + dt*J).store(&du[bsi0][offset]);
    //(Vecnd().load(&du[bsi1][offset]) - dt*J).store(&du[bsi1][offset]);
  }
  else {
    for (int zi=0; zi<VECSIZE; zi++) {
      int di = dm_ids[offset + zi];
      domain_id bsi0 = bsp0T[bi*ndomains + di];
      domain_id bsi1 = bsp1T[bi*ndomains + di];
      
      REAL b0 = u1[bsi0][offset + zi];
      REAL b1 = u1[bsi1][offset + zi];
      REAL J = k_onT[bi*ndomains + di]*(totT[bi*ndomains + di]-b0)*b1 - k_offT[bi*ndomains + di]*b0;

      if(update_species[bsi0]) {
	du[bsi0][offset + zi] += dt*J;
      }
      else {
	du[bsi0][offset + zi] = b0 + dt*J;
      }
      
      if(update_species[bsi1]) {
	du[bsi1][offset + zi] -= dt*J;
      }
      else {
	du[bsi1][offset + zi] = b1 - dt*J;
      }
    }
  }

  /*
  for (int zi=0; zi<VECSIZE; zi++) {
    int di = dm_ids[offset + zi];
    domain_id bsi0 = bsp0T[bi*ndomains + di];
    domain_id bsi1 = bsp1T[bi*ndomains + di];
    
    REAL b0 = u1[bsi0][offset + zi];
    REAL b1 = u1[bsi1][offset + zi];
    REAL J = k_onT[bi*ndomains + di]*(totT[bi*ndomains + di]-b0)*b1 - k_offT[bi*ndomains + di]*b0;
    
    // Update species
    du[bsi0][offset + zi] += dt*J;
    du[bsi1][offset + zi] -= dt*J;
  }
  */
}

extern "C" void reaction_SIMD(double **u1, double **du, double * k_onT, double * k_offT, double *totT, domain_id* bsp0T, domain_id* bsp1T, int bi, int offset2, REAL dt, const domain_id* dm_ids, int ndomains, int nz, domain_id* update_species) {
  int zi=0;
  for (zi=0; zi<(nz&-VECSIZE); zi+=VECSIZE) {
    unsigned int offset = offset2 + zi;
    reaction(u1, du, k_onT, k_offT, totT, bsp0T, bsp1T, bi, offset, dt, dm_ids, ndomains, update_species);
  }
  #if defined(SIMDFLAG)
  #pragma simd
  #else
  #pragma ivdep
  #endif
  for (; zi<nz; zi++) {
    unsigned int offset = offset2 + zi;
    int di = dm_ids[offset];
    domain_id bsi0 = bsp0T[bi*ndomains + di];
    domain_id bsi1 = bsp1T[bi*ndomains + di];
    
    REAL b0 = u1[bsi0][offset];
    REAL b1 = u1[bsi1][offset];
    REAL J = k_onT[bi*ndomains + di]*(totT[bi*ndomains + di]-b0)*b1 - k_offT[bi*ndomains + di]*b0;
    

    if(update_species[bsi0]) {
      du[bsi0][offset] += dt*J;
    }
    else {
      du[bsi0][offset] = b0 + dt*J;
    }

    if(update_species[bsi1]) {
      du[bsi1][offset] -= dt*J;
    }
    else {
      du[bsi1][offset] = b1 - dt*J;
    }
  }
}


static inline void kernel_LOTR(double *u1, double *du, double *coef_table, const domain_id *dm_ids, char* stencil_test, int offset, int offset2, int x_shift, int y_shift, int z_shift, Vecndb maska, Vecndb maskm, Vecndb maskp, Vecnd bcast) {
  Vecnd u, um, up, uu, ud, uf, ub, d;

  int offsetm = offset - z_shift, offsetp = offset + z_shift;  // Z direction
  int offsetu = offset - y_shift, offsetd = offset + y_shift;  // Y direction
  int offsetb = offset - x_shift, offsetf = offset + x_shift;  // X direction
  
  u  = Vecnd().load(&u1[offset]);
  um = load(&u1[offsetm], maskm);
  up = load(&u1[offsetp], maskp);
  uu = Vecnd().load(&u1[offsetu]);
  ud = Vecnd().load(&u1[offsetd]);
  ub = Vecnd().load(&u1[offsetb]);
  uf = Vecnd().load(&u1[offsetf]);

#if !defined(GATHERONLY)

  if(!stencil_test[offset2]) {
    Vecnd c(coef_table[dm_ids[offset] + dm_ids[offset]*max_num_domains]);
    //d = mul_add(c, mul_add(bcast, u, um + up + uu + ud + uf + ub), d);
    d = mul_add(c, mul_add(bcast, u, um + up + uu + ud + uf + ub), u);
  }
  else {  
#endif
    Vecnd cm, cp, cu, cd, cf, cb;
    Vecni dicv, dirv;
    dirv = max_num_domains*get_index(&dm_ids[offset]);
    cm = lookupi(dirv + get_index(&dm_ids[offsetm]), coef_table, maskm);
    cp = lookupi(dirv + get_index(&dm_ids[offsetp]), coef_table, maskp);
    cu = lookupi(dirv + get_index(&dm_ids[offsetu]), coef_table, maska);
    cd = lookupi(dirv + get_index(&dm_ids[offsetd]), coef_table, maska);
    cb = lookupi(dirv + get_index(&dm_ids[offsetb]), coef_table, maska);
    cf = lookupi(dirv + get_index(&dm_ids[offsetf]), coef_table, maska);    
    d = mul_add(-u, cm + cp + cu + cd + cf + cb, cm*um + cp*up  + cu*uu + cd*ud + cf*uf  + cb*ub + u);
#if !defined(GATHERONLY)
  }
#endif
  store(&du[offset], maska, d);}

static inline Vecndb make_mask(int dz) {
#if INSTRSET > 8
  return (__mmask8)((1<<dz)-1);
#else
  int64_t t[VECSIZE] = {0};
  for(int i=0; i<dz; i++) t[i] = -1;
  return Vecndb(Vecnd().load((double*)t));
#endif
}

extern "C" void kernel_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5) {
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1;
  int nz2 = (nz-1)&-VECSIZE;
  int dz = nz-nz2;
  
  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      Vecndb maska, maskm, maskp;
      Vecnd bcast;
      int offset = xi*x_shift + yi*y_shift;
      REAL *coef;
      //left

      maska = true , maskm = maska, maskp = maska;
      maskm.insert(0,false);
      bcast = select(maskm, -6.0, -5.0);
      #pragma forceinline recursive
      kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, z_shift, maska, maskm, maskp, bcast);
      coef = coef_table+(dm_ids[offset]*max_num_domains);
      du[offset] += ghosted4 ? -coef[ghost_dm_ids[4][xi*ny + yi]]*u1[offset] : 0;
      //center
      maska = true, maskm = maska, maskp = maska;
      bcast = -6.0;
      for (int zi=VECSIZE;zi<nz2; zi+=VECSIZE) {
	offset = xi*x_shift + yi*y_shift + zi;
        #pragma forceinline recursive
	kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, z_shift, maska, maskm, maskp, bcast);
      }
      //right
      offset = xi*x_shift + yi*y_shift + nz2;
      maska = make_mask(dz), maskm = maska, maskp = make_mask(dz-1) & maska;
      bcast = select(maskp, -6.0, -5.0);
      #pragma forceinline recursive
      kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, z_shift, maska, maskm, maskp, bcast);
      coef = coef_table+(dm_ids[offset+dz-1]*max_num_domains);
      du[offset+dz-1] += ghosted5 ? -coef[ghost_dm_ids[5][xi*ny + yi]]*u1[offset+dz-1] : 0;
    }
  }
}

extern "C" void kernel_alpha_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, REAL *alphax, REAL *alphay, REAL *alphaz) {
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1;
  int nz2 = (nz-1)&-VECSIZE;
  int dz = nz-nz2;

  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      Vecndb maska, maskm, maskp;
      Vecnd bcast;  
      int offset = xi*x_shift + yi*y_shift;
      //left
      REAL *coef;
      maska = true , maskm = maska, maskp = maska;
      maskm.insert(0,false);
      bcast = select(maskm, -6.0, -5.0);
      #pragma forceinline recursive
      kernel_alpha(u1,du, alphax, alphay, alphaz, offset, x_shift, y_shift, maska, maskm, maskp);
      coef = coef_table+(dm_ids[offset]*max_num_domains);
      du[offset] += ghosted4 ? -coef[ghost_dm_ids[4][xi*ny + yi]]*u1[offset] : 0;
      //center
      maska = true, maskm = maska, maskp = maska;
      bcast = -6.0;
      for (int zi=VECSIZE;zi<nz2; zi+=VECSIZE) {
	offset = xi*x_shift + yi*y_shift + zi;
        #pragma forceinline recursive
	kernel_alpha(u1,du, alphax, alphay, alphaz, offset, x_shift, y_shift, maska, maskm, maskp);
      }
      //right
      offset = xi*x_shift + yi*y_shift + nz2;
      maska = make_mask(dz), maskm = maska, maskp = make_mask(dz-1) & maska;
      bcast = select(maskp, -6.0, -5.0);
      #pragma forceinline recursive
      kernel_alpha(u1,du, alphax, alphay, alphaz, offset, x_shift, y_shift, maska, maskm, maskp);
      coef = coef_table+(dm_ids[offset+dz-1]*max_num_domains);
      du[offset+dz-1] += ghosted5 ? -coef[ghost_dm_ids[5][xi*ny + yi]]*u1[offset+dz-1] : 0;
    }
  }
}

