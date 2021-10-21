#include <string.h>
#include <stdio.h>
#include <x86intrin.h>
#include "vectorclass.h"

typedef unsigned char domain_id;
typedef double REAL;

#include "vectorclass_extra.h"

#define max_num_domains 5   // assuming five domains (cyt,cleft,jsr,nsr,tt)

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
  //d  = Vecnd().load(&du[offset]);

  
  cm = load(&alphaz[offsetm], maskm);
  cp = load(&alphaz[offset ], maskp); 
  cu = Vecnd().load(&alphay[offsetu]);
  cd = Vecnd().load(&alphay[offset ]);
  cb = Vecnd().load(&alphax[offsetb]);
  cf = Vecnd().load(&alphax[offset ]);
  
  d = mul_add(-u, cm + cp + cu + cd + cf + cb, cm*um + cp*up  + cu*uu + cd*ud + cf*uf  + cb*ub + d);

  _mm256_stream_pd(&du[offset],d);
  //store(&du[offset], maska, d);
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

  d  = Vecnd().load(&du[offset]);

  Vecnd cm, cp, cu, cd, cf, cb;
  Vecni dicv, dirv;
  dirv = max_num_domains*get_index(&dm_ids[offset]);

  cm = lookupi(dirv + get_index(&dm_ids[offsetm]), coef_table, maskm);
  cp = lookupi(dirv + get_index(&dm_ids[offsetp]), coef_table, maskp);
  cu = lookupi(dirv + get_index(&dm_ids[offsetu]), coef_table, maska);
  cd = lookupi(dirv + get_index(&dm_ids[offsetd]), coef_table, maska);
  cb = lookupi(dirv + get_index(&dm_ids[offsetb]), coef_table, maska);
  cf = lookupi(dirv + get_index(&dm_ids[offsetf]), coef_table, maska);    

  d = mul_add(-u, cm + cp + cu + cd + cf + cb, cm*um + cp*up  + cu*uu + cd*ud + cf*uf  + cb*ub + d);

  _mm256_stream_pd(&du[offset],d);
  //store(&du[offset], maska, d);
}

static inline Vecndb make_mask(int dz) {
#if INSTRSET > 8
  return (__mmask8)((1<<dz)-1);
#else
  int64_t t[VECSIZE] = {0};
  for(int i=0; i<dz; i++) t[i] = -1;
  return Vecndb(Vecnd().load((double*)t));
#endif
}

extern "C" void kernel_SIMD2(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5) {
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
      kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, z_shift, maska, maskm, maskp, bcast);
      coef = coef_table+(dm_ids[offset]*max_num_domains);
      du[offset] += ghosted4 ? -coef[ghost_dm_ids[4][xi*ny + yi]]*u1[offset] : 0;
      //center
      maska = true, maskm = maska, maskp = maska;
      bcast = -6.0;
      for (int zi=VECSIZE; zi<nz2; zi+=VECSIZE) {
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

extern "C" void kernel_alpha_SIMD2(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, REAL *alphax, REAL *alphay, REAL *alphaz) {
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
      for (int zi=VECSIZE; zi<nz2; zi+=VECSIZE) {
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

extern "C" void kernel_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5) {
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1;
  Vecndb maska = true, maskm = maska, maskp = maska;
  Vecnd bcast = -6.0;
  
  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      for (int zi=0; zi<nz; zi+=VECSIZE) {
	int offset = xi*x_shift + yi*y_shift + zi;
        #pragma forceinline recursive
	kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, z_shift, maska, maskm, maskp, bcast);
      }
    }
  }
}

void kernel_block2(double *u1, double *du, double *coef_table, const domain_id *dm_ids, char* stencil_test, int nx, int ny, int nz, int offset, int x_shift, int y_shift, REAL *alphax, REAL *alphay, REAL *alphaz) {
  Vecndb maska = true, maskm = maska, maskp = maska;
  Vecnd bcast = -6.0;
  for(int i=0; i<nx; i++) {
    for(int j=0; j<ny; j++) {
      for(int k=0; k<nz; k+=VECSIZE) {
	int offset = i*x_shift + j*y_shift + k;
	//int offset = i*ny*nz + j*nz + k;
	//int offset = i*ny*nz + j*nz + k;
	//offset = 1;
	//kernel_LOTR(u1, du, coef_table, dm_ids, stencil_test, offset, offset, x_shift, y_shift, 1 , maska, maskm, maskp, bcast);
	kernel_alpha(u1,du, alphax, alphay, alphaz, offset, x_shift, y_shift, maska, maskm, maskp);
      }
    }
  }
}

extern "C" void kernel_block(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, REAL *alphax, REAL *alphay, REAL *alphaz) {
  //int nbx = (nx-2)/64, nby = (ny-2)/64, nbz = nz/32;
  int nbx = (nx-2)/1, nby = (ny-2)/1, nbz = nz/1;
  //printf("blocks %d %d %d\n", nbx, nby, nbz);
  int x_shift = nz*ny, y_shift = nz;

  for (int xi=1; xi<nx-1; xi += nbx) {
  for (int yi=1; yi<ny-1; yi += nby) {
  for (int zi=0; zi<nz;   zi += nbz) {

    int offset = xi*x_shift + yi*y_shift + zi;
    //printf("offset %d\n", offset);
    int nx2 = nx-xi >= nbx ? nbx : nx-xi;
    int ny2 = ny-yi >= nby ? nby : ny-yi;
    int nz2 = nz-zi >= nbz ? nbz : nz-zi;
    //printf("\tblocks %d %d %d\n", xi, yi, zi);
    //#pragma forceinline recursive
    kernel_block2(&u1[offset], &du[offset], coef_table, dm_ids, stencil_test, nx2, ny2, nz2, offset, x_shift, y_shift, alphax, alphay, alphaz);

}
  }
  }  

}


extern "C" void kernel_alpha_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, REAL *alphax, REAL *alphay, REAL *alphaz) {
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1;
  
  Vecndb maska = true, maskm = maska, maskp = maska;
  Vecnd bcast = -6.0;
  
  #pragma omp parallel for collapse(2)
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      for (int zi=0; zi<nz; zi+=VECSIZE) {
	int offset = xi*x_shift + yi*y_shift + zi;
        #pragma forceinline recursive
	kernel_alpha(u1,du, alphax, alphay, alphaz, offset, x_shift, y_shift, maska, maskm, maskp);
      }
    }
  }
}

extern "C" void kernel_auto(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5) {
  const unsigned int x_shift = ny*nz, y_shift = nz, z_shift = 1;

  #pragma omp parallel for collapse(2)  
  for (int xi=1; xi<nx-1; xi++) {
    for (int yi=1; yi<ny-1; yi++) {
      int offset, offset_m, offset_p;
      domain_id di, di_n;
      double u;
      #pragma ivdep
      for (int zi=1;zi<nz-1; zi++) {
	offset = xi*x_shift + yi*y_shift + zi;
	double *coef = coef_table+(dm_ids[offset]*max_num_domains);
	u = u1[offset];
	// Z direction
	offset_m = offset-1; di = dm_ids[offset_m];
	offset_p = offset+1; di_n = dm_ids[offset_p];
	du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
	// Y direction
	offset_m = offset-y_shift; di = dm_ids[offset_m];
	offset_p = offset+y_shift; di_n = dm_ids[offset_p];
	du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
	// X direction
	offset_m = offset-x_shift; di = dm_ids[offset_m];
	offset_p = offset+x_shift; di_n = dm_ids[offset_p];
	du[offset] += coef[di]*(u1[offset_m]-u) + coef[di_n]*(u1[offset_p]-u);
      }
    }
  }  
}
