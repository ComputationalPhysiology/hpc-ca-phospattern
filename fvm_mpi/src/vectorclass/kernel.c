#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>
#include "kernel.h"

//#define NX 672
//#define NY 672
//#define NZ 168

#define NX (512+2)
#define NY (512+2)
#define NZ 256

//#define NX 84
//#define NY 84
//#define NZ 42

//#define NX 10
//#define NY 10
//#define NZ 16

#define max_num_domains 5
#define N2 max_num_domains*max_num_domains
#define N3 NX*NY*NZ

#define TSTEP 1

//#define VERSION 3

void set_float(double *a, int n, double k) {
  for(int i=0; i<n; i++) a[i] = k;
}

void foo(double *a, double *b, int n) {
  for(int i=0; i<n; i++) a[i] += b[i];
}

int main(void) {
  //printf("N3 %d\n", N3);
  double *u1 = malloc(sizeof *u1 *N3);
  //double *du = malloc(sizeof *du *N3);
  double *du = _mm_malloc(sizeof *du *N3, 64);
  domain_id *dm_ids = malloc(sizeof *dm_ids *N3);
  char* stencil_test = malloc(sizeof *stencil_test *N3);

  double *alphax = malloc(sizeof *alphax *N3);
  double *alphay = malloc(sizeof *alphay *N3);
  double *alphaz = malloc(sizeof *alphaz *N3);
  
  double *coef_table = malloc(sizeof *coef_table *N2);
  domain_id **ghost_dm_ids = malloc(6*sizeof *ghost_dm_ids);
  ghost_dm_ids[4] = malloc(NX*NY*sizeof *ghost_dm_ids[4]);
  ghost_dm_ids[5] = malloc(NX*NY*sizeof *ghost_dm_ids[5]);

  int nx = NX, ny = NY, nz = NZ;
  unsigned char ghosted4, ghosted5;
  ghosted4 = 1;
  ghosted5 = 1;
  
  set_float(u1, N3, 3.14159);
  set_float(du, N3, 3.14159);

  set_float(alphax, N3, 3.14159);
  set_float(alphay, N3, 3.14159);
  set_float(alphaz, N3, 3.14159);

  set_float(coef_table, N2, 3.14159);

  memset(dm_ids, 0, sizeof *dm_ids *N3);
  for(int i=0; i<NX*NY; i++) ghost_dm_ids[4][i] = 0;
  for(int i=0; i<NX*NY; i++) ghost_dm_ids[5][i] = 0;

  double dtime;
  dtime = -omp_get_wtime();
  for(int i=0; i<TSTEP; i++) {
    #if   VERSION == 1
    kernel_auto(u1, du, coef_table, dm_ids, ghost_dm_ids, stencil_test, nx, ny, nz, ghosted4, ghosted5);
    #elif VERSION == 2
    kernel_SIMD(u1, du, coef_table, dm_ids, ghost_dm_ids, stencil_test, nx, ny, nz, ghosted4, ghosted5);
    #elif VERSION == 3
    kernel_alpha_SIMD(u1, du, coef_table, dm_ids, ghost_dm_ids, stencil_test, nx, ny, nz, ghosted4, ghosted5, alphax, alphay, alphaz);
    #else
    //kernel_block(u1, du, coef_table, dm_ids, ghost_dm_ids, stencil_test, nx, ny, nz, ghosted4, ghosted5);
    kernel_block(u1, du, coef_table, dm_ids, ghost_dm_ids, stencil_test, nx, ny, nz, ghosted4, ghosted5, alphax, alphay, alphaz);
    #endif
  }
  dtime += omp_get_wtime();
  double mem = 8*N3/1E9;
  double bw = TSTEP*mem/dtime;
  printf("dtime %.2f, %.2f GB, 2*bw %.2f, 3*bw %.2f, 9*bw %.2f \n", dtime, mem, 2*bw, 3*bw, 9*bw);
  dtime = -omp_get_wtime();
  memcpy(du, u1, sizeof(double)*N3);
  dtime += omp_get_wtime();
  printf("dtime %.2e, 2bw %.2f \n", dtime, 2*mem/dtime);

  dtime = -omp_get_wtime();
  foo(du, u1, N3);
  dtime += omp_get_wtime();
  printf("dtime %.2e, 3bw %.2f \n", dtime, 3*mem/dtime);

}
