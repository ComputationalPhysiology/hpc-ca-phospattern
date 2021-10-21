#include <math.h>
#include <assert.h>
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "mpi.h"
#include "hdf5.h"
#include <sys/stat.h> 

//#define DEBUG_TEST
#define DB_PF 0
#define MAX_LINE_LENGTH 80
//#define __PAPI__

#ifdef   __PAPI__
#include <papi.h>
#endif

typedef int(*CONDCF)(int a, int b);

#define H5T_DATA_TYPE H5T_NATIVE_SHORT
typedef short int hdf5_data_type;
#define H5_DATA_LIMIT_0 -32768 // Data type specific
#define H5_DATA_LIMIT_1 32767  // Data type specific
#define H5_DATA_SIZE H5_DATA_LIMIT_1 - H5_DATA_LIMIT_0 // Data type specific

double timing();
void *mpi_malloc ( int id, int bytes);  /* IN - Bytes to allocate */
inline double my_random();
double my_min(double* ar, int len);
double my_max(double* ar, int len);
void stern(double t, double* y0, double* y1, double Ca);
void stern_discrete(double dt, int* y0, int* y1, double Ca);

void laplace3D (int nx0, int ny0, int nz0, double* C0,
                int nx1, int ny1, int nz1, double* C1,
                double alpha);//, int num_threads)

void reaction3D (int nx0, int ny0, int nz0, double* Ca,
                 int nx1, int ny1, int nz1, double* buff,
                 double B_tot, double k_on, double k_off, double dt);//, int num_threads)

void serca3D (int nx0, int ny0, int nz0, double* Ca_i,
              int nx1, int ny1, int nz1, double* Ca_SR,
              double dt, double gamma, double fudge);//, int num_threads)

void update_ryr(int h_scale,int nx0, int ny0, int nz0, double* Ca_i, double* Ca_SR, double* Ca_CSQN, 
                double* C10, double* C12, double* C13, double* C14,
		double k_on_CSQN, double k_off_CSQN, double CSQN_tot,
		double gamma, double K, double dt,
		int ryr_len, int* i0_ryr, int* i1_ryr, int* i2_ryr,
		int csqn_len, int* i0_csqn, int* i1_csqn, int* i2_csqn,
		int cleft_len, int* i0_cleft, int* i1_cleft, int* i2_cleft,int* cleft_nb,
		int* states0, int* states1);

void store2Dmatrixfile_double_1D(char* outfile, double* ar, int rows, int cols, int x_strid);
void store2Dmatrixfile_double_bin(char* outfile, double* ar, int rows, int cols, int x_strid);
void transfer_hdf5_data(hdf5_data_type* h5_data, double* ar0, double* ar1, 
			double scale_value, hsize_t* chunk_dims);
void store2Dmatrixfile_int_1D(char* outfile, int* ar, int rows, int cols);

//int   less(int a, int b);
//int   giant(int a, int b);
//int*  loadRyRindexfile_int(char* infile, CONDFN cf, int cond);
int*    loadRyRindexfile_int(char* infile, int* count);

int idxinrank(int nx, int ny, int nz,
              int i0, int i1, int i2,
              int rank, MPI_Comm comm3d);

int idxbl2rank(int nx, int ny, int nz,
               int i0, int i1, int i2,
	       int* coords,
               MPI_Comm comm3d);

int load_indices_serial(int nx, int ny, int nz, int h, 
			int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
			int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
			int** i0_cleft, int** i1_cleft, int** i2_cleft, int** cleft_nb, int* cleft_len,
			int x_slice_mid,int x_slice_width, int x_slice_num, int use_failing);

 
int IsTrueCleft(int coord_y, int coord_z, int size_y, int size_z, int *i1_csqn, int *i2_csqn, int* y_index, int csqn_len);
void BinarySort_two(int* pData, int* vData, int Count);
void dichotomy_two(int* pData,int* vData, int left,int right);

int distr_ryr_csqn_state(int h, int size_x, int size_y, int size_z,
			 int nx, int ny, int nz, 
			 int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
			 int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
			 int** i0_cleft, int** i1_cleft, int** i2_cleft,int** cleft_nb, int* cleft_len,
			 int** states0, int** states1,
			 int x_slice_mid,int x_slice_width, int x_slice_num,
			 MPI_Comm comm3d, MPI_Comm, int use_failing);

void readparam(int* iconf, double* conf);

void updateBound(double* C00, double* C01, double* C02, double* C03, double* C04, 
    int C_flag, int nx0, int ny0, int nz0,
    double* yz_sbuf0,double* yz_rbuf0,
    double* xz_sbuf0,double* xz_rbuf0,
    double* xy_sbuf0,double* xy_rbuf0,
    double* yz_sbuf1,double* yz_rbuf1,
    double* xz_sbuf1,double* xz_rbuf1,
    double* xy_sbuf1,double* xy_rbuf1,
    int* neighbor, MPI_Status* ar_status, MPI_Request* ar_send_req, MPI_Request* ar_recv_req,
    MPI_Comm comm,MPI_Comm comm3d);

void putin_sendbuffer_yz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);
void putin_sendbuffer_xz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);
void putin_sendbuffer_xy(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);
void getout_recvbuffer_yz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);
void getout_recvbuffer_xz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);
void getout_recvbuffer_xy(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len);

void compute_pde_ode(int nx0, int ny0, int nz0, double dt,double gamma, double fudge,
    double* alpha, double* B_tot, double* k_on, double* k_off,
    double** C0, double** C1, int div_y);

#define NUM_SAVE_SPECIES 5
int save_species[NUM_SAVE_SPECIES] = {0,1,4,5,6};
char* species_names[7] = {"Cai", "CaSR", "CaCMDN", "CaATP", "CaFluo", "CaTRPN", "CaCSQN"};

int main(int argc, char **argv)
{ 

  int i,j,k;
#ifdef __PAPI__
  //  int Events[] = { PAPI_L1_DCA, PAPI_L1_DCM  };
  //  int Events[] = {PAPI_L3_TCM, PAPI_L3_TCA, PAPI_L2_TCM,PAPI_L2_TCA};
  int Events[] = {PAPI_DP_OPS,PAPI_L3_TCM};
  int NUM_EVENTS = sizeof(Events)/sizeof(Events[0]);
  long long res_papi[NUM_EVENTS];
  char EventName[128];
  int num_hwcntrs = 0;
  int EventSet = PAPI_NULL;
  int retval;

  retval = PAPI_library_init( PAPI_VER_CURRENT );
  retval = PAPI_create_eventset( &EventSet );

  if (PAPI_add_events( EventSet, Events, NUM_EVENTS) != PAPI_OK){
    printf("PAPI_add_events failed\n");
  }

  for (i=0; i<NUM_EVENTS; i++){
    res_papi[i] = 0;
  }
#endif

  double time_main=0.0;
  double time_comm=0.0;
  double time_conc=0.0;
  double time_ryr=0.0; 
  double time_io=0.0; 
 
  int save_data=0;
  int use_rand_seed=1;
  int use_failing=0;
  int idx;
  int h_scale=1;
  int h=30;
  int div_y=1;
  int save_binary_file=0;
  int save_hdf5=0;

  double T=1.0;
  double DT=0.05; // plotting time step
  int TimeStep=2;
  int size_x, size_y, size_z, my_id, x_domains, y_domains, z_domains;
  int iconf[12];
  double conf[2];
  /* MPI variables */
  int nproc, ndims;
  MPI_Comm comm, comm3d;
  int dims[3];
  int periods[3];
  int reorganisation = 0;
  MPI_Datatype matrix_type_oyz, matrix_type_oxz, matrix_type_oxy;
  int ZN=0, ZP=1, YN=2, YP=3, XN=4, XP=5;
  int NeighBor[6];
  hid_t h5_file_id;

  hdf5_data_type* h5_data;

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &my_id);
  MPI_Info info = MPI_INFO_NULL;
  
  if (my_id==0) {
    readparam(iconf, conf);
  }

  MPI_Bcast(iconf, 12, MPI_INT, 0, comm);
  MPI_Bcast(conf, 2, MPI_DOUBLE, 0, comm);
  
  h  = iconf[0];
  size_x = iconf[1];
  size_y = iconf[2];
  size_z = iconf[3];  
  x_domains = iconf[4];
  y_domains = iconf[5];
  z_domains = iconf[6];  
  save_data = iconf[7];  
  use_failing = iconf[8];  
  save_binary_file = iconf[9]; // Save Ca in binary file instead of ascii file 
  save_hdf5 = iconf[10];       // Save data in hdf5 file format
  div_y = iconf[11];           // Block size on y direction for cache
  T  = conf[0];
  DT = conf[1];
  h_scale=30/h;

  if(use_rand_seed) srand(my_id);

  char hdf5_dataset_name[200];
  char hdf5_group_name[200];
  char h5_basename[200];
  char outdirname[200];
  if(save_hdf5)
  {
    sprintf(h5_basename, "output_%d_%d_%d_%d_%d", h, size_x, size_y, size_z, use_failing);
  }
  else if(save_binary_file)
  {
    sprintf(outdirname, "output_%d_%d_%d_%d_%d_bin", h, size_x, size_y, size_z, use_failing);
  }
  else
  {
    sprintf(outdirname, "output_%d_%d_%d_%d_%d", h, size_x, size_y, size_z, use_failing);
  }

  if(!my_id)
  {
    if(save_data && !save_hdf5){
      if(access(outdirname,0))
      {
	if (mkdir(outdirname, 0755)==-1)
	{
	  printf("make directory failed\n");
	}
	else
	{
	  printf("make directory: %s\n", outdirname);
	}
      }
      else
      {
	printf("directory %s existed\n",outdirname);
      }
    }
  }
  MPI_Barrier(comm);

  if((my_id==0) && (nproc!=(x_domains*y_domains*z_domains))) {
    printf("Number of processes not equal to Number of subdomains\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if((my_id==0)&&(size_x%x_domains!=0)) {
    printf("Number of x_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if((my_id==0)&&(size_y%y_domains!=0)) {
    printf("Number of y_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if((my_id==0)&&(size_z%z_domains!=0)) {
    printf("Number of z_domains is not divisible in scale\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(((size_y/y_domains)%div_y)!=0){
    div_y=1;
    if(my_id==0){
      printf("Warning: div_y is not divisible on each node, so set div_y=1 for default \n");
    }
  }
  
  /* Create 3D cartesian grid */

  periods[0] = 0;
  periods[1] = 0;
  periods[2] = 0;  

  ndims = 3;
  dims[0]=z_domains;
  dims[1]=y_domains;
  dims[2]=x_domains;  

  MPI_Cart_create(comm, ndims, dims, periods, reorganisation, &comm3d);

 /* MPI variables */
  MPI_Status  ar_status[6];
  MPI_Request ar_send_req[6];
  MPI_Request ar_recv_req[6];
  int  coord[3];
  int  dim[3];
  int  period[3];
  int mid_coord_x=0;
  int in_midx_slice=0;
  int x_slice_num;
  int x_slice_width;
  int x_slice_mid;
  MPI_Cart_get(comm3d, 3, dim, period, coord);
  x_slice_num=(int)(ceil((double)(size_x*h)/2100.0));
  if((size_x%x_slice_num)!=0) 
  {
    printf("x dimension can not be divided by %d\n", x_slice_num);
    MPI_Abort(comm,5);
  }

  x_slice_width=size_x/x_slice_num;
  x_slice_mid=(x_slice_width+1)/2;
  for(i=0;i<x_slice_num;i++)
  {
    if(((x_slice_width*i+x_slice_mid)>=(coord[2]*size_x/x_domains))&&
       ((x_slice_width*i+x_slice_mid)<((coord[2]+1)*size_x/x_domains))){
      if(in_midx_slice==1){
	printf("dont put two x_slice in a x partition\n");
	MPI_Abort(comm,5);
      }
      in_midx_slice=1;
      mid_coord_x=(x_slice_width*i+x_slice_mid)-(coord[2]*size_x/x_domains)+1;//+1 for ghost bound
      //check x partition thickness, so far, for simplify, dont cut a csqn and no-flux into two x-partitions 
      if((mid_coord_x)<(h_scale+3)||(size_x/x_domains-mid_coord_x)<(h_scale+3)){
	printf("x partition is too thine for CSQN and cleft extend \n");
	MPI_Abort(comm,5);
      }
    }
  }

  //printf("Rank: %d, coord: [%d, %d, %d]\n", my_id, coord[0], coord[1], coord[2]);

  /* Identify process neighbors */
  NeighBor[0] = MPI_PROC_NULL;
  NeighBor[1] = MPI_PROC_NULL;
  NeighBor[2] = MPI_PROC_NULL;
  NeighBor[3] = MPI_PROC_NULL;
  NeighBor[4] = MPI_PROC_NULL;  
  NeighBor[5] = MPI_PROC_NULL;    

  /* Left/West and right/Est neigbors Z direction*/
  MPI_Cart_shift(comm3d,0,1,&NeighBor[ZN],&NeighBor[ZP]);

  /* Bottom/South and Upper/North neigbors Y direction*/
  MPI_Cart_shift(comm3d,1,1,&NeighBor[YN],&NeighBor[YP]);

  /* Zdown/South and Zup/North neigbors X direction*/
  MPI_Cart_shift(comm3d,2,1,&NeighBor[XN],&NeighBor[XP]);

  //--------------------------------------------------------------------
  int nx=(size_x/x_domains);
  int ny=(size_y/y_domains);
  int nz=(size_z/z_domains);  
  int nx0, ny0, nz0;
  int nx1, ny1, nz1;
  nx0=nx+2; 
  ny0=ny+2; 
  nz0=nz+2; 
  nx1=nx+2; 
  ny1=ny+2; 
  nz1=nz+2; 
  int len;
  len=nx0*ny0*nz0;

  /* Create matrix data types to communicate */
  MPI_Type_vector(ny, nz, nz0, MPI_DOUBLE, &matrix_type_oyz);
  MPI_Type_commit(&matrix_type_oyz);

  /* Create matrix data type to communicate on vertical Oxz plan */
  MPI_Type_vector(nx, nz, ny0*nz0, MPI_DOUBLE, &matrix_type_oxz);
  MPI_Type_commit(&matrix_type_oxz);

  /* Create matrix data type to communicate on vertical Oxy plan */
  MPI_Datatype matrix_type_liney; 
  MPI_Type_vector(ny, 1, nz0, MPI_DOUBLE, &matrix_type_liney);
  MPI_Type_commit(&matrix_type_liney);

  //  MPI_Type_vector(nx*ny, 1, nz0, MPI_DOUBLE, &matrix_type_oxy);
  MPI_Type_hvector(nx, 1, ny0*nz0*sizeof(double), matrix_type_liney, &matrix_type_oxy);
  MPI_Type_commit(&matrix_type_oxy);

  if(!my_id)
    printf("Simulation Begin!\n");

  //Define where the RyRs are:
  int* i0_ryr;
  int* i1_ryr;
  int* i2_ryr;
  int* i0_csqn;
  int* i1_csqn;
  int* i2_csqn;
  int* i0_cleft;
  int* i1_cleft;
  int* i2_cleft;
  int* cleft_nb;
  int  ryr_len;
  int  csqn_len;
  int  cleft_len;
  int* states0;
  int* states1;

  h_scale=distr_ryr_csqn_state( h,  size_x,  size_y,  size_z, nx,  ny,  nz, 
				&i0_ryr,  &i1_ryr,  &i2_ryr,  &ryr_len,
				&i0_csqn, &i1_csqn, &i2_csqn, &csqn_len,
				&i0_cleft, &i1_cleft, &i2_cleft, &cleft_nb,&cleft_len,
				&states0, &states1, 
				x_slice_mid,x_slice_width, x_slice_num,
				comm3d, comm, use_failing);

  // store2Dmatrixfile_int_1D("i0.txt",i0,n_ryr,1);
  // store2Dmatrixfile_int_1D("i1.txt",i1,n_ryr,1);
  // store2Dmatrixfile_int_1D("i2.txt",i2,n_ryr,1);

  double Vfraction;
  //first set the numbers of RyR in a CaRU;
  //All CaRU placed mid-sarcomere
  Vfraction=(30.0/h)*(30.0/h)*(30.0/h); // scaling of RyR when changing dx
  // Set constants and dt based on these:
  double D_i=250e3; // 220e3
  double D_SR=73e3; // 73.3e3;
  double D_ATP=140e3;
  double D_CMDN=22e3;
  double D_Fluo=42e3;

  double dt=(1./6)*h*h/D_i;

  double alpha_i = dt*D_i/(h*h);
  double Ca0 = 140e-3;
  double CaSR0 = 1.3e3;
  double* Ca_i;
  Ca_i=(double*)malloc(len*sizeof(double));
  for ( i = 0; i < len; i += 1 ) {
    Ca_i[i]=Ca0;
  } 

  double alpha_SR = dt*D_SR/(h*h);
  double* Ca_SR;
  Ca_SR=(double*)malloc(len*sizeof(double));
  for ( i = 0; i < len; i += 1 ) {
    Ca_SR[i]=CaSR0;
  } 

  double k_on_CMDN = 34e-3;
  double k_off_CMDN = 238e-3;
  double CMDN_tot = 24;
  double alpha_CMDN = dt*D_CMDN/(h*h);

  double k_on_ATP = 255e-3;
  double k_off_ATP = 45;
  double ATP_tot = 455;
  double alpha_ATP = dt*D_ATP/(h*h);

  double k_on_Fluo = 110e-3;
  double k_off_Fluo = 110e-3;
  double Fluo_tot = 25; // 25;
  double alpha_Fluo = dt*D_Fluo/(h*h);

  double k_on_TRPN = 32.7e-3;
  double k_off_TRPN = 19.6e-3; // 26.16e-3;
  double TRPN_tot = 70; // 50;

  double k_on_CSQN = 102e-3;
  double k_off_CSQN = 65;
  double CSQN_tot = 30e3;

  double alpha[7];
  double k_on[7];
  double k_off[7];
  double B_tot[7];

  alpha[0]=alpha_i; alpha[1]=alpha_SR; alpha[2]=alpha_CMDN; alpha[3]=alpha_ATP; alpha[4]=alpha_Fluo; alpha[5]=0;          alpha[6]=0;
  k_on[0]=0      ;  k_on[1]=       0;  k_on[2]= k_on_CMDN;  k_on[3]=k_on_ATP ;  k_on[4]=k_on_Fluo ;  k_on[5]=k_on_TRPN;   k_on[6]=k_on_CSQN;
  k_off[0]=0      ; k_off[1]=       0; k_off[2]=k_off_CMDN; k_off[3]=k_off_ATP; k_off[4]=k_off_Fluo; k_off[5]=k_off_TRPN; k_off[6]=k_off_CSQN;
  B_tot[0]=0      ; B_tot[1]=       0; B_tot[2]=CMDN_tot  ; B_tot[3]=ATP_tot  ; B_tot[4]=Fluo_tot  ; B_tot[5]=TRPN_tot;   B_tot[6]=CSQN_tot;


  // Calculate steady state IC for the buffers based on Ca_i ...
  double Ca_CMDN0=B_tot[2]*Ca0/(Ca0+k_off[2]/k_on[2]);
  double Ca_ATP0 =B_tot[3]*Ca0/(Ca0+k_off[3]/k_on[3]);
  double Ca_Fluo0=B_tot[4]*Ca0/(Ca0+k_off[4]/k_on[4]);
  double Ca_TRPN0=B_tot[5]*Ca0/(Ca0+k_off[5]/k_on[5]);

  // and Ca_SR:
  double Ca_CSQN0 = CSQN_tot*Ca_SR[0]/(Ca_SR[0] +  k_off_CSQN/k_on_CSQN);

  double init_values[7] = {Ca0, CaSR0, Ca_CMDN0, Ca_ATP0, Ca_Fluo0, Ca_TRPN0, Ca_CSQN0};

  //printf("%f %f %f %f %f \n ", Ca_ATP0, Ca_CMDN0, Ca_Fluo0, Ca_TRPN0, Ca_CSQN0);
  if(my_id==0)
    printf("cubiod_c: h:%d size_x:%d size_y:%d size_z:%d dt:%f, T:%f, TimeStep:%d, DT:%f outfilenum:%d, x_slice_num:%d, use_failing:%d, div_y:%d, save_binary:%d \n", 
	h, size_x, size_y, size_z,dt,T,
	(int)(T/dt),DT,(int)(T/DT)*save_data,x_slice_num,use_failing,
	div_y,save_binary_file); 

  // Allocate the data structure for the solution
  double *Ca_ATP  ;
  double *Ca_CMDN ; 
  double *Ca_Fluo ; 
  double *Ca_TRPN ; 
  double *Ca_CSQN ; 

  Ca_ATP =(double*)malloc(len*sizeof(double));
  Ca_CMDN=(double*)malloc(len*sizeof(double));
  Ca_Fluo=(double*)malloc(len*sizeof(double));
  Ca_TRPN=(double*)malloc(len*sizeof(double));
  Ca_CSQN=(double*)malloc(len*sizeof(double));

  for ( i = 0; i < len; i += 1 ) {
    Ca_ATP[i] = Ca_ATP0;
    Ca_CMDN[i] = Ca_CMDN0;
    Ca_Fluo[i] = Ca_Fluo0;
    Ca_TRPN[i] = Ca_TRPN0;
    Ca_CSQN[i] = Ca_CSQN0;
  }

  double* C0[7];
  double* C1[7];
  double* C_temp;

  C0[0]=(double*)malloc(len*sizeof(double));
  C1[0]=Ca_i;
  memcpy(C0[0],C1[0],len*sizeof(double));

  C0[1]=(double*)malloc(len*sizeof(double));
  C1[1]=Ca_SR;
  memcpy(C0[1],C1[1],len*sizeof(double));

  C0[2]=(double*)malloc(len*sizeof(double));
  C1[2]=Ca_CMDN;
  memcpy(C0[2],C1[2],len*sizeof(double));

  C0[3]=(double*)malloc(len*sizeof(double));
  C1[3]=Ca_ATP;
  memcpy(C0[3],C1[3],len*sizeof(double));

  C0[4]=(double*)malloc(len*sizeof(double));
  C1[4]=Ca_Fluo;
  memcpy(C0[4],C1[4],len*sizeof(double));

  C0[5]=(double*)malloc(len*sizeof(double));
  C1[5]=Ca_TRPN;
  memcpy(C0[5],C1[5],len*sizeof(double));

  C0[6]=(double*)malloc(len*sizeof(double));
  C1[6]=Ca_CSQN;
  memcpy(C0[6],C1[6],len*sizeof(double));

  //Ca = [[Ca_i.copy(),    Ca_i   ],
  //      [Ca_SR.copy(),   Ca_SR  ],
  //      [Ca_CMDN.copy(), Ca_CMDN],
  //      [Ca_ATP.copy(),  Ca_ATP ],
  //      [Ca_Fluo.copy(), Ca_Fluo],
  //      [Ca_TRPN, Ca_TRPN],
  //      [Ca_CSQN, Ca_CSQN]]

  double gamma = 0.02; // SR volume fraction
  int   cai=0;
  int   sri=1;
//  int cmdni=2;
//  int  atpi=3;
//  int fluoi=4;
//  int trpni=5; 
  int csqni=6;
  double fraction[7]={1,1,1,1,1,1,1};
  fraction[1]=gamma;
  fraction[6]=gamma;

  // Ryr conductance:
  double k_s = (Vfraction)*150/2; // 1/ms, based on 0.5pA of Ca2+ into (30nm)^3.
  double K = exp(-k_s*dt*(1+1/gamma)); // factor need in the integration below

  if(my_id==0){
   printf("dt = dt: %e\n", dt);
    printf("k_s = (Vfraction)*150/2: %e\n", k_s);
    printf("K = exp(-k_s*dt*(1+1/gamma)): %e\n", K);
  }
  double t=0;
  int counter=0;
//  int mean[7];

  time_main-=timing();

  FILE *fpdata; 
  char meanfile[200];
  if (save_hdf5)
    sprintf(meanfile,"%s_mean.txt", h5_basename);
  else
    sprintf(meanfile,"%s/mean.txt", outdirname);
  if(!my_id){
    if(save_data){
      if ((fpdata=fopen(meanfile, "w"))==NULL)
      {
	printf("failed open output file ");
	printf("%s", meanfile);
	printf(" ! \n ");
	exit(0);
      }
    }
  }

  // H5 Setup
  if (save_hdf5)
  {
  
    char h5_data_file[200];

    // Set up file access property list with parallel I/O access
    // property list identifier
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    
    sprintf(h5_data_file, "%s.h5", h5_basename);
    
    // Create a new file collectively and release property list identifier.
    h5_file_id = H5Fcreate(h5_data_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    
    const int data_rank = 2;
    hsize_t dimsf[2] = {size_y, size_z};         /* dataset dimensions */
    hsize_t chunk_dims[2] = {ny, nz};            /* chunk dimensions */
    
    // Offset into dataset based on the MPI coord from MPI_Cart_get
    hsize_t h5_offset[2] = {coord[1]*nz, coord[0]*ny};
    hsize_t h5_count[2] = {1, 1};
    hsize_t data_size=ny*nz;
    h5_data = (hdf5_data_type*)malloc(data_size*sizeof(hdf5_data_type));

    if (!my_id)
    {
      printf("Total data size per species: %zu, %zu\n", dimsf[0], dimsf[1]);
      printf("Total data size per chunk per species: %zu, %zu\n", chunk_dims[0], chunk_dims[1]);
    }  

    printf("rank %d | h5 offset [%zu, %zu]\n", my_id, h5_offset[0], h5_offset[1]);

    // Create data space for the datatype limits
    hsize_t dims = 1;
    hid_t attr_space = H5Screate_simple(1, &dims, NULL);
    
    // Create a time attribute
    hid_t limit_id = H5Acreate(h5_file_id, "data_type_size", H5T_NATIVE_DOUBLE, 
			       attr_space, H5P_DEFAULT, H5P_DEFAULT);

    // Write the attribute data
    double data_type_size = (double)H5_DATA_SIZE;
    herr_t status = H5Awrite(limit_id, H5T_NATIVE_DOUBLE, &data_type_size);
    
    // Cleanup
    H5Aclose(limit_id);
    H5Sclose(attr_space);

    // Save hard coded data ranges 
    for (i=0; i<NUM_SAVE_SPECIES; i++)
    {
      
      // Get species
      int species = save_species[i];

      // Create data scale attribute
      sprintf(hdf5_dataset_name, "%s_scale", species_names[species]);

      // Create data space for the species scale attribute
      hsize_t dims = 1;
      hid_t attr_space = H5Screate_simple(1, &dims, NULL);
    
      // Create a time attribute
      hid_t scale_id = H5Acreate(h5_file_id, hdf5_dataset_name, H5T_NATIVE_DOUBLE, 
				 attr_space, H5P_DEFAULT, H5P_DEFAULT);

      // Write the attribute data
      herr_t status = H5Awrite(scale_id, H5T_NATIVE_DOUBLE, &init_values[species]);
    
      // Cleanup
      H5Aclose(scale_id);
      H5Sclose(attr_space);

      // Create init value attribute
      sprintf(hdf5_dataset_name, "%s_init", species_names[species]);

      // Create data space for the species init attribute
      dims = 1;
      attr_space = H5Screate_simple(1, &dims, NULL);
    
      // Create a time attribute
      hid_t init_id = H5Acreate(h5_file_id, hdf5_dataset_name, H5T_NATIVE_DOUBLE, 
				attr_space, H5P_DEFAULT, H5P_DEFAULT);

      // Write the attribute data
      status = H5Awrite(init_id, H5T_NATIVE_DOUBLE, &init_values[species]);
    
      // Cleanup
      H5Aclose(init_id);
      H5Sclose(attr_space);
    }

  }


double* yz_sbuf0; double* yz_rbuf0;
double* xz_sbuf0; double* xz_rbuf0;
double* xy_sbuf0; double* xy_rbuf0;
double* yz_sbuf1; double* yz_rbuf1;
double* xz_sbuf1; double* xz_rbuf1;
double* xy_sbuf1; double* xy_rbuf1;

yz_sbuf0=(double*)mpi_malloc(my_id,5*ny*nz*sizeof(double));
xz_sbuf0=(double*)mpi_malloc(my_id,5*nx*nz*sizeof(double));
xy_sbuf0=(double*)mpi_malloc(my_id,5*nx*ny*sizeof(double));

yz_sbuf1=(double*)mpi_malloc(my_id,5*ny*nz*sizeof(double));
xz_sbuf1=(double*)mpi_malloc(my_id,5*nx*nz*sizeof(double));
xy_sbuf1=(double*)mpi_malloc(my_id,5*nx*ny*sizeof(double));

yz_rbuf0=(double*)mpi_malloc(my_id,5*ny*nz*sizeof(double));
xz_rbuf0=(double*)mpi_malloc(my_id,5*nx*nz*sizeof(double));
xy_rbuf0=(double*)mpi_malloc(my_id,5*nx*ny*sizeof(double));

yz_rbuf1=(double*)mpi_malloc(my_id,5*ny*nz*sizeof(double));
xz_rbuf1=(double*)mpi_malloc(my_id,5*nx*nz*sizeof(double));
xy_rbuf1=(double*)mpi_malloc(my_id,5*nx*ny*sizeof(double));

#ifdef __PAPI__
  if ( PAPI_start( EventSet ) != PAPI_OK){
    printf("PAPI_read_counters failed\n");
  }
#endif
  //settime
  //T=1000*dt;
  //for ( T = 0; T < TimeStep; T += 1 ) 
  int t_counter=0;
  while(t<T)
    //while(0)
  {
    t+=dt;
    t_counter++;
    time_comm-=timing();
    updateBound(C0[0], C0[1], C0[2], C0[3], C0[4],
      t_counter, nx0, ny0, nz0,
      yz_sbuf0,yz_rbuf0, xz_sbuf0,xz_rbuf0, xy_sbuf0,xy_rbuf0,
      yz_sbuf1,yz_rbuf1, xz_sbuf1,xz_rbuf1, xy_sbuf1,xy_rbuf1,
      NeighBor, ar_status,ar_send_req,ar_recv_req,
      comm, comm3d);
    time_comm+=timing();

    // Diffusion update
    time_conc-=timing();
    // Change to use a faster computing function
    compute_pde_ode(nx0, ny0, nz0, dt, gamma, 1e-4,
	alpha,  B_tot, k_on, k_off,
	C0, C1, div_y);
//    for ( i = 0; i < 5; i += 1 ) {
//      laplace3D(nx0,ny0,nz0,C0[i],nx1,ny1,nz1,C1[i],alpha[i]);
//    }
//    for ( i = 2; i < 6; i += 1 ) {
//      reaction3D(nx1,ny1,nz1,C1[cai],nx1,ny1,nz1,C1[i],B_tot[i],k_on[i],k_off[i],dt);
//    }
//    serca3D(nx1,ny1,nz1, C1[cai],nx1,ny1,nz1, C1[sri], dt, gamma, 1.0);
    time_conc+=timing();

    // Update at RyRs, one at the time
    time_ryr-=timing();
    update_ryr(h_scale, nx0, ny0, nz0, C1[cai], C1[sri], C1[csqni], 
	       C1[0],C1[2],C1[3],C1[4],
	       k_on_CSQN, k_off_CSQN,CSQN_tot,
	       gamma, K, dt,
	       ryr_len, i0_ryr, i1_ryr, i2_ryr, 
	       csqn_len, i0_csqn, i1_csqn, i2_csqn, 
	       cleft_len, i0_cleft, i1_cleft, i2_cleft,cleft_nb,
	       states0, states1);
    
    time_ryr+=timing();

    double sum_c_i_root[7];
    double sum_c_i[7];
    double cai_min;
    double cai_min_root=0.0;
    double cai_max;
    double cai_max_root=1.0;
    double sm;
    double ca[8];
    char caoutfile[100];

    if ((fmod(t,DT)<dt)||(t==dt)){
      time_io-=timing();
      for(idx=0; idx<7; idx++){
	sum_c_i[idx]=0.0;
	for ( i = 1; i <= nx; i += 1 ) 
	  for ( j = 1; j <= ny; j += 1 ) 
	    for ( k = 1; k <= nz; k += 1 ) 
	      sum_c_i[idx]+=C1[idx][i*ny0*nz0+j*nz0+k];
      }
      cai_min=my_min(C1[cai],len);
      cai_max=my_max(C1[cai],len);

      /* reduce operation comm*/
      MPI_Reduce(&sum_c_i[0], &sum_c_i_root[0], 7, MPI_DOUBLE, MPI_SUM, 0, comm);
      MPI_Reduce(&cai_min, &cai_min_root, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&cai_max, &cai_max_root, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      if(!my_id){
	sm = 0;
	ca[0] = t;
	if(save_data) fprintf(fpdata,"%f ", ca[0]);
	for(idx=0; idx<7; idx++){
	  sm += fraction[idx]*sum_c_i_root[idx];
	  ca[idx+1]  = sum_c_i_root[idx]/((double)nx*x_domains*(double)ny*y_domains*(double)nz*z_domains);
	  if(DB_PF){
	    printf("ca[%d]: %f , sum : %f, nx ny nz: %d %d %d \n",idx+1, ca[idx+1], 
		   sum_c_i_root[idx],nx*x_domains,ny*y_domains,nz*z_domains);
	  }
	  if(save_data) fprintf(fpdata,"%f ", ca[idx+1]);
	}
	if(save_data) fprintf(fpdata,"\n ");
	printf("%3d, %.3f, %3.2f, %7.2f, %3.2f, %4.2f, %.2f \n",
	       counter, t, ca[1], ca[2], cai_min_root, cai_max_root, sm);
      }

      if(save_data && in_midx_slice)
      {

	// If saving in hdf5
	if (save_hdf5)
	{

	  hsize_t dimsf[2] = {size_y, size_z};         /* dataset dimensions */
	  hsize_t chunk_dims[2] = {ny, nz};            /* chunk dimensions */

	  hsize_t h5_offset[2] = {coord[1]*nz, coord[0]*ny};
	  hsize_t h5_count[2] = {1, 1};

	  // Create group name
	  sprintf(hdf5_group_name, "/data_%d", counter);
	  hid_t group_id = H5Gcreate(h5_file_id, hdf5_group_name, 
				     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	  // Create data space for the time attribute
	  hsize_t dims = 1;
	  hid_t attr_space = H5Screate_simple(1, &dims, NULL);
	  
	  // Create a time attribute
	  hid_t time_id = H5Acreate(group_id, "time", H5T_NATIVE_DOUBLE, attr_space, 
				    H5P_DEFAULT, H5P_DEFAULT);

	  // Write the attribute data
	  double time_data = counter*DT;
	  herr_t status = H5Awrite(time_id, H5T_NATIVE_DOUBLE, &time_data);
	  
	  // Cleanup
	  H5Aclose(time_id);
	  H5Sclose(attr_space);

	  for (i=0; i<NUM_SAVE_SPECIES; i++)
	  {
	  
	    // Get species
	    int species = save_species[i];

	    sprintf(hdf5_dataset_name, "%s/%s", hdf5_group_name, species_names[species]);

	    // file and dataset identifiers
	    hid_t filespace = H5Screate_simple(2, dimsf, NULL); 
	    hid_t memspace  = H5Screate_simple(2, chunk_dims, NULL); 

	    // Create chunked dataset. 
	    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
	    H5Pset_chunk(plist_id, 2, chunk_dims);

	    // Create compression filter (Not supported in parallel yet...)
	    //unsigned int gzip_level = 9;
	    //herr_t status = H5Pset_filter(plist_id, H5Z_FILTER_DEFLATE, 
	    //				  H5Z_FLAG_OPTIONAL, 1, &gzip_level);
	    
	    hid_t dset_id = H5Dcreate(h5_file_id, hdf5_dataset_name, 
				      H5T_DATA_TYPE, filespace,
				      H5P_DEFAULT, plist_id, H5P_DEFAULT);

	    H5Pclose(plist_id);
	    H5Sclose(filespace);

	    // Select hyperslab in the file.
	    filespace = H5Dget_space(dset_id);
	    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, 
					 h5_offset, NULL, h5_count, chunk_dims);

	    // Copy data to h5_data
	    transfer_hdf5_data(h5_data, 
			       &(C0[species][ny0*nz0*mid_coord_x]), 
			       &(C1[species][ny0*nz0*mid_coord_x]), 
			       init_values[species], chunk_dims);

	    // Create property list for collective dataset write.
	    plist_id = H5Pcreate(H5P_DATASET_XFER);
	    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
	    status = H5Dwrite(dset_id, H5T_DATA_TYPE, memspace, filespace,
			      plist_id, h5_data);

	    // Close/release resources.
	    H5Dclose(dset_id);
	    H5Sclose(filespace);
	    H5Sclose(memspace);
	    H5Pclose(plist_id);
	  }
	  
	  H5Gclose(group_id);

	}

	// No HDF5
	else 
	{

	  // Get species
	  int species = save_species[i];
	  for (i=0; i<NUM_SAVE_SPECIES; i++)
	  {
	    sprintf(caoutfile, "%s/Ca%d_T%d_rank%d_%d_%d.np", outdirname, species, counter, 
		    coord[2], coord[1], coord[0]);
	    if(save_binary_file) 
	      store2Dmatrixfile_double_bin(caoutfile, C1[species], ny0, nz0, mid_coord_x);
	    else
	      store2Dmatrixfile_double_1D(caoutfile, C1[species], ny0, nz0, mid_coord_x);
	  }
	}
      }

      counter += 1;
    }

    // # Update Ca
    for(i=0;i<7;i++){
      C_temp=C0[i];
      C0[i]=C1[i];
      C1[i]=C_temp;
    }
  MPI_Waitall(6, ar_send_req, ar_status); 
  }
  time_main+=timing();
  if(my_id==0){
    if(save_data) fclose(fpdata);
    printf("cubiod_c: h:%d size_x:%d size_y:%d size_z:%d dt:%f, T:%f, TimeStep:%d, DT:%f, x_slice_num:%d\n",
	   h, size_x, size_y, size_z,dt,T,(int)(T/dt),DT,x_slice_num); 
    printf("nx0:%d ny0:%d nz0:%d size/array:%7.3f MB total size:%7.3f MB\n", 
	   nx0,ny0,nz0,nx0*ny0*nz0*8*1e-6,nx0*ny0*nz0*8*1e-6*12); 
#ifdef __PAPI__
    if ( PAPI_stop( EventSet, res_papi ) != PAPI_OK){
      printf("PAPI_accum_counters failed\n");
    }
    for (i = 0; i<NUM_EVENTS; i++){
      PAPI_event_code_to_name(Events[i], EventName);
      printf("PAPI Event name: %s, value: %lld\n", EventName, res_papi[i]);
    }
#endif
    printf("computing time: %7.3f \n",  time_conc);
    printf("updateryr time: %7.3f \n",  time_ryr);
    printf("communica time: %7.3f \n",  time_comm);
    printf("main      time: %7.3f \n",  time_main);


#ifdef __PAPI__
    printf("PAPI Performanc/core: %7.3f GFLOPS\n", res_papi[0]/1e9/time_conc);
#endif
  }

  if (save_hdf5)
  {
    H5Fclose(h5_file_id);
    free(h5_data);
  }

  for(i=0;i<5;i++){
    free(C0[i]);
    free(C1[i]);
  }
  
  free(C0[6]);
  free(C0[5]);
  free(i0_ryr);
  free(i1_ryr);
  free(i2_ryr);
  free(i0_csqn);
  free(i1_csqn);
  free(i2_csqn);
  free(i0_cleft);
  free(i1_cleft);
  free(i2_cleft);
  free(cleft_nb);
  MPI_Finalize();

  return 0;
}

void laplace3D (int nx0, int ny0, int nz0, double* C0,
                int nx1, int ny1, int nz1, double* C1,
                double alpha)//, int num_threads)
{

  // Set num threads
  // omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double C0_tmp;
  
  
  // Main kernel loop
  // #pragma omp parallel for private(i, j, k, C0_tmp) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        C0_tmp = -6*C0[i*nz0*ny0+j*nz0+k] + 
	  C0[(i-1)*nz0*ny0+j*nz0+k] + C0[(i+1)*nz0*ny0+j*nz0+k] + 
	  C0[i*nz0*ny0+(j-1)*nz0+k] + C0[i*nz0*ny0+(j+1)*nz0+k] + 
	  C0[i*nz0*ny0+j*nz0+k-1] + C0[i*nz0*ny0+j*nz0+k+1];
  
        // Put value back into return array with offset to indices 
        C1[i*nz1*ny1+j*nz1+k] = C0[i*nz1*ny1+j*nz1+k] + C0_tmp*alpha;
      }
    }  
  }    
}

void reaction3D (int nx0, int ny0, int nz0, double* Ca,
                 int nx1, int ny1, int nz1, double* buff,
                 double B_tot, double k_on, double k_off, double dt)//, int num_threads)
{

  // Set num threads
  //  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double* Ca_ijk;
  double* buff_ijk;
  
  // Main kernel loop
  //  #pragma omp parallel for private(i, j, k, J, Ca_ijk, buff_ijk) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_ijk = &Ca[i*nz0*ny0+j*nz0+k];
        buff_ijk = &buff[i*nz0*ny0+j*nz0+k];
        J = k_on*(B_tot - *buff_ijk)*(*Ca_ijk) - k_off*(*buff_ijk);
        *Ca_ijk -= dt*J;
        *buff_ijk += dt*J;
      }
    }  
  }    
}

void serca3D (int nx0, int ny0, int nz0, double* Ca_i,
              int nx1, int ny1, int nz1, double* Ca_SR,
              double dt, double gamma, double fudge)//, int num_threads)
{

  // Set num threads
  //  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double Ca_i2_ijk;
  double Ca_SR2_ijk;
  
  // Main kernel loop
  //  #pragma omp parallel for private(i, j, k, J, Ca_i2_ijk, Ca_SR2_ijk) //collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_i2_ijk = Ca_i[i*nz0*ny0+j*nz0+k];
        Ca_SR2_ijk = Ca_SR[i*nz0*ny0+j*nz0+k];
        Ca_i2_ijk *= Ca_i2_ijk;
        Ca_SR2_ijk *= Ca_SR2_ijk;
        J = fudge*(570997802.885875*Ca_i2_ijk - 0.0425239333622699*Ca_SR2_ijk)/(106720651.206402*Ca_i2_ijk + 182.498197548666*Ca_SR2_ijk + 5.35062954944879);
        Ca_i[i*nz0*ny0+j*nz0+k] -= dt*J;
        Ca_SR[i*nz0*ny0+j*nz0+k] += dt*J/gamma;
      }
    }  
  }    
}

void update_ryr(int h_scale,int nx0, int ny0, int nz0, double* Ca_i, double* Ca_SR, double* Ca_CSQN, 
                double* C10, double* C12, double* C13, double* C14,
		double k_on_CSQN, double k_off_CSQN, double CSQN_tot,
		double gamma, double K, double dt,
		int ryr_len, int* i0_ryr, int* i1_ryr, int* i2_ryr,
		int csqn_len, int* i0_csqn, int* i1_csqn, int* i2_csqn,
		int cleft_len, int* i0_cleft, int* i1_cleft, int* i2_cleft,int* cleft_nb,
		int* states0, int* states1)
{
  int i,j;
  int x_copy_from;
  int x,y,z;
  int nb_y,nb_z;
  int idx,idx_cleft,idx_csqn;
  double J;
  int open;
  double c0,c1;
  //extend csqn on x direction
//  for(j=(1-h_scale);j<h_scale;j++){
//extend cdqn on x+ direction for 30nm
  for(j=0;j<h_scale;j++){
    for(i=0;i<csqn_len;i+=1){
      x=i0_csqn[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
      y=i1_csqn[i];
      z=i2_csqn[i];
      idx=x*ny0*nz0+y*nz0+z;
      //CSQN step:
      J = k_on_CSQN*(CSQN_tot - Ca_CSQN[idx])*Ca_SR[idx] -  k_off_CSQN*Ca_CSQN[idx];
      Ca_SR[idx] -= dt*J;
      Ca_CSQN[idx] += dt*J;
    }
  }
 
//add no_flux boundary by copy the neighbour's  value on no_flux voxel

//add x+ front no-flux plane on ryr with +1 offset, and copy from -1 x-plane(where ryr is on)
  j=1;
  x_copy_from=-1;
  for(i=0;i<csqn_len;i+=1){
    x=i0_csqn[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
    y=i1_csqn[i];
    z=i2_csqn[i];
    idx_cleft=x*ny0*nz0+y*nz0+z;
    idx_csqn =(x+x_copy_from)*ny0*nz0+y*nz0+z;
    C10[idx_cleft]=C10[idx_csqn];
    C12[idx_cleft]=C12[idx_csqn];
    C13[idx_cleft]=C13[idx_csqn];
    C14[idx_cleft]=C14[idx_csqn];
  }
//add x+ back no-flux plane on ryr with h_scale offset, and copy from +1 x-plane(outside of csqn)
  if(h_scale==2)//15 nm
    j=h_scale+1;//guarantee that there is at least one voxel inner the no-flux boundary  
  else//5nm 3mn 1nm, 
    j=h_scale;
  x_copy_from=+1;
  for(i=0;i<csqn_len;i+=1){
    x=i0_csqn[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
    y=i1_csqn[i];
    z=i2_csqn[i];
    idx_cleft=x*ny0*nz0+y*nz0+z;
    idx_csqn =(x+x_copy_from)*ny0*nz0+y*nz0+z;
    C10[idx_cleft]=C10[idx_csqn];
    C12[idx_cleft]=C12[idx_csqn];
    C13[idx_cleft]=C13[idx_csqn];
    C14[idx_cleft]=C14[idx_csqn];
  }

 //extend y-z plane no_flux boundary along x+ direction with +1 offset and copy value from outside of CSQN by cleft_nb index
   int k;
  if(h_scale==2)//15 nm
    k=1;//guarantee that there is at least one voxel inner the no-flux boundary  
  else//5nm 3mn 1nm, 
    k=0;
  for(j=2;j<h_scale+k;j++){
    for(i=0;i<cleft_len;i+=1){
      x=i0_cleft[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
      y=i1_cleft[i];
      z=i2_cleft[i];
      nb_y=cleft_nb[i]/8-1;
      nb_z=cleft_nb[i]%8-1;
      idx_cleft=x*ny0*nz0+y*nz0+z;
      idx_csqn =x*ny0*nz0+(y+nb_y)*nz0+z+nb_z;
      C10[idx_cleft]=C10[idx_csqn];
      C12[idx_cleft]=C12[idx_csqn];
      C13[idx_cleft]=C13[idx_csqn];
      C14[idx_cleft]=C14[idx_csqn];
    }
  }

//add x- front no-flux plane on ryr with -h_scale/2(15nm) offset, and copy from +1 x-plane(t-tubule)
  j=0-h_scale/2;
  x_copy_from=1;
  for(i=0;i<csqn_len;i+=1){
    x=i0_csqn[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
    y=i1_csqn[i];
    z=i2_csqn[i];
    idx_cleft=x*ny0*nz0+y*nz0+z;
    idx_csqn =(x+x_copy_from)*ny0*nz0+y*nz0+z;
    C10[idx_cleft]=C10[idx_csqn];
    C12[idx_cleft]=C12[idx_csqn];
    C13[idx_cleft]=C13[idx_csqn];
    C14[idx_cleft]=C14[idx_csqn];
  }

//add x- back no-flux plane on ryr with -h_scale/2+1 offset, and copy from -1 x-plane(t-tubule)
/*    if(h_scale=2)
    j=0-h_scale/2-h_scale;
  else
    j=0-h_scale/2-h_scale+1;
 */
/* how thick should t-tubule be?  now, just set it  2 lines on x- direction */
//  j=0-h_scale/2-h_scale-1;
  j=0-h_scale/2-1;
  x_copy_from=-1;
  for(i=0;i<csqn_len;i+=1){
    x=i0_csqn[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
    y=i1_csqn[i];
    z=i2_csqn[i];
    idx_cleft=x*ny0*nz0+y*nz0+z;
    idx_csqn =(x+x_copy_from)*ny0*nz0+y*nz0+z;
    C10[idx_cleft]=C10[idx_csqn];
    C12[idx_cleft]=C12[idx_csqn];
    C13[idx_cleft]=C13[idx_csqn];
    C14[idx_cleft]=C14[idx_csqn];
  }

/* how thick should t-tubule be? */
/*  
 //extend y-z plane no_flux boundary along x- direction with +1 offset and copy value from outside of CSQN by cleft_nb index
   int k;
  if(h_scale==2)//15 nm
    k=1;//guarantee that there is at least one voxel inner the no-flux boundary  
  else//5nm 3mn 1nm, 
    k=0;
  for(j=0-h_scale/2-1;j>0-h_scale/2-h_scale+1-k;j--){
    for(i=0;i<cleft_len;i+=1){
      x=i0_cleft[i]+j;
#ifdef DEBUG_TEST
      if((x<0)||x>(nx0-1))
      {
	printf("wrong csqn x index\n");
	exit(0);
      }
#endif      
      y=i1_cleft[i];
      z=i2_cleft[i];
      nb_y=cleft_nb[i]/8-1;
      nb_z=cleft_nb[i]%8-1;
      idx_cleft=x*ny0*nz0+y*nz0+z;
      idx_csqn =x*ny0*nz0+(y+nb_y)*nz0+z+nb_z;
      C10[idx_cleft]=C10[idx_csqn];
      C12[idx_cleft]=C12[idx_csqn];
      C13[idx_cleft]=C13[idx_csqn];
      C14[idx_cleft]=C14[idx_csqn];
    }
  } 
*/

  for ( i = 0; i < ryr_len; i += 1 ) {
    x=i0_ryr[i];
    y=i1_ryr[i];
    z=i2_ryr[i];
    idx=x*ny0*nz0+y*nz0+z;
    //	#Continous formulation
    //	#states[:,i] += dt*stern(t, states[:,i], Ca_i[idx])
    stern_discrete(dt, &states0[i],&states1[i], Ca_i[idx]);
    open = states0[i]*(1-states1[i]);
    //	#Exp Euler:
    //	#J_RyR = k*open*(Ca_SR[idx]-Ca_i[idx])
    //	#Ca_i[idx]  += dt*J_RyR
    //	#Ca_SR[idx] -= dt*J_RyR/gamma;
    //	#Analytical update:
    // K = exp(-k_s*dt*(1+1/gamma))
    if (open){
      if(DB_PF)      printf("open [%d] ryr[%d,%d,%d] \n", i, x, y,z);
      c0 = (Ca_i[idx] + gamma*Ca_SR[idx])/(1+gamma);
      c1 = (Ca_i[idx] - Ca_SR[idx])/(1+1/gamma);
      Ca_i[idx]  = c0 + c1*K;
      Ca_SR[idx] = c0 - c1*K/gamma;
    }
  }
}

void stern(double t, double* y0, double* y1, double Ca){
  double m = *y0;
  double h = *y1;
  double kim = 0.005;
  double kom = 0.06;
  double K_i = 0.01*10;
  double K_o = 0.01*41.4;
  double ki = kim/K_i;
  double ko = kom/(K_o*K_o);
  double dm = ko*Ca*Ca*(1-m)-kom*m;
  double dh = ki*Ca*(1-h)-kim*h;
  *y0=dm;
  *y1=dh;
}

void stern_discrete(double dt, int* y0, int* y1, double Ca){
  double kim  = 0.002;  // 1/ms 
  double kom  = 1.5;  // 0.5 1/ms
  double kd_i = 20.0;  // 20.0 um*ms
  double kd_o = 0.9;   // um*ms^N 0.7, 0.8, 0.9, 1.0
  double Ca_ki  = Ca/kd_i; 
  double Ca_ko  = Ca/kd_o;
  double ki = Ca_ki*Ca_ki; // (Ca/kd_i)^2
  double ko = Ca_ko*Ca_ko*Ca_ko*Ca_ko;  // ko = (Ca/kd_o)^4

  //double kim = 0.005;          // Original: 0.005 
  //double kom = 0.04;           // Original: 0.06
  //double ki = Ca*1.5*1e-3;     // Original: Ca*0.5*1e-3
  //double ko = 1e-6*Ca*Ca*3500; // Original: 1e-6*Ca*Ca*{35,1200,2000,3500}
  double r;
  int m, h;

  m = *y0;
  if(m==1){
    r = my_random();
    m = 1 - (r<(dt*kom));
  }
  else
  {
    r=my_random();
    m = 1*(r<(dt*ko));
  }

  h = *y1;
  if(h==1){
    r = my_random();
    h = 1 - (r<(dt*kim));
  }
  else{
    r = my_random();
    h = 1*(r<(dt*ki));
  }
  *y0=m;
  *y1=h;
}

inline double my_random()
{
  double r;
  double x;
  //  r=(double)(rand()%100000000);
  //    x=(r*1e-8);
  x=((double)rand())/(double)RAND_MAX;
  return x;
}

void  store2Dmatrixfile_double_1D(char* outfile, double* ar, int rows, int cols, int x_strid){
  FILE *fpdata; 
  int i,j;       
  if ((fpdata=fopen(outfile, "w"))==NULL)
  {
    printf("fialed open output file ");
    printf("%s",outfile);
    printf(" ! \n ");
    exit(0);
  }
  //    printf("----Generating list output to ");
  //    printf("%s",outfile);
  //    printf(" file----\n");
  for(i=0;i<rows;i++)
  {
    for(j=0;j<cols;j++)
    {
      fprintf(fpdata,"%.9e ", ar[x_strid*rows*cols+i*cols+j]);
    }
    fprintf(fpdata,"\n");
  }
  fclose(fpdata);
  return;
}

void store2Dmatrixfile_double_bin(char* outfile, double* ar, int rows, int cols, int x_strid)
{
  FILE *fpdata; 
  int i,j;       
  if ((fpdata=fopen(outfile, "wb"))==NULL)
  {
    printf("failed open output file ");
    printf("%s",outfile);
    printf(" ! \n ");
    exit(0);
  }
  fwrite(&ar[x_strid*rows*cols],sizeof(double),rows*cols,fpdata);
  fclose(fpdata);
  return;
}

void transfer_hdf5_data(hdf5_data_type* h5_data, double* ar0, double* ar1, 
			double scale_value, hsize_t* chunk_dims)
{
  int i,j;
  int rows=chunk_dims[0];
  int cols=chunk_dims[1];
  
  // Transfer data from padded ar to stripped data
  for(i=0;i<rows;i++)
  {
    for(j=0;j<cols;j++)
    {
      double rel_data_diff = (ar1[i*(cols+2)+j+1]-ar0[i*(cols+2)+j+1])/scale_value;
      h5_data[i*cols+j] = (hdf5_data_type)round(rel_data_diff*H5_DATA_LIMIT_1);
    }
  }
}

void  store2Dmatrixfile_int_1D(char* outfile, int* ar, int rows, int cols){
  FILE *fpdata; 
  int i,j;       
  if ((fpdata=fopen(outfile, "w"))==NULL)
  {
    printf("failed open output file ");
    printf("%s",outfile);
    printf(" ! \n ");
    exit(0);
  }
  printf("----Generating list output to ");
  printf("%s",outfile);
  printf(" file----\n");
  for(i=0;i<rows;i++)
  {
    for(j=0;j<cols;j++)
      fprintf(fpdata,"%d ",ar[i*cols+j]);
    fprintf(fpdata,"\n");
  }
  fclose(fpdata);
  return;
}

double my_min(double* ar, int len)
{
  double min=ar[0];
  int i;
  for ( i = 0; i < len; i += 1 ) {
    if(ar[i]<min) min=ar[i];
  }
  return min;
}

double my_max(double* ar, int len)
{
  double max=ar[0];
  int i;
  for ( i = 0; i < len; i += 1 ) {
    if(ar[i]>max) max=ar[i];
  }
  return max;
}

double timing(){
  double time;
  struct timeval timmer;
  gettimeofday(&timmer,NULL);
  time = 1000000*timmer.tv_sec + timmer.tv_usec;
  time /= 1000000;
  return time;
}


int load_indices_serial(int nx, int ny, int nz, int h, 
			int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
			int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
			int** i0_cleft, int** i1_cleft, int** i2_cleft, int** cleft_nb, int* cleft_len,
			int x_slice_mid, int x_slice_width, int x_slice_num, int use_failing)
{
  int i,j,k;
  int nx_old;
  int ny_old;
  int nz_old;
  nx_old=nx;
  ny_old=ny;
  nz_old=nz;
  // Scale nx, xy, nz in terms of RyR
  if(30%h!=0){
    printf("30 must be divisible by h!");
    exit(1);
  }
  int h_scale;
  h_scale = 30/h;
  nx = nx/h_scale;
  ny = ny/h_scale;
  nz = nz/h_scale;

  // All CaRU placed mid-sarcomere
  //  int mid_x = (nx+1)/2;

  // load RyR indices from file
  int* i1;
  int* i2;
  int  i1_len;
  int  i2_len;
  char i_RyR_indices_name[200];
  char j_RyR_indices_name[200];
  sprintf(i_RyR_indices_name, "i_RyR_indices%s.dat", use_failing ? "_failing" : "");
  sprintf(j_RyR_indices_name, "j_RyR_indices%s.dat", use_failing ? "_failing" : "");

  if (use_failing)
    printf("Load failing indices");
  else
    printf("Load normal indices");

  i1=loadRyRindexfile_int(i_RyR_indices_name, &i1_len);
  i2=loadRyRindexfile_int(j_RyR_indices_name, &i2_len);
  //    # Only use the subset which are inside the geometry
  if(i1_len==i2_len)
    printf("num RyR before reduction: %d\n", i1_len);
  else
    printf("num RyR is wrong: i1_len!=i2_len\n");

  int* i1_temp;
  int* i2_temp;
  int  i1_temp_len=0;
  for ( i = 0; i < i1_len; i += 1 ) {
    if(i1[i]<ny) i1_temp_len++;
  }
  i1_temp=malloc(i1_temp_len*sizeof(int));
  i2_temp=malloc(i1_temp_len*sizeof(int));
  j=0;
  for ( i = 0; i < i1_len; i += 1 ) {
    if(i1[i]<ny){
      i1_temp[j]=i1[i];
      i2_temp[j]=i2[i];
      j++;
    }
  }
  free(i1);
  free(i2);

  int i1_ryr_len=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    if(i2_temp[i]<nz) i1_ryr_len++;
  }
  *i0_ryr=malloc(x_slice_num*i1_ryr_len*sizeof(int));
  *i1_ryr=malloc(x_slice_num*i1_ryr_len*sizeof(int));
  *i2_ryr=malloc(x_slice_num*i1_ryr_len*sizeof(int));
  j=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    if(i2_temp[i]<nz){
      for(k=0; k < x_slice_num; k++){
	(*i1_ryr)[k*i1_ryr_len+j]=i1_temp[i];
	(*i2_ryr)[k*i1_ryr_len+j]=i2_temp[i];
      }
      j++;
    }
  }
  free(i1_temp);
  free(i2_temp);

  // Scale indices and move to center of macro voxel
  for ( i = 0; i < i1_ryr_len; i += 1 ) {
    for(k=0; k < x_slice_num; k++){
      (*i0_ryr)[k*i1_ryr_len+i] = k*x_slice_width+x_slice_mid;
      //for those ryr just on 0 boundary, avoid to subtracting their coords to negative 
      if((*i1_ryr)[k*i1_ryr_len+i]>0)
	(*i1_ryr)[k*i1_ryr_len+i] = (*i1_ryr)[k*i1_ryr_len+i]*h_scale - floor((double)h_scale/2);
      else
	(*i1_ryr)[k*i1_ryr_len+i] = (*i1_ryr)[k*i1_ryr_len+i]*h_scale;
      if((*i2_ryr)[k*i1_ryr_len+i]>0)
	(*i2_ryr)[k*i1_ryr_len+i] = (*i2_ryr)[k*i1_ryr_len+i]*h_scale - floor((double)h_scale/2);
      else
	(*i2_ryr)[k*i1_ryr_len+i] = (*i2_ryr)[k*i1_ryr_len+i]*h_scale;
    }
  }
  *ryr_len=i1_ryr_len*x_slice_num;
  

  // load CSQN indices from file
  char i_csqn_indices_name[200];
  char j_csqn_indices_name[200];
  sprintf(i_csqn_indices_name, "i_csqn_indices%s.dat", use_failing ? "_failing" : "");
  sprintf(j_csqn_indices_name, "j_csqn_indices%s.dat", use_failing ? "_failing" : "");
  
  i1 = loadRyRindexfile_int(i_csqn_indices_name, &i1_len); 
  i2 = loadRyRindexfile_int(j_csqn_indices_name, &i2_len);
  if(i1_len==i2_len)
    printf("num CSQN before reduction: %d\n", i1_len);
  else
    printf("num CSQN is wrong: i1_len!=i2_len\n");

  //# Only use the subset which are inside the geometry
  //  i1_csqn = i1[i2<nz]*h_scale
  //    i2_csqn = i2[i2<nz]*h_scale
  //    i0_csqn = np.ones(len(i1_csqn), dtype=int)*mid_x*h_scale
  i1_temp_len=0;
  for ( i = 0; i < i1_len; i += 1 ) {
    if(i1[i]<ny) i1_temp_len++;
  }
  i1_temp=malloc(i1_temp_len*sizeof(int));
  i2_temp=malloc(i1_temp_len*sizeof(int));
  j=0;
  for ( i = 0; i < i1_len; i += 1 ) {
    if(i1[i]<ny){
      i1_temp[j]=i1[i];
      i2_temp[j]=i2[i];
      j++;
    }
  }
  free(i1);
  free(i2);

  int i1_csqn_len=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    if(i2_temp[i]<nz) i1_csqn_len++;
  }
  *i0_csqn=malloc(x_slice_num*i1_csqn_len*sizeof(int));
  *i1_csqn=malloc(x_slice_num*i1_csqn_len*sizeof(int));
  *i2_csqn=malloc(x_slice_num*i1_csqn_len*sizeof(int));
  j=0;
  for ( i = 0; i < i1_temp_len; i += 1 ) {
    if(i2_temp[i]<nz){
      for(k=0; k < x_slice_num; k++){
	(*i1_csqn)[k*i1_csqn_len+j]=i1_temp[i];
	(*i2_csqn)[k*i1_csqn_len+j]=i2_temp[i];
      }
      j++;
    }
  }
  free(i1_temp);
  free(i2_temp);

  // Scale indices and move to center of macro voxel
    for(k=0; k < x_slice_num; k++){
      for ( i = 0; i < i1_csqn_len; i += 1 ) {
      (*i0_csqn)[k*i1_csqn_len+i] =  k*x_slice_width+x_slice_mid;
      (*i1_csqn)[k*i1_csqn_len+i] = (*i1_csqn)[k*i1_csqn_len+i]*h_scale;
      (*i2_csqn)[k*i1_csqn_len+i] = (*i2_csqn)[k*i1_csqn_len+i]*h_scale; 
    }
  }

  int* i0_csqn_list;
  int* i1_csqn_list;
  int* i2_csqn_list;
  int  m;
  int csqn_count;

  *csqn_len=x_slice_num*i1_csqn_len*h_scale*h_scale;
  *cleft_len=0;//x_slice_num*i1_csqn_len*4*h_scale;
  //    # Add CSQN to all voxels covered by the original CSQN array
  if (h_scale > 1){
    i0_csqn_list=malloc(x_slice_num*i1_csqn_len*h_scale*h_scale*sizeof(int));
    i1_csqn_list=malloc(x_slice_num*i1_csqn_len*h_scale*h_scale*sizeof(int));
    i2_csqn_list=malloc(x_slice_num*i1_csqn_len*h_scale*h_scale*sizeof(int));
    csqn_count=0;
    //        # Add offsetted versions of the csqn
    for ( m = 0; m < x_slice_num; m += 1 ) {
      for ( i = 0; i < h_scale; i += 1 ) {
	for ( j = 0; j < h_scale; j += 1 ) {
	  for ( k = 0; k < i1_csqn_len; k += 1 ) {
	    i0_csqn_list[csqn_count]=(*i0_csqn)[m*i1_csqn_len+k];
	    i1_csqn_list[csqn_count]=(*i1_csqn)[m*i1_csqn_len+k]+i;
	    i2_csqn_list[csqn_count]=(*i2_csqn)[m*i1_csqn_len+k]+j;
	    csqn_count++;
	  }
	}
      }
    }
    if(csqn_count!=(*csqn_len))
    {
      printf("csqn_count wrong\n");
      exit(0);	
    }
  }
  else
  {
    i0_csqn_list=(*i0_csqn);
    i1_csqn_list=(*i1_csqn);
    i2_csqn_list=(*i2_csqn);
  }

  int a_slice_csqn_len=i1_csqn_len*h_scale*h_scale;
  BinarySort_two(&i1_csqn_list[0],&i2_csqn_list[0],a_slice_csqn_len);
  int* y_index;
  y_index=malloc(ny_old*sizeof(int));
  for ( i = 0; i < ny_old; i += 1 ) {
    y_index[i]=-1;
  }
  for ( i = a_slice_csqn_len-1; i >= 0; i -= 1 ) {
    y_index[i1_csqn_list[i]]=i;
  }
  
//generate cleft index on Y-Z plane,just wrapping the outside of a group of CSQN,
//If cleft is in the outside of the mesh or is already indexed by a CSQN, then it is not a true cleft.
//Also generate the relative coordinates for th neighbour of each cleft from which to copy the value.
//the relative coordinate of y is cleft_nb%8-1, and that of z is cleft_nb/8-1  
  int coord_y,coord_z; 
  *i1_cleft=(int*)malloc(i1_csqn_len*4*h_scale*sizeof(int));
  *i2_cleft=(int*)malloc(i1_csqn_len*4*h_scale*sizeof(int));
  *cleft_nb=(int*)malloc(i1_csqn_len*4*h_scale*sizeof(int));
  *cleft_len=0;
  for ( k = 0; k < i1_csqn_len; k += 1 ) {
    for ( j = 0; j < h_scale; j += 1 ) {
      //z bottom line
      coord_y=(*i1_csqn)[k]-1;
      coord_z=(*i2_csqn)[k]+j;
      if(IsTrueCleft(coord_y, coord_z, ny_old, nz_old, i1_csqn_list, i2_csqn_list, y_index, a_slice_csqn_len))
      {
	(*i1_cleft)[(*cleft_len)]=coord_y;
	(*i2_cleft)[(*cleft_len)]=coord_z;
//copy from outside
//	(*cleft_nb)[(*cleft_len)]=0+1;
//	copy from inside
	(*cleft_nb)[(*cleft_len)]=16+1;
	(*cleft_len)++;
      }
      //y left line
      coord_y=(*i1_csqn)[k]+j;
      coord_z=(*i2_csqn)[k]-1;
      if(IsTrueCleft(coord_y, coord_z, ny_old, nz_old, i1_csqn_list, i2_csqn_list, y_index, a_slice_csqn_len))
      {
	(*i1_cleft)[(*cleft_len)]=coord_y;
	(*i2_cleft)[(*cleft_len)]=coord_z;
//copy from inside
//	(*cleft_nb)[(*cleft_len)]=8+0;
//copy from inside	
	(*cleft_nb)[(*cleft_len)]=8+2;
	(*cleft_len)++;
      }
      //z top line
      coord_y=(*i1_csqn)[k]+h_scale;
      coord_z=(*i2_csqn)[k]+j;
      if(IsTrueCleft(coord_y, coord_z, ny_old, nz_old, i1_csqn_list, i2_csqn_list, y_index, a_slice_csqn_len))
      {
	(*i1_cleft)[(*cleft_len)]=coord_y;
	(*i2_cleft)[(*cleft_len)]=coord_z;
	//copy from outside
//	(*cleft_nb)[(*cleft_len)]=16+1;
//	copy from inside
	(*cleft_nb)[(*cleft_len)]=0+1;
	(*cleft_len)++;
      }
      //y right line
      coord_y=(*i1_csqn)[k]+j;
      coord_z=(*i2_csqn)[k]+h_scale;
      if(IsTrueCleft(coord_y, coord_z, ny_old, nz_old, i1_csqn_list, i2_csqn_list, y_index, a_slice_csqn_len))
      {
	(*i1_cleft)[(*cleft_len)]=coord_y;
	(*i2_cleft)[(*cleft_len)]=coord_z;
	//copy from outside
//	(*cleft_nb)[(*cleft_len)]=8+2;
//	copy from inside
	(*cleft_nb)[(*cleft_len)]=8+0;
	(*cleft_len)++;
      }
    }
  }
  if((*cleft_len)>i1_csqn_len*4*h_scale){
    printf("wrong cleft_len found\n");
    exit(0);
  }

//add cleft for multiple 2um x-slices   
  int* i0_cleft_list;
  int* i1_cleft_list;
  int* i2_cleft_list;
  int* cleft_nb_list;
  i0_cleft_list=malloc(x_slice_num*(*cleft_len)*sizeof(int));
  i1_cleft_list=malloc(x_slice_num*(*cleft_len)*sizeof(int));
  i2_cleft_list=malloc(x_slice_num*(*cleft_len)*sizeof(int));
  cleft_nb_list=malloc(x_slice_num*(*cleft_len)*sizeof(int));
  for(k=0; k < x_slice_num; k++){
    for ( i = 0; i < (*cleft_len); i += 1 ) {
      i0_cleft_list[k*(*cleft_len)+i] =  k*x_slice_width+x_slice_mid;
      i1_cleft_list[k*(*cleft_len)+i] = (*i1_cleft)[i];
      i2_cleft_list[k*(*cleft_len)+i] = (*i2_cleft)[i]; 
      cleft_nb_list[k*(*cleft_len)+i] = (*cleft_nb)[i]; 
    }
  }
  free(*i1_cleft);
  free(*i2_cleft);
  free(*cleft_nb);
  *i0_cleft=i0_cleft_list;
  *i1_cleft=i1_cleft_list;
  *i2_cleft=i2_cleft_list;
  *cleft_nb=cleft_nb_list;
  *cleft_len=x_slice_num*(*cleft_len);

  if (h_scale > 1){
    free(*i0_csqn);
    free(*i1_csqn);
    free(*i2_csqn);
    *i0_csqn=i0_csqn_list;
    *i1_csqn=i1_csqn_list;
    *i2_csqn=i2_csqn_list;
  }
  return h_scale;
}

int IsTrueCleft(int coord_y, int coord_z, int size_y, int size_z, int *i1_csqn, int *i2_csqn, int* y_index, int csqn_len)
{
  int i;
  //in outside of the mesh
  if((coord_y<0)||(coord_y>=size_y)||(coord_z<0)||(coord_z>=size_z))
    return 0;
  i=y_index[coord_y];
  //not in CSQN 
  if(i<0)
    return 1;
  while(i1_csqn[i]==coord_y){
    //in  CSQN 
    if(i2_csqn[i]==coord_z) 
      return 0;
    i++;
    //not in CSQN
    if(i>=csqn_len)
      return 1;
  }
  return 1;
}

int idxinrank(int nx, int ny, int nz,
              int i0, int i1, int i2,
              int rank, MPI_Comm comm3d)
{
  int coords[3];
  MPI_Cart_coords(comm3d,rank,3,coords);
  if( (i0>=coords[2]*nx)&&((i0<coords[2]+1)*nx)&&
      (i1>=coords[1]*ny)&&((i1<coords[1]+1)*ny)&&
      (i2>=coords[0]*nz)&&((i2<coords[0]+1)*nz))
  {
    return 1;
  }
  else
    return 0;
}

int idxbl2rank(int nx, int ny, int nz,
               int i0, int i1, int i2,
	       int* coords,
               MPI_Comm comm3d)
{
  int rank=0;
  coords[2]=i0/nx;
  coords[1]=i1/ny;
  coords[0]=i2/nz;
  MPI_Cart_rank(comm3d,coords,&rank);
  return rank;
}

int distr_ryr_csqn_state(int h, int size_x, int size_y, int size_z,
			 int nx, int ny, int nz, 
			 int** i0_ryr,  int** i1_ryr,  int** i2_ryr,  int* ryr_len,
			 int** i0_csqn, int** i1_csqn, int** i2_csqn, int* csqn_len,
			 int** i0_cleft, int** i1_cleft, int** i2_cleft,int** cleft_nb, int* cleft_len,
			 int** states0, int** states1,
			 int x_slice_mid,int x_slice_width, int x_slice_num,
			 MPI_Comm comm3d, MPI_Comm comm, int use_failing)
{

  int i,j;
  int h_scale;
  int* global_i0_ryr;
  int* global_i1_ryr;
  int* global_i2_ryr;
  int* global_i0_ryr_reorder;
  int* global_i1_ryr_reorder;
  int* global_i2_ryr_reorder;
  int* global_i0_csqn;
  int* global_i1_csqn;
  int* global_i2_csqn;
  int* global_i0_csqn_reorder;
  int* global_i1_csqn_reorder;
  int* global_i2_csqn_reorder;
  int* global_i0_cleft;
  int* global_i1_cleft;
  int* global_i2_cleft;
  int* global_cleft_nb;
  int* global_i0_cleft_reorder;
  int* global_i1_cleft_reorder;
  int* global_i2_cleft_reorder;
  int* global_cleft_nb_reorder;
  int  global_ryr_len;
  int  global_csqn_len;
  int  global_cleft_len;
  int* global_states0;
  int* global_states0_reorder;
  int* ryr_rec_count;
  int* ryr_rec_disp;
  int* ryr_rec_offset;
  int* csqn_rec_count;
  int* csqn_rec_disp;
  int* csqn_rec_offset;
  int* cleft_rec_count;
  int* cleft_rec_disp;
  int* cleft_rec_offset;
  int my_id;
  int nproc;
  int coords[3];
  MPI_Comm_rank(comm,&my_id);
  MPI_Comm_size(comm,&nproc);

  if(my_id==0){
    h_scale=load_indices_serial(size_x, size_y, size_z, h,
				&global_i0_ryr, &global_i1_ryr, &global_i2_ryr, &global_ryr_len,
				&global_i0_csqn, &global_i1_csqn,&global_i2_csqn,&global_csqn_len,
				&global_i0_cleft, &global_i1_cleft, &global_i2_cleft, &global_cleft_nb,
				&global_cleft_len, x_slice_mid,x_slice_width,x_slice_num, 
				use_failing);

    printf("load indices from file: h:%d, h_scale:%d, nx:%d, ny:%d, nz:%d, ryr_len:%d, csqn_len:%d cleft_len:%d\n",
	   h, h_scale, nx, ny, nz, global_ryr_len, global_csqn_len, global_cleft_len);

    if(global_ryr_len>0)
      global_states0=malloc(global_ryr_len*sizeof(int));
    else
      global_states0=malloc(1*sizeof(int));

    for ( i = 0; i < global_ryr_len; i++) 
      global_states0[i]=0;

    if(global_ryr_len>=23){
      for ( i = 1; i < 23; i =i+3 ) 
	global_states0[i]=1;
    }
    else
    {
      for ( i = 1; i < global_ryr_len ; i =i+10 ) 
	global_states0[i]=1; 
    }
    if(DB_PF){
      for(i=0;i<global_ryr_len;i++){
	if(global_states0[i]==1)
	  printf("ryr[%d]:%d,%d,%d \n",i,global_i0_ryr[i],global_i1_ryr[i],global_i2_ryr[i]);
      }
    }

    ryr_rec_count=malloc(nproc*sizeof(int));
    csqn_rec_count=malloc(nproc*sizeof(int));
    cleft_rec_count=malloc(nproc*sizeof(int));
    for (i = 0; i < nproc; i++) {
      ryr_rec_count[i]=0;
      csqn_rec_count[i]=0;
      cleft_rec_count[i]=0;
    }
    for(i=0;i<global_ryr_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_ryr[i],global_i1_ryr[i],global_i2_ryr[i],coords,comm3d);
      ryr_rec_count[j]++;
    }
    for(i=0;i<global_csqn_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_csqn[i],global_i1_csqn[i],global_i2_csqn[i],coords,comm3d);
      csqn_rec_count[j]++;
    }
    for(i=0;i<global_cleft_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_cleft[i],global_i1_cleft[i],global_i2_cleft[i],coords,comm3d);
      cleft_rec_count[j]++;
    }
    for (i = 0; i < nproc; i++) {
      if(DB_PF) printf("ryr_rec_count[%d]: %d\n",i, ryr_rec_count[i]);
      if(DB_PF) printf("csqn_rec_count[%d]: %d\n",i, csqn_rec_count[i]);
      if(DB_PF) printf("cleft_rec_count[%d]: %d\n",i, cleft_rec_count[i]);
    }
    ryr_rec_disp =  malloc(nproc*sizeof(int));
    csqn_rec_disp =  malloc(nproc*sizeof(int));
    cleft_rec_disp = malloc(nproc*sizeof(int));
    ryr_rec_disp[0] = 0;
    csqn_rec_disp[0] = 0;
    cleft_rec_disp[0] = 0;
    for (i = 1; i < nproc; i++) {
      ryr_rec_disp[i] = ryr_rec_disp[i-1] + ryr_rec_count[i-1];
      csqn_rec_disp[i] = csqn_rec_disp[i-1] + csqn_rec_count[i-1];
      cleft_rec_disp[i] = cleft_rec_disp[i-1] + cleft_rec_count[i-1];
    }
    if(global_ryr_len!=ryr_rec_disp[nproc-1]+ryr_rec_count[nproc-1])
    {
      printf("Global ryr Count mismatch %d\n", ryr_rec_disp[nproc-1]+ryr_rec_count[nproc-1]);
    }
    if(global_csqn_len!=csqn_rec_disp[nproc-1]+csqn_rec_count[nproc-1])
    {
      printf("Global csqn Count mismatch %d\n", csqn_rec_disp[nproc-1]+csqn_rec_count[nproc-1]);
    }
    if(global_cleft_len!=cleft_rec_disp[nproc-1]+cleft_rec_count[nproc-1])
    {
      printf("Global cleft Count mismatch %d\n", cleft_rec_disp[nproc-1]+cleft_rec_count[nproc-1]);
    }

    ryr_rec_offset =  malloc(nproc*sizeof(int));
    csqn_rec_offset =  malloc(nproc*sizeof(int));
    cleft_rec_offset =  malloc(nproc*sizeof(int));
    for (i = 0; i < nproc; i++) {
      ryr_rec_offset[i]=0;
      csqn_rec_offset[i]=0;
      cleft_rec_offset[i]=0;
    }

    global_i0_ryr_reorder=malloc(global_ryr_len*sizeof(int));
    global_i1_ryr_reorder=malloc(global_ryr_len*sizeof(int));
    global_i2_ryr_reorder=malloc(global_ryr_len*sizeof(int));
    global_states0_reorder=malloc(global_ryr_len*sizeof(int));
    for(i=0;i<global_ryr_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_ryr[i],global_i1_ryr[i],global_i2_ryr[i],coords,comm3d);
      global_i0_ryr_reorder[ryr_rec_disp[j]+ryr_rec_offset[j]]=global_i0_ryr[i]-coords[2]*nx+1;
      global_i1_ryr_reorder[ryr_rec_disp[j]+ryr_rec_offset[j]]=global_i1_ryr[i]-coords[1]*ny+1;
      global_i2_ryr_reorder[ryr_rec_disp[j]+ryr_rec_offset[j]]=global_i2_ryr[i]-coords[0]*nz+1;
      global_states0_reorder[ryr_rec_disp[j]+ryr_rec_offset[j]]=global_states0[i];
      ryr_rec_offset[j]++;
    }
    for (i = 0; i < nproc; i++) {
      if(ryr_rec_offset[i]!=ryr_rec_count[i])
	printf("ryr reorder count error on proc %d \n",i);
    }
    free(global_i0_ryr);
    free(global_i1_ryr);
    free(global_i2_ryr);
    free(global_states0);
    free(ryr_rec_offset);
   //distribute cleft to there own MPI process
    global_i0_csqn_reorder=malloc(global_csqn_len*sizeof(int));
    global_i1_csqn_reorder=malloc(global_csqn_len*sizeof(int));
    global_i2_csqn_reorder=malloc(global_csqn_len*sizeof(int));
    for(i=0;i<global_csqn_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_csqn[i],global_i1_csqn[i],global_i2_csqn[i],coords,comm3d);
      global_i0_csqn_reorder[csqn_rec_disp[j]+csqn_rec_offset[j]]=global_i0_csqn[i]-coords[2]*nx+1;
      global_i1_csqn_reorder[csqn_rec_disp[j]+csqn_rec_offset[j]]=global_i1_csqn[i]-coords[1]*ny+1;
      global_i2_csqn_reorder[csqn_rec_disp[j]+csqn_rec_offset[j]]=global_i2_csqn[i]-coords[0]*nz+1;
      csqn_rec_offset[j]++;
    }
    for (i = 0; i < nproc; i++) {
      if(csqn_rec_offset[i]!=csqn_rec_count[i])
	printf("csqn reorder count error on proc %d \n",i);
    }
    free(global_i0_csqn);
    free(global_i1_csqn);
    free(global_i2_csqn);
    free(csqn_rec_offset);

    global_i0_cleft_reorder=malloc(global_cleft_len*sizeof(int));
    global_i1_cleft_reorder=malloc(global_cleft_len*sizeof(int));
    global_i2_cleft_reorder=malloc(global_cleft_len*sizeof(int));
    global_cleft_nb_reorder=malloc(global_cleft_len*sizeof(int));
    for(i=0;i<global_cleft_len;i++) {
      j=idxbl2rank(nx,ny,nz,global_i0_cleft[i],global_i1_cleft[i],global_i2_cleft[i],coords,comm3d);
      global_i0_cleft_reorder[cleft_rec_disp[j]+cleft_rec_offset[j]]=global_i0_cleft[i]-coords[2]*nx+1;
      global_i1_cleft_reorder[cleft_rec_disp[j]+cleft_rec_offset[j]]=global_i1_cleft[i]-coords[1]*ny+1;
      global_i2_cleft_reorder[cleft_rec_disp[j]+cleft_rec_offset[j]]=global_i2_cleft[i]-coords[0]*nz+1;
      global_cleft_nb_reorder[cleft_rec_disp[j]+cleft_rec_offset[j]]=global_cleft_nb[i];
      cleft_rec_offset[j]++;
    }
    for (i = 0; i < nproc; i++) {
      if(cleft_rec_offset[i]!=cleft_rec_count[i])
	printf("cleft reorder count error on proc %d \n",i);
    }
    free(global_i0_cleft);
    free(global_i1_cleft);
    free(global_i2_cleft);
    free(global_cleft_nb);
    free(cleft_rec_offset);

  }
  //MPI_Gather(&n_ryr,1,MPI_INT,&states_rec_count[0],1,MPI_INT,0,comm);
  MPI_Scatter(&ryr_rec_count[0],1,MPI_INT,ryr_len,1, MPI_INT,0,comm);
  MPI_Scatter(&csqn_rec_count[0],1,MPI_INT,csqn_len,1, MPI_INT,0,comm);
  MPI_Scatter(&cleft_rec_count[0],1,MPI_INT,cleft_len,1, MPI_INT,0,comm);
  if(*ryr_len>0){
    *i0_ryr=(int*)mpi_malloc(my_id,*ryr_len*sizeof(int));
    *i1_ryr=(int*)mpi_malloc(my_id,*ryr_len*sizeof(int));
    *i2_ryr=(int*)mpi_malloc(my_id,*ryr_len*sizeof(int));
  }
  else
  {
    *i0_ryr=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i1_ryr=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i2_ryr=(int*)mpi_malloc(my_id,1*sizeof(int));
  }

  if(*csqn_len>0)
  {
    *i0_csqn=(int*)mpi_malloc(my_id,*csqn_len*sizeof(int));
    *i1_csqn=(int*)mpi_malloc(my_id,*csqn_len*sizeof(int));
    *i2_csqn=(int*)mpi_malloc(my_id,*csqn_len*sizeof(int));
  }
  else
  {
    *i0_csqn=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i1_csqn=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i2_csqn=(int*)mpi_malloc(my_id,1*sizeof(int));
  }

  if(*cleft_len>0)
  {
    *i0_cleft=(int*)mpi_malloc(my_id,*cleft_len*sizeof(int));
    *i1_cleft=(int*)mpi_malloc(my_id,*cleft_len*sizeof(int));
    *i2_cleft=(int*)mpi_malloc(my_id,*cleft_len*sizeof(int));
    *cleft_nb=(int*)mpi_malloc(my_id,*cleft_len*sizeof(int));
  }
  else
  {
    *i0_cleft=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i1_cleft=(int*)mpi_malloc(my_id,1*sizeof(int));
    *i2_cleft=(int*)mpi_malloc(my_id,1*sizeof(int));
    *cleft_nb=(int*)mpi_malloc(my_id,1*sizeof(int));
  }

  if(*ryr_len>0){
    *states0=(int*)mpi_malloc(my_id,*ryr_len*sizeof(int));
    *states1=(int*)mpi_malloc(my_id,*ryr_len*sizeof(int));
    for ( i = 0; i < *ryr_len; i += 1 ) {
      (*states0)[i]=0;
      (*states1)[i]=0;
    }
  }
  else
  {
    *states0=(int*)mpi_malloc(my_id,1*sizeof(int));
    *states1=(int*)mpi_malloc(my_id,1*sizeof(int));
    (*states0)[0]=0;
    (*states1)[0]=0;
  }

  MPI_Scatterv(global_i0_ryr_reorder, ryr_rec_count,ryr_rec_disp, MPI_INT, *i0_ryr, *ryr_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i1_ryr_reorder, ryr_rec_count,ryr_rec_disp, MPI_INT, *i1_ryr, *ryr_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i2_ryr_reorder, ryr_rec_count,ryr_rec_disp, MPI_INT, *i2_ryr, *ryr_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i0_csqn_reorder, csqn_rec_count,csqn_rec_disp, MPI_INT, *i0_csqn, *csqn_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i1_csqn_reorder, csqn_rec_count,csqn_rec_disp, MPI_INT, *i1_csqn, *csqn_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i2_csqn_reorder, csqn_rec_count,csqn_rec_disp, MPI_INT, *i2_csqn, *csqn_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i0_cleft_reorder, cleft_rec_count,cleft_rec_disp, MPI_INT, *i0_cleft, *cleft_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i1_cleft_reorder, cleft_rec_count,cleft_rec_disp, MPI_INT, *i1_cleft, *cleft_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_i2_cleft_reorder, cleft_rec_count,cleft_rec_disp, MPI_INT, *i2_cleft, *cleft_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_cleft_nb_reorder, cleft_rec_count,cleft_rec_disp, MPI_INT, *cleft_nb, *cleft_len, MPI_INT, 0, comm);
  MPI_Scatterv(global_states0_reorder, ryr_rec_count, ryr_rec_disp, MPI_INT, *states0, *ryr_len, MPI_INT, 0, comm);
  //MPI_Bcast(&global_ryr_num,1,MPI_INT,0,comm);


  if(DB_PF) printf("Thread%d: ryr_len=%d\n",my_id, *ryr_len);
  //	sprintf(caoutfile,"%s/Ca%d_T%d_rank%d_%d_%d_s0.np",outdirname,i,counter,coord[2],coord[1],coord[0]);
  //	store2Dmatrixfile_double_1D(caoutfile,C1[i],ny0,nz0,30);
 
  //MPI_Gatherv(states0, n_ryr, MPI_INT, global_states0, states_rec_count, states_rec_disp, MPI_INT, 0, comm);
 
  //  if(my_id==2) {
  //    for(i=0;i<*ryr_len;i++) printf("Thread2 states[%d]: %d\n",i,(*states0)[i]);
  //  }
  if(DB_PF){
    for(i=0;i<*ryr_len;i++){
      if((*states0)[i]==1){
	printf("Proc%d,ryr_len=%d,ryr[%d]:%d,%d,%d \n",my_id, *ryr_len,i,(*i0_ryr)[i],(*i1_ryr)[i],(*i2_ryr)[i]);
      }
    }
  }

  if(my_id==0){
    free(ryr_rec_count);
    free(ryr_rec_disp);
    free(csqn_rec_count);
    free(csqn_rec_disp);
    free(cleft_rec_count);
    free(cleft_rec_disp);
    free(global_i0_ryr_reorder);
    free(global_i1_ryr_reorder);
    free(global_i2_ryr_reorder);
    free(global_i0_csqn_reorder);
    free(global_i1_csqn_reorder);
    free(global_i2_csqn_reorder);
    free(global_i0_cleft_reorder);
    free(global_i1_cleft_reorder);
    free(global_i2_cleft_reorder);
    free(global_cleft_nb_reorder);
    free(global_states0_reorder);
  }
  return 30/h;
}

//int*  loadRyRindexfile_int(char* infile, CONDFN cf, int cond)
int* loadRyRindexfile_int(char* infile, int* count)
{
  FILE *fpdata; 
  int* arreturn; 
  int i;       
  int temp_d;
  *count=0;
  if(DB_PF) printf("Load file name: %s\n", infile);
  fpdata = fopen(infile, "r");    
  if(fpdata==NULL)
  {
    printf("\nFailure to open input file.\n");
    exit(0);
  }
  while(fscanf(fpdata, "%d", &temp_d)!=EOF){
    //       if(cf(temp_d,cond)) count++;
    (*count)++;
    //	 printf("%d,",temp_d);
  }
  if(DB_PF) printf("There are %d indices satisfy the condition\n",*count);
  arreturn = malloc((*count)*sizeof(int));
  if (arreturn == NULL)
  {
    printf("\nFailure trying to allocate room for array.\n");
    exit(0);
  }
  rewind(fpdata);
  i=0;
  while(fscanf(fpdata, "%d", &temp_d)!=EOF){
    //      if(cf(temp_d,cond)) {
    arreturn[i]=temp_d;
    i++;
    //      }
  }
  fclose(fpdata);
  if (*count != i)
  {
    printf("Wrong indices number\n");
    exit(0);
  }
  if(DB_PF) printf("load file %s over \n", infile);
  return arreturn;    
}


void readparam(int* iconf, double* conf)
{
  FILE* file2;
  char  Data[MAX_LINE_LENGTH];
  if((file2=fopen("param","r")) == NULL)
  { printf("Error opening param file\n");
    return;
  }

  // h
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[0]);

  // size_x
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[1]);

  // size_y
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[2]);

  // size_z
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[3]);

  // x_domains
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[4]);

  // y_domains
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[5]);

  // z_domains
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[6]);

  // save_data
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[7]);

  // use_failing
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[8]);

  // T
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%le\n",&conf[0]);

  // DT
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%le\n",&conf[1]);

  // save data in binary file
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[9]);

  // save data in hdf5 format
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[10]);

  // blocking_y_for_cache
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d",&iconf[11]);

  fclose(file2);
}   


void updateBound(double* C00, double* C01, double* C02, double* C03, double* C04, 
    int C_flag, int nx0, int ny0, int nz0,
    double* yz_sbuf0,double* yz_rbuf0,
    double* xz_sbuf0,double* xz_rbuf0,
    double* xy_sbuf0,double* xy_rbuf0,
    double* yz_sbuf1,double* yz_rbuf1,
    double* xz_sbuf1,double* xz_rbuf1,
    double* xy_sbuf1,double* xy_rbuf1,
    int* neighbor, MPI_Status* ar_status, MPI_Request* ar_send_req, MPI_Request* ar_recv_req,
    MPI_Comm comm, MPI_Comm comm3d) 
{
  int i,j,k;
  int nx=nx0-2;
  int ny=ny0-2;
  int nz=nz0-2;
  int dims[3];
  int periods[3];
  int coords[3];
  int ZN=0, ZP=1, YN=2, YP=3, XN=4, XP=5;
  MPI_Cart_get(comm3d, 3, dims, periods, coords);

  // Ghost X end sheet
  if(coords[2]==0){
    i=0;
    for (j=1; j<ny0-1; j++)
      for (k=1; k<nz0-1; k++){
	C00[i*nz0*ny0+j*nz0+k] = C00[(i+1)*nz0*ny0+j*nz0+k];
	C01[i*nz0*ny0+j*nz0+k] = C01[(i+1)*nz0*ny0+j*nz0+k];
	C02[i*nz0*ny0+j*nz0+k] = C02[(i+1)*nz0*ny0+j*nz0+k];
	C03[i*nz0*ny0+j*nz0+k] = C03[(i+1)*nz0*ny0+j*nz0+k];
	C04[i*nz0*ny0+j*nz0+k] = C04[(i+1)*nz0*ny0+j*nz0+k];
      }
  }
  else
  {
    putin_sendbuffer_yz(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &yz_sbuf0[0*ny*nz],ny*nz);
    putin_sendbuffer_yz(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &yz_sbuf0[1*ny*nz],ny*nz);
    putin_sendbuffer_yz(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &yz_sbuf0[2*ny*nz],ny*nz);
    putin_sendbuffer_yz(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &yz_sbuf0[3*ny*nz],ny*nz);
    putin_sendbuffer_yz(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &yz_sbuf0[4*ny*nz],ny*nz);
  }
   MPI_Isend(yz_sbuf0,5*ny*nz,MPI_DOUBLE,neighbor[XN],C_flag+1000, comm, &ar_send_req[0]);
   MPI_Irecv(yz_rbuf0,5*ny*nz,MPI_DOUBLE,neighbor[XN],C_flag+1000, comm, &ar_recv_req[0]);
//  MPI_Sendrecv(yz_sbuf0,5*ny*nz,MPI_DOUBLE,neighbor[XN],C_flag+1000,
//               yz_rbuf0,5*ny*nz,MPI_DOUBLE,neighbor[XN],C_flag+1000,comm,&status);
 

  if(coords[2]==(dims[2]-1))
  {
    i=nx0-1;
    for (j=1; j<ny0-1; j++)
      for (k=1; k<nz0-1; k++){
	C00[i*nz0*ny0+j*nz0+k] = C00[(i-1)*nz0*ny0+j*nz0+k];
	C01[i*nz0*ny0+j*nz0+k] = C01[(i-1)*nz0*ny0+j*nz0+k];
	C02[i*nz0*ny0+j*nz0+k] = C02[(i-1)*nz0*ny0+j*nz0+k];
	C03[i*nz0*ny0+j*nz0+k] = C03[(i-1)*nz0*ny0+j*nz0+k];
	C04[i*nz0*ny0+j*nz0+k] = C04[(i-1)*nz0*ny0+j*nz0+k];
      }
  }
  else
  {
    putin_sendbuffer_yz( (nx0-2)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &yz_sbuf1[0*ny*nz],ny*nz);
    putin_sendbuffer_yz( (nx0-2)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &yz_sbuf1[1*ny*nz],ny*nz);
    putin_sendbuffer_yz( (nx0-2)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &yz_sbuf1[2*ny*nz],ny*nz);
    putin_sendbuffer_yz( (nx0-2)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &yz_sbuf1[3*ny*nz],ny*nz);
    putin_sendbuffer_yz( (nx0-2)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &yz_sbuf1[4*ny*nz],ny*nz);
  }
  MPI_Isend(yz_sbuf1,5*ny*nz,MPI_DOUBLE,neighbor[XP],C_flag+1000, comm, &ar_send_req[1]);
  MPI_Irecv(yz_rbuf1,5*ny*nz,MPI_DOUBLE,neighbor[XP],C_flag+1000, comm, &ar_recv_req[1]);
//  MPI_Sendrecv(yz_sbuf1,5*ny*nz,MPI_DOUBLE,neighbor[XP],C_flag+1000,
//               yz_rbuf1,5*ny*nz,MPI_DOUBLE,neighbor[XP],C_flag+1000,comm,&status);

 

//  printf("exchange X end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
  // Ghost Y end sheet
  if(coords[1]==0){
    j=0;
    for (i=1; i<nx0-1; i++)
      for (k=1; k<nz0-1; k++){
	C00[i*nz0*ny0+j*nz0+k] = C00[i*nz0*ny0+(j+1)*nz0+k];
	C01[i*nz0*ny0+j*nz0+k] = C01[i*nz0*ny0+(j+1)*nz0+k];
	C02[i*nz0*ny0+j*nz0+k] = C02[i*nz0*ny0+(j+1)*nz0+k];
	C03[i*nz0*ny0+j*nz0+k] = C03[i*nz0*ny0+(j+1)*nz0+k];
	C04[i*nz0*ny0+j*nz0+k] = C04[i*nz0*ny0+(j+1)*nz0+k];
      }
  }
  else
  {
    putin_sendbuffer_xz(      1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &xz_sbuf0[0*nx*nz],nx*nz);
    putin_sendbuffer_xz(      1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &xz_sbuf0[1*nx*nz],nx*nz);
    putin_sendbuffer_xz(      1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &xz_sbuf0[2*nx*nz],nx*nz);
    putin_sendbuffer_xz(      1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &xz_sbuf0[3*nx*nz],nx*nz);
    putin_sendbuffer_xz(      1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &xz_sbuf0[4*nx*nz],nx*nz);
  }
  MPI_Isend(xz_sbuf0,5*nx*nz,MPI_DOUBLE,neighbor[YN],C_flag+2000, comm, &ar_send_req[2]);
  MPI_Irecv(xz_rbuf0,5*nx*nz,MPI_DOUBLE,neighbor[YN],C_flag+2000, comm, &ar_recv_req[2]);
//  MPI_Sendrecv(xz_sbuf0,5*nx*nz,MPI_DOUBLE,neighbor[YN],C_flag+2000,
//               xz_rbuf0,5*nx*nz,MPI_DOUBLE,neighbor[YN],C_flag+2000,comm,&status);
 

  if(coords[1]==(dims[1]-1))
  {
    j=ny0-1;
    for (i=1; i<nx0-1; i++)
      for (k=1; k<nz0-1; k++){
	C00[i*nz0*ny0+j*nz0+k] = C00[i*nz0*ny0+(j-1)*nz0+k];
	C01[i*nz0*ny0+j*nz0+k] = C01[i*nz0*ny0+(j-1)*nz0+k];
	C02[i*nz0*ny0+j*nz0+k] = C02[i*nz0*ny0+(j-1)*nz0+k];
	C03[i*nz0*ny0+j*nz0+k] = C03[i*nz0*ny0+(j-1)*nz0+k];
	C04[i*nz0*ny0+j*nz0+k] = C04[i*nz0*ny0+(j-1)*nz0+k];
      }
  }
  else
  {
    putin_sendbuffer_xz( 1*nz0*ny0+(ny0-2)*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &xz_sbuf1[0*nx*nz],nx*nz);
    putin_sendbuffer_xz( 1*nz0*ny0+(ny0-2)*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &xz_sbuf1[1*nx*nz],nx*nz);
    putin_sendbuffer_xz( 1*nz0*ny0+(ny0-2)*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &xz_sbuf1[2*nx*nz],nx*nz);
    putin_sendbuffer_xz( 1*nz0*ny0+(ny0-2)*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &xz_sbuf1[3*nx*nz],nx*nz);
    putin_sendbuffer_xz( 1*nz0*ny0+(ny0-2)*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &xz_sbuf1[4*nx*nz],nx*nz);
  }
  MPI_Isend(xz_sbuf1,5*nx*nz,MPI_DOUBLE,neighbor[YP],C_flag+2000, comm, &ar_send_req[3]);
  MPI_Irecv(xz_rbuf1,5*nx*nz,MPI_DOUBLE,neighbor[YP],C_flag+2000, comm, &ar_recv_req[3]);
//  MPI_Sendrecv(xz_sbuf1,5*nx*nz,MPI_DOUBLE,neighbor[YP],C_flag+2000,
//               xz_rbuf1,5*nx*nz,MPI_DOUBLE,neighbor[YP],C_flag+2000,comm,&status);
  
//  printf("exchange Y end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
  // Ghost Z end sheet
  if(coords[0]==0){
    k=0;
    for (i=1; i<nx0-1; i++)
      for (j=1; j<ny0-1; j++){
	C00[i*nz0*ny0+j*nz0+k] = C00[i*nz0*ny0+j*nz0+k+1];
	C01[i*nz0*ny0+j*nz0+k] = C01[i*nz0*ny0+j*nz0+k+1];
	C02[i*nz0*ny0+j*nz0+k] = C02[i*nz0*ny0+j*nz0+k+1];
	C03[i*nz0*ny0+j*nz0+k] = C03[i*nz0*ny0+j*nz0+k+1];
	C04[i*nz0*ny0+j*nz0+k] = C04[i*nz0*ny0+j*nz0+k+1];
      }
  }
  else
  {
    putin_sendbuffer_xy(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &xy_sbuf0[0*nx*ny],nx*ny);
    putin_sendbuffer_xy(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &xy_sbuf0[1*nx*ny],nx*ny);
    putin_sendbuffer_xy(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &xy_sbuf0[2*nx*ny],nx*ny);
    putin_sendbuffer_xy(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &xy_sbuf0[3*nx*ny],nx*ny);
    putin_sendbuffer_xy(1*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &xy_sbuf0[4*nx*ny],nx*ny);
  }
  MPI_Isend(xy_sbuf0,5*nx*ny,MPI_DOUBLE,neighbor[ZN],C_flag+3000, comm, &ar_send_req[4]);
  MPI_Irecv(xy_rbuf0,5*nx*ny,MPI_DOUBLE,neighbor[ZN],C_flag+3000, comm, &ar_recv_req[4]);
//  MPI_Sendrecv(xy_sbuf0,5*nx*ny,MPI_DOUBLE,neighbor[ZN],C_flag+3000,
//               xy_rbuf0,5*nx*ny,MPI_DOUBLE,neighbor[ZN],C_flag+3000,comm,&status);
 

  if(coords[0]==(dims[0]-1))
  {
    k=nz0-1;
    for (i=1; i<nx0-1; i++)
      for (j=1; j<ny0-1; j++){
	C00[i*nz0*ny0+j*nz0+k] = C00[i*nz0*ny0+j*nz0+k-1];
	C01[i*nz0*ny0+j*nz0+k] = C01[i*nz0*ny0+j*nz0+k-1];
	C02[i*nz0*ny0+j*nz0+k] = C02[i*nz0*ny0+j*nz0+k-1];
	C03[i*nz0*ny0+j*nz0+k] = C03[i*nz0*ny0+j*nz0+k-1];
	C04[i*nz0*ny0+j*nz0+k] = C04[i*nz0*ny0+j*nz0+k-1];
      }
  }
  else
  {
    putin_sendbuffer_xy(  1*nz0*ny0+nz0+nz0-2,nx0,ny0, nz0, C00, nx, ny, nz, &xy_sbuf1[0*nx*ny],nx*ny);
    putin_sendbuffer_xy(  1*nz0*ny0+nz0+nz0-2,nx0,ny0, nz0, C01, nx, ny, nz, &xy_sbuf1[1*nx*ny],nx*ny);
    putin_sendbuffer_xy(  1*nz0*ny0+nz0+nz0-2,nx0,ny0, nz0, C02, nx, ny, nz, &xy_sbuf1[2*nx*ny],nx*ny);
    putin_sendbuffer_xy(  1*nz0*ny0+nz0+nz0-2,nx0,ny0, nz0, C03, nx, ny, nz, &xy_sbuf1[3*nx*ny],nx*ny);
    putin_sendbuffer_xy(  1*nz0*ny0+nz0+nz0-2,nx0,ny0, nz0, C04, nx, ny, nz, &xy_sbuf1[4*nx*ny],nx*ny);
  }
  MPI_Isend(xy_sbuf1,5*nx*ny,MPI_DOUBLE,neighbor[ZP],C_flag+3000, comm, &ar_send_req[5]);
  MPI_Irecv(xy_rbuf1,5*nx*ny,MPI_DOUBLE,neighbor[ZP],C_flag+3000, comm, &ar_recv_req[5]);
//  MPI_Sendrecv(xy_sbuf1,5*nx*ny,MPI_DOUBLE,neighbor[ZP],C_flag+3000,
//               xy_rbuf1,5*nx*ny,MPI_DOUBLE,neighbor[ZP],C_flag+3000,comm,&status);

  MPI_Waitall(6, ar_recv_req, ar_status); 

  if(coords[2]!=0){
    getout_recvbuffer_yz(               1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &yz_rbuf0[0*ny*nz],ny*nz);
    getout_recvbuffer_yz(               1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &yz_rbuf0[1*ny*nz],ny*nz);
    getout_recvbuffer_yz(               1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &yz_rbuf0[2*ny*nz],ny*nz);
    getout_recvbuffer_yz(               1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &yz_rbuf0[3*ny*nz],ny*nz);
    getout_recvbuffer_yz(               1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &yz_rbuf0[4*ny*nz],ny*nz);
  }

  if(coords[2]!=(dims[2]-1)){
    getout_recvbuffer_yz((nx0-1)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &yz_rbuf1[0*ny*nz],ny*nz);
    getout_recvbuffer_yz((nx0-1)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &yz_rbuf1[1*ny*nz],ny*nz);
    getout_recvbuffer_yz((nx0-1)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &yz_rbuf1[2*ny*nz],ny*nz);
    getout_recvbuffer_yz((nx0-1)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &yz_rbuf1[3*ny*nz],ny*nz);
    getout_recvbuffer_yz((nx0-1)*nz0*ny0+1*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &yz_rbuf1[4*ny*nz],ny*nz);
  }

  if(coords[1]!=0){
    getout_recvbuffer_xz(           1*nz0*ny0+1,nx0,ny0, nz0, C00, nx, ny, nz, &xz_rbuf0[0*nx*nz],nx*nz);
    getout_recvbuffer_xz(           1*nz0*ny0+1,nx0,ny0, nz0, C01, nx, ny, nz, &xz_rbuf0[1*nx*nz],nx*nz);
    getout_recvbuffer_xz(           1*nz0*ny0+1,nx0,ny0, nz0, C02, nx, ny, nz, &xz_rbuf0[2*nx*nz],nx*nz);
    getout_recvbuffer_xz(           1*nz0*ny0+1,nx0,ny0, nz0, C03, nx, ny, nz, &xz_rbuf0[3*nx*nz],nx*nz);
    getout_recvbuffer_xz(           1*nz0*ny0+1,nx0,ny0, nz0, C04, nx, ny, nz, &xz_rbuf0[4*nx*nz],nx*nz);
  }

  if(coords[1]!=(dims[1]-1)){
    getout_recvbuffer_xz(1*nz0*ny0+(ny0-1)*nz0+1,nx0,ny0, nz0, C00, nx, ny, nz, &xz_rbuf1[0*nx*nz],nx*nz);
    getout_recvbuffer_xz(1*nz0*ny0+(ny0-1)*nz0+1,nx0,ny0, nz0, C01, nx, ny, nz, &xz_rbuf1[1*nx*nz],nx*nz);
    getout_recvbuffer_xz(1*nz0*ny0+(ny0-1)*nz0+1,nx0,ny0, nz0, C02, nx, ny, nz, &xz_rbuf1[2*nx*nz],nx*nz);
    getout_recvbuffer_xz(1*nz0*ny0+(ny0-1)*nz0+1,nx0,ny0, nz0, C03, nx, ny, nz, &xz_rbuf1[3*nx*nz],nx*nz);
    getout_recvbuffer_xz(1*nz0*ny0+(ny0-1)*nz0+1,nx0,ny0, nz0, C04, nx, ny, nz, &xz_rbuf1[4*nx*nz],nx*nz);
  }

  if(coords[0]!=0){
    getout_recvbuffer_xy(     1*nz0*ny0+1*nz0,  nx0,ny0, nz0, C00, nx, ny, nz, &xy_rbuf0[0*nx*ny],nx*ny);
    getout_recvbuffer_xy(     1*nz0*ny0+1*nz0,  nx0,ny0, nz0, C01, nx, ny, nz, &xy_rbuf0[1*nx*ny],nx*ny);
    getout_recvbuffer_xy(     1*nz0*ny0+1*nz0,  nx0,ny0, nz0, C02, nx, ny, nz, &xy_rbuf0[2*nx*ny],nx*ny);
    getout_recvbuffer_xy(     1*nz0*ny0+1*nz0,  nx0,ny0, nz0, C03, nx, ny, nz, &xy_rbuf0[3*nx*ny],nx*ny);
    getout_recvbuffer_xy(     1*nz0*ny0+1*nz0,  nx0,ny0, nz0, C04, nx, ny, nz, &xy_rbuf0[4*nx*ny],nx*ny);
  }

  if(coords[0]!=(dims[0]-1)){
    getout_recvbuffer_xy( 1*nz0*ny0+nz0+nz0-1,nx0,ny0, nz0, C00, nx, ny, nz, &xy_rbuf1[0*nx*ny],nx*ny);
    getout_recvbuffer_xy( 1*nz0*ny0+nz0+nz0-1,nx0,ny0, nz0, C01, nx, ny, nz, &xy_rbuf1[1*nx*ny],nx*ny);
    getout_recvbuffer_xy( 1*nz0*ny0+nz0+nz0-1,nx0,ny0, nz0, C02, nx, ny, nz, &xy_rbuf1[2*nx*ny],nx*ny);
    getout_recvbuffer_xy( 1*nz0*ny0+nz0+nz0-1,nx0,ny0, nz0, C03, nx, ny, nz, &xy_rbuf1[3*nx*ny],nx*ny);
    getout_recvbuffer_xy( 1*nz0*ny0+nz0+nz0-1,nx0,ny0, nz0, C04, nx, ny, nz, &xy_rbuf1[4*nx*ny],nx*ny);
  }

//  printf("exchange Z end sheet ok! coords[%d,%d,%d]\n",coords[0],coords[1],coords[2]);
}

void putin_sendbuffer_yz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i;
  if(sbuf_len!=ny*nz)
  {
    printf("yz sbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < ny; i += 1 ) {
    memcpy(&sbuf[i*nz],&arr[base_addr+i*nz0],nz*sizeof(double));
  }
}

void putin_sendbuffer_xz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i;
  if(sbuf_len!=nx*nz)
  {
    printf("xz sbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < nx; i += 1 ) {
    memcpy(&sbuf[i*nz],&arr[base_addr+i*ny0*nz0],nz*sizeof(double));
  }
}

void putin_sendbuffer_xy(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i, j;
  if(sbuf_len!=nx*ny)
  {
    printf("xy sbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < nx; i += 1 ) {
    for ( j = 0; j < ny; j += 1 ) {
     sbuf[i*ny+j]=arr[base_addr+i*ny0*nz0+j*nz0];
    }
  }
}

void getout_recvbuffer_yz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i;
  if(sbuf_len!=ny*nz)
  {
    printf("yz rbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < ny; i += 1 ) {
    memcpy(&arr[base_addr+i*nz0],&sbuf[i*nz],nz*sizeof(double));
  }
}

void getout_recvbuffer_xz(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i;
  if(sbuf_len!=nx*nz)
  {
    printf("xz rbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < nx; i += 1 ) {
    memcpy(&arr[base_addr+i*ny0*nz0],&sbuf[i*nz],nz*sizeof(double));
  }
}

void getout_recvbuffer_xy(int base_addr,int nx0,int ny0, int nz0, double* arr,
    int nx, int ny, int nz, double* sbuf, int sbuf_len)
{
  int i, j;
  if(sbuf_len!=nx*ny)
  {
    printf("xy rbuf_len error!\n");
    exit(0);
  }
  for ( i = 0; i < nx; i += 1 ) {
    for ( j = 0; j < ny; j += 1 ) {
     arr[base_addr+i*ny0*nz0+j*nz0]=sbuf[i*ny+j];
    }
  }
}
   
void BinarySort_two(int* pData, int* vData, int Count) 
{     
  dichotomy_two(pData,vData,0,Count-1); 
}

void dichotomy_two(int* pData,int* vData, int left,int right) {
  int i,j;     
  int middle,iTemp;     
  i = left;     
  j = right;     
  middle = pData[(left+right)/2];
  do{       
    while((pData[i]<middle) && (i<right))
      i++;      
      while((pData[j]>middle) && (j>left))
	j--;       
      if(i<=j)
      {       
	iTemp = pData[i];         
	pData[i] = pData[j];         
	pData[j] = iTemp;         
	iTemp   =vData[i];
	vData[i]=vData[j];
	vData[j]=iTemp;
	i++;        
	j--;       
      }    
  }while(i<=j);
  if(left<j)      
    dichotomy_two(pData,vData,left,j);     
  if(right>i)      
    dichotomy_two(pData,vData,i,right);
}

void *mpi_malloc (
		  int id,     /* IN - Process rank */
		  int bytes)  /* IN - Bytes to allocate */
{
  void *buffer;
  if ((buffer = malloc ((size_t) bytes)) == NULL) {
    printf ("Error: Malloc failed for process %d\n", id);
    fflush (stdout);
    MPI_Abort (MPI_COMM_WORLD, 4);
  }
  return buffer;
}


void compute_pde_ode(int nx0, int ny0, int nz0, double dt,double gamma, double fudge,
    double* alpha, double* B_tot, double* k_on, double* k_off,
    double** C0, double** C1, int div_y)
{
  // Main kernel
  int i,j,k,jj,idx;
  int ny;
  double J;
  double Ca_ijk;
  double buff_ijk;
  double Ca_i2_ijk;
  double Ca_SR2_ijk;
  ny=ny0-2;
  for (i=1; i<nx0-1; i++)
  {
    for (jj=0; jj<ny/div_y; jj++)
    {
      //blocking for cache size on y line
      for (j=jj*div_y+1; j<(jj+1)*div_y+1; j++)
      {
	//Laplace diffusion process five array together
	for(idx=0;idx<5;idx++)
	{
#pragma ivdep 
#pragma prefetch
	  for (k=1; k<nz0-1; k++)
	  {
	    C1[idx][i*nz0*ny0+j*nz0+k] =alpha[idx]*(
		C0[idx][i*nz0*ny0+j*nz0+k]*(-6)+ 
		C0[idx][(i-1)*nz0*ny0+j*nz0+k] + C0[idx][(i+1)*nz0*ny0+j*nz0+k] + 
		C0[idx][i*nz0*ny0+(j-1)*nz0+k] + C0[idx][i*nz0*ny0+(j+1)*nz0+k] + 
		C0[idx][i*nz0*ny0+j*nz0+k-1]   + C0[idx][i*nz0*ny0+j*nz0+k+1])  +
	      C0[idx][i*nz0*ny0+j*nz0+k];
	  }
	}
	//Reaction
	for(idx=2;idx<6;idx++)
	{
#pragma ivdep 
#pragma prefetch
	  for (k=1; k<nz0-1; k++)
	  {
	    Ca_ijk   = C1[0][i*nz0*ny0+j*nz0+k];
	    buff_ijk = C1[idx][i*nz0*ny0+j*nz0+k];
	    J = k_on[idx]*(B_tot[idx] - buff_ijk)*Ca_ijk - k_off[idx]*buff_ijk;
	    C1[0][i*nz0*ny0+j*nz0+k] -= dt*J;
	    C1[idx][i*nz0*ny0+j*nz0+k] += dt*J;
	  }
	}
	//	serca3D
#pragma ivdep 
#pragma prefetch
	for (k=1; k<nz0-1; k++)
	{
	  // Main kernel
	  Ca_i2_ijk = C1[0][i*nz0*ny0+j*nz0+k];
	  Ca_SR2_ijk = C1[1][i*nz0*ny0+j*nz0+k];
	  Ca_i2_ijk *= Ca_i2_ijk;
	  Ca_SR2_ijk *= Ca_SR2_ijk;
	  J = fudge*(570997802.885875*Ca_i2_ijk - 0.0425239333622699*Ca_SR2_ijk)/(106720651.206402*Ca_i2_ijk + 182.498197548666*Ca_SR2_ijk + 5.35062954944879);
	  C1[0][i*nz0*ny0+j*nz0+k] -= dt*J;
	  C1[1][i*nz0*ny0+j*nz0+k] += dt*J/gamma;
	}
      }
    }  
  }
}





