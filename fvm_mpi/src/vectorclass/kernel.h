typedef unsigned char domain_id;
typedef double REAL;

void kernel_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5);

void kernel_alpha_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, double *alphax, double *alphay, double *alphaz);

void kernel_auto(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5);

void kernel_block(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5,double *alphax, double *alphay, double *alphaz);
