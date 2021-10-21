
void kernel_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5);

void kernel_alpha_SIMD(double *u1, double *du, double *coef_table, const domain_id *dm_ids, domain_id** ghost_dm_ids, char* stencil_test, int nx, int ny, int nz, unsigned char ghosted4, unsigned char ghosted5, double *alphax, double *alphay, double *alphaz);

void side_SIMD(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset0, int offset20, int incr1, int incr2, int incr3, int ghosted, int len2);

void side_ghosted_SIMD(const REAL* u1, REAL* du, REAL* coef_table, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids, int dsi, int offset0, int offset20, int incr1, int incr2, int incr3, int ghosted, int len2);

void innerZ_SIMD(const REAL *u1, REAL *du, REAL *coef_table, int len2, int idx1, int incr1, int incr2, int offset2, const REAL* ghost_value_array, const domain_id* dm_ids, domain_id* ghost_dm_ids);

void innerZ_join_SIMD(const REAL *u1, REAL *du, REAL *coef_table, int len2, int idx1, int incr1, int incr2, const REAL* ghost_value_array4, const REAL* ghost_value_array5, const domain_id* dm_ids, domain_id** ghost_dm_ids);

void reaction_SIMD(double **u1, double **du, double * k_onT, double * k_offT, double *totT, domain_id* bsp0T, domain_id* bsp1T, int bi, int offset2, REAL dt, const domain_id* dm_ids, int ndomains, int nz,  domain_id* update_species);

void init_stencil_test2(const domain_id* dm_ids, char * voxels_stencil_test, int nx, int ny, int nz);

void apply_du(REAL *u1, REAL *du, int total_n, int nx, int ny, int nz);
