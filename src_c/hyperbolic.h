#ifndef HYPERBOLIC_H
#define HYPERBOLIC_H

/*************************************************
 * function prototype
 *************************************************/

void 
hyper_gene_c(float *x2d, float *z2d, float *step, int nx, 
             int nz, float coef, int t2b, int flag_stretch);

int
cal_smooth_coef(float coef, float *x2d, float *z2d,
                int nx, int nz, int k, int t2b, float *coef_e);

int 
cal_matrix(float *x2d, float *z2d, int nx, int k, float *step,
           float *a, float *b, float *c, float *d, float *area);

int
modify_matrix(float *x2d, float *z2d, int nx, int k, float *a,
              float *b, float *c, float *d, float *coef_e);

int
thomas_block(int n, float *a, float *b, float *c, float *d, 
             float *xz, float *D, float *y);

int
assign_coords(float *xz, float *x2d, float *z2d, int nx, int nz, int k);

#endif
