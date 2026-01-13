#ifndef PARABOLIC_H
#define PARABOLIC_H

/*************************************************
 * function prototype
 *************************************************/

int 
para_gene(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b);

int 
predict_point(float *x2d, float *z2d, int nx, int nz, int k, int t2b, 
              float coef, float *step_len, float *x_pre, float *z_pre);

int
update_point(float *x2d, float *z2d, float *var_th, int nx, int k,
             float *x_pre, float *z_pre);

int 
assign_bdry_coords(float *x2d, float *z2d, int nx, int k);

int
thomas(int n, float *a, float *b, float *c, float *d_x, 
       float *d_z, float *u_x, float *u_z);

#endif
