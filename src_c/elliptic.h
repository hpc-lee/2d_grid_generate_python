#ifndef ELLIPTIC_H
#define ELLIPTIC_H

/*************************************************
 * function prototype
 *************************************************/

void interp_inner_source_c(float *P, float *P_x1, float *P_x2, 
                           float *P_z1, float *P_z2, float *Q, 
                           float *Q_x1, float *Q_x2, float *Q_z1, 
                           float *Q_z2, int nx, int nz, int gni1, 
                           int gnk1, int total_nx, int  total_nz, 
                           float *coef, float *weight);

                   
void compute_residual_c(float *x2d, float *z2d, float *x2d_tmp, 
                        float *z2d_tmp, float *local_max, 
                        int nx, int nz);                         

 #endif