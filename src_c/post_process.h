#ifndef POST_PROCESS_H
#define POST_PROCESS_H

void sample_interp_c(float *x2d, float *z2d,
                     float *x2d_new, float *z2d_new,
                     int nx, int nz, int nx_new, int nz_new);

void cal_min_dist_c(float *x2d, float *z2d, int nx, int nz,
                    int *indx_i, int *indx_k, float *dL_min);

void cal_orth_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_jacobi_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_ratio_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_step_x_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_step_z_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_smooth_x_c(float *x2d, float *z2d, float *var, int nx, int nz);
void cal_smooth_z_c(float *x2d, float *z2d, float *var, int nx, int nz);

#endif
