#ifndef UTILS_H
#define UTILS_H

#define CONST_NDIM  2
#define PI 3.14159265358979323846264

int 
zt_arc_stretch(float *x2d, float *z2d, float *step, int nx, int nz);

int
flip_coord_z(float *x2d, float *z2d, int nx, int nz);

int
flip_step_z(float *step, int nz);

#endif