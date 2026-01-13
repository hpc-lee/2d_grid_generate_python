#include "grid_interface.h"
#include "parabolic.h"
#include "hyperbolic.h"

int para_gene_c(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b) {
    return para_gene(x2d, z2d, step, nx, nz, coef, t2b);
}

int hyper_gene_c(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b, int flag_stretch) {
    return hyper_gene(x2d, z2d, step, nx, nz, coef, t2b, flag_stretch);
}