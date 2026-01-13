#ifndef GRID_INTERFACE_H
#define GRID_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// 暴露给Python的函数
int para_gene_c(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b);
int hyper_gene_c(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b, int flag_stretch);

#ifdef __cplusplus
}
#endif

#endif