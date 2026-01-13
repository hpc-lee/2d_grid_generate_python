#ifndef LIB_MATH_H
#define LIB_MATH_H

int
mat_invert2x2(float matrix[][2]);

int
mat_mul2x2(float A[][2], float B[][2], float C[][2]);

int
mat_mul2x1(float A[][2], float *B, float *C);

int
mat_add2x2(float A[][2], float B[][2], float C[][2]);

int
vec_add2x1(float *A, float *B, float *C);

int
vec_sub2x1(float *A, float *B, float *C);

int
mat_sub2x2(float A[][2], float B[][2], float C[][2]);

int
mat_copy2x2(float A[][2], float B[][2]);

int
mat_iden2x2(float A[][2]);

#endif
