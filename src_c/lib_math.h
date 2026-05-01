#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <stdint.h>

typedef float mat2x2_t[2][2];

typedef float vec2_t[2];

int mat_invert2x2(mat2x2_t matrix);

int mat_mul2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C);

int mat_mul2x1(const mat2x2_t A, const vec2_t B, vec2_t C);

int mat_add2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C);

int vec_add2x1(const vec2_t A, const vec2_t B, vec2_t C);

int vec_sub2x1(const vec2_t A, const vec2_t B, vec2_t C);

int mat_sub2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C);

void mat_copy2x2(const mat2x2_t A, mat2x2_t B);

void mat_iden2x2(mat2x2_t A);

float
dist_point2line(float p0[2], float p1[2], float p2[2]);

#endif 
