#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "lib_math.h"

typedef float mat2x2_t[2][2];
typedef float vec2_t[2];

int mat_invert2x2(mat2x2_t matrix)
{
    const float det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    
    const float eps = 1e-8f;
    if (fabsf(det) < eps) {
        return -1; 
    }
    
    const float inv_det = 1.0f / det;
    
    const float a = matrix[0][0];
    const float b = matrix[0][1];
    const float c = matrix[1][0];
    const float d = matrix[1][1];
    
    matrix[0][0] = d * inv_det;
    matrix[0][1] = -b * inv_det;
    matrix[1][0] = -c * inv_det;
    matrix[1][1] = a * inv_det;
    
    return 0;
}

int mat_mul2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C)
{
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
    
    return 0;
}

int mat_mul2x1(const mat2x2_t A, const vec2_t B, vec2_t C)
{
    C[0] = A[0][0] * B[0] + A[0][1] * B[1];
    C[1] = A[1][0] * B[0] + A[1][1] * B[1];
    
    return 0;
}

int mat_add2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C)
{
    C[0][0] = A[0][0] + B[0][0];
    C[0][1] = A[0][1] + B[0][1];
    C[1][0] = A[1][0] + B[1][0];
    C[1][1] = A[1][1] + B[1][1];
    
    return 0;
}

int vec_add2x1(const vec2_t A, const vec2_t B, vec2_t C)
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    
    return 0;
}

int vec_sub2x1(const vec2_t A, const vec2_t B, vec2_t C)
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    
    return 0;
}

int mat_sub2x2(const mat2x2_t A, const mat2x2_t B, mat2x2_t C)
{
    C[0][0] = A[0][0] - B[0][0];
    C[0][1] = A[0][1] - B[0][1];
    C[1][0] = A[1][0] - B[1][0];
    C[1][1] = A[1][1] - B[1][1];
    
    return 0;
}

void mat_copy2x2(const mat2x2_t A, mat2x2_t B)
{
    B[0][0] = A[0][0];
    B[0][1] = A[0][1];
    B[1][0] = A[1][0];
    B[1][1] = A[1][1];
}

void mat_iden2x2(mat2x2_t A)
{
    A[0][0] = 1.0f;
    A[0][1] = 0.0f;
    A[1][0] = 0.0f;
    A[1][1] = 1.0f;
}

float
dist_point2line(float p0[2], float p1[2], float p2[2])
{
  float A, B, C;

  A = p2[1] - p1[1];
  B = p1[0] - p2[0];
  C = -p1[1] * B - p1[0] * A;

  return fabsf((A * p0[0] + B * p0[1] + C) / sqrtf(A * A + B * B));
}