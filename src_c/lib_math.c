#include <stdio.h>
#include <math.h>

#include "lib_math.h"

int
mat_invert2x2(float matrix[][2])
{
  for (int k=0; k<2; k++)
  {
     float con = matrix[k][k];
     matrix[k][k] = 1.0;

     for (int i=0; i<2; i++) {
       matrix[k][i] = matrix[k][i]/con;
     }

     for (int i=0; i<2; i++)
     {
        if (i!=k) {
           con = matrix[i][k];
           matrix[i][k] = 0.0;
           for (int j=0; j<2; j++) {
             matrix[i][j] = matrix[i][j] - matrix[k][j] * con;
           }
        }
     }
  }

  return 0;
}

int
mat_mul2x2(float A[][2], float B[][2], float C[][2])
{
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++){
      C[i][j] = 0.0;
      for (int k=0; k<2; k++)
        C[i][j] += A[i][k] * B[k][j];
    }

  return 0;
}

int
mat_mul2x1(float A[][2], float *B, float *C)
{
  C[0] = A[0][0]*B[0] + A[0][1]*B[1];
  C[1] = A[1][0]*B[0] + A[1][1]*B[1];

  return 0;
}

int
mat_add2x2(float A[][2], float B[][2], float C[][2])
{
  for (int i=0; i<2; i++){
    for (int j=0; j<2; j++){
        C[i][j] = A[i][j] + B[i][j];
    }
  }

  return 0;
}

int
vec_add2x1(float *A, float *B, float *C)
{
  for (int i=0; i<2; i++){
    C[i] = A[i] + B[i];
  }

  return 0;
}

int
vec_sub2x1(float *A, float *B, float *C)
{
  for (int i=0; i<2; i++){
    C[i] = A[i] - B[i];
  }

  return 0;
}

int
mat_sub2x2(float A[][2], float B[][2], float C[][2])
{
  for (int i=0; i<2; i++){
    for (int j=0; j<2; j++){
        C[i][j] = A[i][j] - B[i][j];
    }
  }

  return 0;
}

int
mat_copy2x2(float A[][2], float B[][2])
{
  for (int i=0; i<2; i++){
    for (int j=0; j<2; j++){
        B[i][j] = A[i][j];
    }
  }

  return 0;
}

int
mat_iden2x2(float A[][2])
{
  for (int i=0; i<2; i++){
    for (int j=0; j<2; j++){
      A[i][j] = 0.0;
      if(i==j) {
        A[i][j] = 1;
      }
    }
  }

  return 0;
}