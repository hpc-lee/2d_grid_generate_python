#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "elliptic.h"


void interp_inner_source_c(float *P, float *P_x1, float *P_x2, 
                           float *P_z1, float *P_z2, float *Q, 
                           float *Q_x1, float *Q_x2, float *Q_z1, 
                           float *Q_z2, int nx, int nz, int gni1, 
                           int gnk1, int total_nx, int  total_nz, 
                           float *coef, float *weight)
{
  int gni, gnk;
  float xi,zt,c0,c1,r0,r1;
  size_t iptr;
  for(int k=1; k<nz-1; k++) {
    for(int i=1; i<nx-1; i++)
    {
      gnk = gnk1 + k-1; 
      gni = gni1 + i;
      xi = (1.0*gni)/(total_nx-1);

      c0 = 1-xi;
      c1 = xi;
      r0 = exp(-coef[0]*xi);
      r1 = exp(-coef[1]*(1-xi)); 
      
      iptr  = k*nx + i;
      P[iptr] = weight[0]*(c0*P_x1[gnk] + c1*P_x2[gnk]);
      Q[iptr] = weight[0]*(r0*Q_x1[gnk] + r1*Q_x2[gnk]);
    }
  }

  for(int k=1; k<nz-1; k++) {
    for(int i=1; i<nx-1; i++)
    {
      gnk = gnk1 + k; 
      gni = gni1 + i-1;
      zt = (1.0*gnk)/(total_nz-1);
      c0 = 1-zt;
      c1 = zt;
      r0 = exp(-coef[2]*zt);
      r1 = exp(-coef[3]*(1-zt)); 
      
      iptr  = k*nx + i;
      P[iptr] = P[iptr] + weight[1]*(r0*P_z1[gni] + r1*P_z2[gni]);
      Q[iptr] = Q[iptr] + weight[1]*(c0*Q_z1[gni] + c1*Q_z2[gni]);
    }
  }

  return;
}

void compute_residual_c(float *x2d, float *z2d, float *x2d_tmp, 
                        float *z2d_tmp, float *local_max, 
                        int nx, int nz)
{
    float dif1, dif2, dif3, dif_x, dif_z;
    size_t iptr, iptr1, iptr2; 
    float resi, resk;
    float max_resi = 0.0f;
    float max_resk = 0.0f;

    for(int k=1; k<nz-1; k++) {
        for(int i=1; i<nx-1; i++)
        {
          iptr  = k*nx + i;
          iptr1 = k*nx + i+1;
          iptr2 = (k+1)*nx + i;
          dif_x = x2d_tmp[iptr]-x2d[iptr];
          dif_z = z2d_tmp[iptr]-z2d[iptr];
          dif1 = sqrt(dif_x * dif_x + dif_z * dif_z);
          dif_x = x2d[iptr1]-x2d[iptr];
          dif_z = z2d[iptr1]-z2d[iptr];
          dif2 = sqrt(dif_x * dif_x + dif_z * dif_z);
          dif_x = x2d[iptr2]-x2d[iptr];
          dif_z = z2d[iptr2]-z2d[iptr];
          dif3 = sqrt(dif_x * dif_x + dif_z * dif_z);
          resi = dif1/dif2;
          resk = dif1/dif3;
          max_resi = fmax(max_resi,resi);
          max_resk = fmax(max_resk,resk);
        }
    }

    local_max[0] = max_resi;
    local_max[1] = max_resk;

    return;
}