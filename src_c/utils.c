#include <stdlib.h>
#include <math.h>
#include "lib_mem.h"

// 2D array flip z direction.  nz-1->0 0->nz-1 i->(nz-1)-i 
int
flip_coord_z(float *x2d, float *z2d, int nx, int nz)
{
  size_t iptr,iptr1;

  float *tmp_coord_x = NULL;
  tmp_coord_x = (float *) malloc(nx*nz*sizeof(float));
  float *tmp_coord_z = NULL;
  tmp_coord_z = (float *) malloc(nx*nz*sizeof(float));
  // copy data
  for(int k=0; k<nz; k++) {
    for(int i=0; i<nx; i++) 
    {
      iptr = k*nx + i;
      tmp_coord_x[iptr] = x2d[iptr];
      tmp_coord_z[iptr] = z2d[iptr];
    }
  }
  // flip coord
  for(int k=0; k<nz; k++) {
    for(int i=0; i<nx; i++) 
    {
      iptr = k*nx + i;
      iptr1 = (nz-1-k)*nx + i;
      x2d[iptr] = tmp_coord_x[iptr1];
      z2d[iptr] = tmp_coord_z[iptr1];
    }
  }

  free(tmp_coord_x);
  free(tmp_coord_z);

  return 0;
}

int
flip_step_z(float *step, int nz)
{
  float *step_tmp = (float *)malloc((nz-1)*sizeof(float));
  for(int k=0; k<nz-1; k++)
  {
    step_tmp[k] = step[k]; 
  }
  for(int k=0; k<nz-1; k++)
  {
    step[nz-2-k] = step_tmp[k]; 
  }

  free(step_tmp);

  return 0;
}

int 
zt_arc_stretch(float *x2d, float *z2d, float *step, int nx, int nz)
{
  size_t iptr,iptr1,iptr2;
  float x_len,z_len,dh_len;
  float r, ratio, zeta;
  int n;

  float *x2d_temp = (float *)mem_calloc_1d_float(
              nz, 0.0, "init");
  float *z2d_temp = (float *)mem_calloc_1d_float(
              nz, 0.0, "init");
  float *arc_len  = (float *)mem_calloc_1d_float(
              nz, 0.0, "init");
  float *s = (float *)mem_calloc_1d_float(
              nz, 0.0, "init");
  float *u = (float *)mem_calloc_1d_float(
              nz, 0.0, "init");

  float arc_len_sum = 0;
  for(int k=1; k<nz; k++)
  {
    arc_len[k] = arc_len[k-1] + step[k-1]; 
    arc_len_sum += step[k-1];
  }

  for(int k=0; k<nz; k++)
  {
    arc_len[k] /= arc_len_sum; 
  }

  // line by line. i=0 -> i=nx-1
  for(int i=0; i<nx; i++)
  {
    // copy old coords to temp space
    for(int k=0; k<nz; k++)
    {
      iptr1 = k*nx + i;     //(i,k)
      x2d_temp[k] = x2d[iptr1];
      z2d_temp[k] = z2d[iptr1];
    }
    // cal arc length
    for(int k=1; k<nz; k++)
    {
      x_len = x2d_temp[k] - x2d_temp[k-1];
      z_len = z2d_temp[k] - z2d_temp[k-1];
      dh_len = sqrt(pow(x_len,2) + pow(z_len,2));
      s[k] = s[k-1] + dh_len;
    }
    // arc length normalized
    for(int k=0; k<nz; k++)
    {
      u[k] = s[k]/s[nz-1];
    }
    for(int k=1; k<nz-1; k++)
    {
      r = arc_len[k];
      for(int m=0; m<nz-1; m++)
      {
        if(r>=u[m] && r<u[m+1]) {
          n=m; 
          break;
        }
      }

      // linear interp
      iptr = k*nx + i;
      x_len = x2d_temp[n+1] - x2d_temp[n];
      z_len = z2d_temp[n+1] - z2d_temp[n];
      ratio = (r - u[n])/(u[n+1]-u[n]);
      x2d[iptr] = x2d_temp[n] + x_len*ratio;
      z2d[iptr] = z2d_temp[n] + z_len*ratio;
    }
  }

  free(x2d_temp);
  free(z2d_temp);
  free(arc_len);
  free(s);
  free(u);

  return 0;
}

