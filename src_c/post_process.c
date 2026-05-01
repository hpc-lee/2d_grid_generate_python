#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "post_process.h"
#include "lib_math.h"

/* grid sample, linear interpolation (from algebra.c) */
void
sample_interp_c(float *x2d, float *z2d,
                float *x2d_new, float *z2d_new,
                int nx, int nz, int nx_new, int nz_new)
{
  float *x2d_temp = (float *)malloc(sizeof(float) * nx);
  float *z2d_temp = (float *)malloc(sizeof(float) * nx);
  float *u = (float *)malloc(sizeof(float) * nz);
  float *v = (float *)malloc(sizeof(float) * nx);
  int n;
  float r, ratio, x_len, z_len;
  size_t iptr, iptr1, iptr2;

  /* first interp zt direction, line by line i=0..nx-1 */
  for (int i = 0; i < nx; i++)
  {
    for (int k = 0; k < nz; k++)
      u[k] = (1.0f * k) / (nz - 1);

    for (int k_new = 0; k_new < nz_new; k_new++)
    {
      r = (1.0f * k_new) / (nz_new - 1);
      for (int m = 0; m < nz - 1; m++)
      {
        if (r >= u[m] && r < u[m + 1])
        {
          n = m;
          break;
        }
      }

      iptr = k_new * nx_new + i;
      iptr1 = n * nx + i;
      iptr2 = (n + 1) * nx + i;
      x_len = x2d[iptr2] - x2d[iptr1];
      z_len = z2d[iptr2] - z2d[iptr1];
      ratio = (r - u[n]) / (u[n + 1] - u[n]);
      x2d_new[iptr] = x2d[iptr1] + x_len * ratio;
      z2d_new[iptr] = z2d[iptr1] + z_len * ratio;
    }
  }

  /* then interp xi direction, line by line k=0..nz_new-1 */
  for (int k_new = 0; k_new < nz_new; k_new++)
  {
    for (int i = 0; i < nx; i++)
    {
      iptr1 = k_new * nx_new + i;
      x2d_temp[i] = x2d_new[iptr1];
      z2d_temp[i] = z2d_new[iptr1];
    }
    for (int i = 0; i < nx; i++)
      v[i] = (1.0f * i) / (nx - 1);

    for (int i_new = 0; i_new < nx_new; i_new++)
    {
      r = (1.0f * i_new) / (nx_new - 1);
      for (int m = 0; m < nx - 1; m++)
      {
        if (r >= v[m] && r < v[m + 1])
        {
          n = m;
          break;
        }
      }

      iptr = k_new * nx_new + i_new;
      x_len = x2d_temp[n + 1] - x2d_temp[n];
      z_len = z2d_temp[n + 1] - z2d_temp[n];
      ratio = (r - v[n]) / (v[n + 1] - v[n]);
      x2d_new[iptr] = x2d_temp[n] + x_len * ratio;
      z2d_new[iptr] = z2d_temp[n] + z_len * ratio;
    }
  }

  free(u);
  free(v);
  free(x2d_temp);
  free(z2d_temp);
}

/* calculate minimum distance for CFL stability (from gd_t.c) */
void
cal_min_dist_c(float *x2d, float *z2d, int nx, int nz,
               int *indx_i, int *indx_k, float *dL_min)
{
  float dL_min_local, dL_min_global = 1e10f;
  size_t siz_iz = nx;
  size_t iptr;

  *indx_i = 1;
  *indx_k = 1;
  *dL_min = dL_min_global;

  for (int k = 1; k < nz - 1; k++)
  {
    for (int i = 1; i < nx - 1; i++)
    {
      iptr = i + k * siz_iz;
      float p0[] = {x2d[iptr], z2d[iptr]};
      dL_min_local = 1e10f;

      for (int kk = -1; kk <= 1; kk += 2)
      {
        for (int ii = -1; ii <= 1; ii += 2)
        {
          float p1[] = {x2d[iptr - ii], z2d[iptr - ii]};
          float p2[] = {x2d[iptr - kk * siz_iz],
                        z2d[iptr - kk * siz_iz]};

          float L = dist_point2line(p0, p1, p2);
          if (dL_min_local > L)
            dL_min_local = L;
        }
      }

      if (dL_min_global > dL_min_local)
      {
        dL_min_global = dL_min_local;
        *dL_min = dL_min_global;
        *indx_i = i;
        *indx_k = k;
      }
    }
  }
}

/* ---- Quality check functions (from quality_check.c) ---- */

void
cal_orth_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  float trans = 180.0f / 3.14159265358979323846f;
  size_t iptr, iptr1, iptr2;
  float x_xi, z_xi, x_zt, z_zt;
  float dot, len_xi, len_zt, cos_angle;

  for (int k = 0; k < nz - 1; k++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      iptr = k * nx + i;
      iptr1 = k * nx + (i + 1);
      x_xi = x2d[iptr1] - x2d[iptr];
      z_xi = z2d[iptr1] - z2d[iptr];

      iptr2 = (k + 1) * nx + i;
      x_zt = x2d[iptr2] - x2d[iptr];
      z_zt = z2d[iptr2] - z2d[iptr];

      dot = x_xi * x_zt + z_xi * z_zt;
      len_xi = sqrtf(x_xi * x_xi + z_xi * z_xi);
      len_zt = sqrtf(x_zt * x_zt + z_zt * z_zt);
      cos_angle = dot / (len_xi * len_zt);
      var[iptr] = 90.0f - fabsf(acosf(cos_angle) * trans - 90.0f);
    }
  }

  for (int k = 0; k < nz; k++)
    var[k * nx + (nx - 1)] = var[k * nx + (nx - 2)];
  for (int i = 0; i < nx; i++)
    var[(nz - 1) * nx + i] = var[(nz - 2) * nx + i];
}

void
cal_jacobi_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1, iptr2;
  float x_xi, z_xi, x_zt, z_zt;

  for (int k = 0; k < nz - 1; k++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      iptr = k * nx + i;
      iptr1 = k * nx + (i + 1);
      x_xi = x2d[iptr1] - x2d[iptr];
      z_xi = z2d[iptr1] - z2d[iptr];

      iptr2 = (k + 1) * nx + i;
      x_zt = x2d[iptr2] - x2d[iptr];
      z_zt = z2d[iptr2] - z2d[iptr];

      var[iptr] = x_xi * z_zt - z_xi * x_zt;
    }
  }

  for (int k = 0; k < nz; k++)
    var[k * nx + (nx - 1)] = var[k * nx + (nx - 2)];
  for (int i = 0; i < nx; i++)
    var[(nz - 1) * nx + i] = var[(nz - 2) * nx + i];
}

void
cal_ratio_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1, iptr2;
  float x_xi, z_xi, x_zt, z_zt;
  float r1, r2, len_xi, len_zt;

  for (int k = 0; k < nz - 1; k++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      iptr = k * nx + i;
      iptr1 = k * nx + (i + 1);
      x_xi = x2d[iptr1] - x2d[iptr];
      z_xi = z2d[iptr1] - z2d[iptr];

      iptr2 = (k + 1) * nx + i;
      x_zt = x2d[iptr2] - x2d[iptr];
      z_zt = z2d[iptr2] - z2d[iptr];

      len_xi = sqrtf(x_xi * x_xi + z_xi * z_xi);
      len_zt = sqrtf(x_zt * x_zt + z_zt * z_zt);

      r1 = len_xi / len_zt;
      r2 = len_zt / len_xi;
      var[iptr] = fmaxf(r1, r2);
    }
  }

  for (int k = 0; k < nz; k++)
    var[k * nx + (nx - 1)] = var[k * nx + (nx - 2)];
  for (int i = 0; i < nx; i++)
    var[(nz - 1) * nx + i] = var[(nz - 2) * nx + i];
}

void
cal_step_x_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1;
  float x_xi, z_xi, len_xi;

  for (int k = 0; k < nz; k++)
  {
    for (int i = 0; i < nx - 1; i++)
    {
      iptr = k * nx + i;
      iptr1 = k * nx + (i + 1);
      x_xi = x2d[iptr1] - x2d[iptr];
      z_xi = z2d[iptr1] - z2d[iptr];
      len_xi = sqrtf(x_xi * x_xi + z_xi * z_xi);
      var[iptr] = len_xi;
    }
  }

  for (int k = 0; k < nz; k++)
    var[k * nx + (nx - 1)] = var[k * nx + (nx - 2)];
}

void
cal_step_z_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1;
  float x_zt, z_zt, len_zt;

  for (int k = 0; k < nz - 1; k++)
  {
    for (int i = 0; i < nx; i++)
    {
      iptr = k * nx + i;
      iptr1 = (k + 1) * nx + i;
      x_zt = x2d[iptr1] - x2d[iptr];
      z_zt = z2d[iptr1] - z2d[iptr];
      len_zt = sqrtf(x_zt * x_zt + z_zt * z_zt);
      var[iptr] = len_zt;
    }
  }

  for (int i = 0; i < nx; i++)
    var[(nz - 1) * nx + i] = var[(nz - 2) * nx + i];
}

void
cal_smooth_x_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1, iptr2;
  float x_xi1, z_xi1, x_xi2, z_xi2;
  float r1, r2, len_xi1, len_xi2;

  for (int k = 0; k < nz; k++)
  {
    for (int i = 1; i < nx - 1; i++)
    {
      iptr = k * nx + i;
      iptr1 = k * nx + (i - 1);
      iptr2 = k * nx + (i + 1);

      x_xi1 = x2d[iptr] - x2d[iptr1];
      z_xi1 = z2d[iptr] - z2d[iptr1];
      x_xi2 = x2d[iptr2] - x2d[iptr];
      z_xi2 = z2d[iptr2] - z2d[iptr];

      len_xi1 = sqrtf(x_xi1 * x_xi1 + z_xi1 * z_xi1);
      len_xi2 = sqrtf(x_xi2 * x_xi2 + z_xi2 * z_xi2);

      r1 = len_xi1 / len_xi2;
      r2 = len_xi2 / len_xi1;
      var[iptr] = fmaxf(r1, r2);
    }
  }

  for (int k = 0; k < nz; k++)
    var[k * nx] = var[k * nx + 1];
  for (int k = 0; k < nz; k++)
    var[k * nx + (nx - 1)] = var[k * nx + (nx - 2)];
}

void
cal_smooth_z_c(float *x2d, float *z2d, float *var, int nx, int nz)
{
  size_t iptr, iptr1, iptr2;
  float x_zt1, z_zt1, x_zt2, z_zt2;
  float r1, r2, len_zt1, len_zt2;

  for (int k = 1; k < nz - 1; k++)
  {
    for (int i = 0; i < nx; i++)
    {
      iptr = k * nx + i;
      iptr1 = (k - 1) * nx + i;
      iptr2 = (k + 1) * nx + i;

      x_zt1 = x2d[iptr] - x2d[iptr1];
      z_zt1 = z2d[iptr] - z2d[iptr1];
      x_zt2 = x2d[iptr2] - x2d[iptr];
      z_zt2 = z2d[iptr2] - z2d[iptr];

      len_zt1 = sqrtf(x_zt1 * x_zt1 + z_zt1 * z_zt1);
      len_zt2 = sqrtf(x_zt2 * x_zt2 + z_zt2 * z_zt2);

      r1 = len_zt1 / len_zt2;
      r2 = len_zt2 / len_zt1;
      var[iptr] = fmaxf(r1, r2);
    }
  }

  for (int i = 0; i < nx; i++)
    var[i] = var[nx + i];
  for (int i = 0; i < nx; i++)
    var[(nz - 1) * nx + i] = var[(nz - 2) * nx + i];
}
