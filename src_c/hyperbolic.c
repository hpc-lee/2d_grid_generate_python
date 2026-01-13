#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include "hyperbolic.h"
#include "utils.h"
#include "lib_mem.h"
#include "lib_math.h"


int 
hyper_gene(float *x2d, float *z2d, float *step, int nx, int nz, float coef, int t2b, int flag_stretch)
{
  int n = nx-2;  // not include bdry 2 points

  float *coef_e = (float *)mem_calloc_1d_float(nx, 0.0, "init");
  float *area = (float *)mem_calloc_1d_float(nx*2, 0.0, "init");
  // malloc space for thomas_block method 
  float *a  = (float *)mem_calloc_1d_float(n*CONST_NDIM*CONST_NDIM, 0.0, "init");
  float *b  = (float *)mem_calloc_1d_float(n*CONST_NDIM*CONST_NDIM, 0.0, "init");
  float *c  = (float *)mem_calloc_1d_float(n*CONST_NDIM*CONST_NDIM, 0.0, "init");
  float *d  = (float *)mem_calloc_1d_float(n*CONST_NDIM, 0.0, "init");
  float *xz = (float *)mem_calloc_1d_float(n*CONST_NDIM, 0.0, "init");
  float *D  = (float *)mem_calloc_1d_float(n*CONST_NDIM*CONST_NDIM, 0.0, "init");
  float *y  = (float *)mem_calloc_1d_float(n*CONST_NDIM, 0.0, "init");

  // Generate k=1 layer coordinates to calculate smooth coefficients
  // Note: k=1 layer coordinates will be regenerated here
  int k=1;
  cal_matrix(x2d,z2d,nx,k,step,a,b,c,d,area);
  modify_matrix(x2d,z2d,nx,k,a,b,c,d,coef_e);
  thomas_block(n,a,b,c,d,xz,D,y);
  assign_coords(xz,x2d,z2d,nx,nz,k);

  for(int k=1; k<nz; k++)
  {
    cal_smooth_coef(coef,x2d,z2d,nx,nz,k,t2b,coef_e);
    cal_matrix(x2d,z2d,nx,k,step,a,b,c,d,area);
    modify_matrix(x2d,z2d,nx,k,a,b,c,d,coef_e);
    thomas_block(n,a,b,c,d,xz,D,y);
    assign_coords(xz,x2d,z2d,nx,nz,k);

    fprintf(stdout,"number of layers is %d\n",k);
    fflush(stdout);
  }

  // linear interp
  if(flag_stretch == 1)
  {
    zt_arc_stretch(x2d, z2d, step, nx, nz);
  }

  if(t2b == 1)
  {
    //fprintf(stdout,"the init bdry is max index, so index must be flip\n");
    flip_coord_z(x2d, z2d, nx, nz);
  }

  free(coef_e);
  free(area);
  free(a);
  free(b);
  free(c);
  free(d);
  free(xz);
  free(D);
  free(y);

  return 0;
}

int
cal_smooth_coef(float coef, float *x2d, float *z2d, int nx, int nz, int k, int t2b, float *coef_e)
{
    float S;
    size_t iptr1, iptr2, iptr3, iptr4;
    float x_xi,z_xi,x_zt,z_zt;
    float xi_len,zt_len,N_xi;
    float x_xi_plus,z_xi_plus,x_xi_minus,z_xi_minus;
    float xi_plus1,xi_minus1,xi_plus2,xi_minus2;
    float d1,d2,delta,delta_mdfy;
    float x_plus,z_plus,x_minus,z_minus;
    float dot,det,theta,alpha;
    float temp;

    S = sqrt((1.0*k)/(nz-1));
    int k1;
    if (k==1)
    {
      k1=2;
    } else {
      k1=k;
    }
  
    for(int i=1; i<nx-1; i++)
    {
      iptr1 = (k1-1)*nx + i+1;   // (i+1,k-1)
      iptr2 = (k1-1)*nx + i-1;   // (i-1,k-1)
      iptr3 = (k1-1)*nx + i;     // (i,k-1)
      iptr4 = (k1-2)*nx + i;     // (i,k-2)
      x_xi = 0.5*(x2d[iptr1] - x2d[iptr2]);
      z_xi = 0.5*(z2d[iptr1] - z2d[iptr2]); 
      x_zt = x2d[iptr3] - x2d[iptr4];
      z_zt = z2d[iptr3] - z2d[iptr4];
      xi_len = sqrt(pow(x_xi,2) + pow(z_xi,2));
      zt_len = sqrt(pow(x_zt,2) + pow(z_zt,2));
      N_xi = zt_len/xi_len;

      iptr1 = (k1-2)*nx + i+1;   // (i+1,k-2)
      iptr2 = (k1-2)*nx + i;     // (i,  k-2)
      iptr3 = (k1-2)*nx + i-1;   // (i-1,k-2)
      x_xi_plus = x2d[iptr1] - x2d[iptr2];
      z_xi_plus = z2d[iptr1] - z2d[iptr2];
      xi_plus1 = sqrt(pow(x_xi_plus,2) + pow(z_xi_plus,2));
      x_xi_minus = x2d[iptr3] - x2d[iptr2];
      z_xi_minus = z2d[iptr3] - z2d[iptr2];
      xi_minus1 = sqrt(pow(x_xi_minus,2) + pow(z_xi_minus,2));

      iptr1 = (k1-1)*nx + i+1;   // (i+1,k-1)
      iptr2 = (k1-1)*nx + i;     // (i,  k-1)
      iptr3 = (k1-1)*nx + i-1;   // (i-1,k-1)
      x_xi_plus = x2d[iptr1] - x2d[iptr2];
      z_xi_plus = z2d[iptr1] - z2d[iptr2];
      xi_plus2 = sqrt(pow(x_xi_plus,2) + pow(z_xi_plus,2));
      x_xi_minus = x2d[iptr3] - x2d[iptr2];
      z_xi_minus = z2d[iptr3] - z2d[iptr2];
      xi_minus2 = sqrt(pow(x_xi_minus,2) + pow(z_xi_minus,2));

      d1 = xi_plus1 + xi_minus1;
      d2 = xi_plus2 + xi_minus2;
      delta = d1/d2;
      delta_mdfy = fmax(pow(delta,2/S),0.01);
      // normalization
      x_plus = (x2d[iptr1]-x2d[iptr2])/xi_plus2;
      z_plus = (z2d[iptr1]-z2d[iptr2])/xi_plus2;
      x_minus = (x2d[iptr3]-x2d[iptr2])/xi_minus2;
      z_minus = (z2d[iptr3]-z2d[iptr2])/xi_minus2;

      dot = x_plus*x_minus + z_plus*z_minus;
      det = x_plus*z_minus - z_plus*x_minus;

      // cal two normal vector clockwise angle.
      // the method from website
      // from plus vector to minus vector
      // z axis upward, so is -det
      theta = atan2(-det,dot);
      if(theta<0)
      {
        theta = theta + 2*PI;
      }

      if(t2b==0)
      {
        theta = 2*PI-theta;
      }
      if(theta<PI)
      {
        alpha = 1.0/(1-pow(cos(theta/2),2));
      }
      if(theta>=PI)
      {
        alpha = 1;
      }
      coef_e[i] = coef*N_xi*S*delta_mdfy*alpha;
    }
    return 0;
}

int 
cal_matrix(float *x2d, float *z2d, int nx, int k, float *step,
           float *a, float *b, float *c, float *d, float *area)
{
    float A[2][2], B[2][2];
    float mat[2][2], vec[2];
    float mat_b[2][2], vec_d[2];
    size_t iptr1,iptr2,iptr3;
    float x_xi0,z_xi0,x_zt0,z_zt0;
    float x_plus,z_plus,x_minus,z_minus;
    float arc_plus,arc_minus,arc_len,temp;

    for(int i=1; i<nx-1; i++)
    {
      iptr1 = (k-1)*nx + i+1;
      iptr2 = (k-1)*nx + i;
      iptr3 = (k-1)*nx + i-1;
      x_xi0 = 0.5*(x2d[iptr1] - x2d[iptr3]);
      z_xi0 = 0.5*(z2d[iptr1] - z2d[iptr3]);
      x_plus = x2d[iptr1] - x2d[iptr2];
      z_plus = z2d[iptr1] - z2d[iptr2];
      arc_plus = sqrt(pow(x_plus,2) + pow(z_plus,2));
      x_minus = x2d[iptr3] - x2d[iptr2];
      z_minus = z2d[iptr3] - z2d[iptr2];
      arc_minus = sqrt(pow(x_minus,2) + pow(z_minus,2));
      arc_len = 0.5*(arc_plus + arc_minus);
      // arc_length -> area
      // area(i) = A0 k-1 layer area 
      // area(i+nx) = A1 k layer area
      if(k==1)
      {
        area[i] = arc_len * step[k-1];
        area[i+nx] = area[i];
      } else {
        area[i] = area[i+nx];
        area[i+nx] = arc_len * step[k-1];
      }
      temp = pow(x_xi0,2) + pow(z_xi0,2);
      x_zt0 = -z_xi0*area[i]/temp;
      z_zt0 =  x_xi0*area[i]/temp;
      // add damping factor, maybe inv(B) singular
      A[0][0] = x_zt0;      A[0][1] = z_zt0;
      A[1][0] = z_zt0;      A[1][1] =-x_zt0; 
      B[0][0] = x_xi0+1e-7; B[0][1] = z_xi0;
      B[1][0] =-z_xi0;      B[1][1] = x_xi0+1e-7; 
      mat_invert2x2(B);
      mat_mul2x2(B,A,mat);
      mat_iden2x2(mat_b);
      vec[0] = 0; vec[1] = area[i+nx];
      mat_mul2x1(B,vec,vec_d);
      iptr1 = (i-1)*CONST_NDIM*CONST_NDIM;
      iptr2 = (i-1)*CONST_NDIM;
      for(int ii=0; ii<2; ii++) {
        for(int jj=0; jj<2; jj++) {
          a[iptr1+2*ii+jj] = -0.5*mat[ii][jj];
          b[iptr1+2*ii+jj] = mat_b[ii][jj];
          c[iptr1+2*ii+jj] = 0.5*mat[ii][jj];
        }
        d[iptr2+ii] = vec_d[ii];
      }
    }

    return 0;
}

int
modify_matrix(float *x2d, float *z2d, int nx, int k, float *a,
              float *b, float *c, float *d, float *coef_e)
{
  float mat[2][2], vec1[2], vec2[2], vec3[2];
  float coef_i;
  size_t iptr1, iptr2, iptr3, iptr4, iptr5;
  mat_iden2x2(mat);
  for(int i=1; i<nx-1; i++)
  {
    iptr1 = (k-1)*nx + i-1;
    iptr2 = (k-1)*nx + i;
    iptr3 = (k-1)*nx + i+1;
    vec1[0] = x2d[iptr1];
    vec1[1] = z2d[iptr1];
    vec2[0] = x2d[iptr2];
    vec2[1] = z2d[iptr2];
    vec3[0] = x2d[iptr3];
    vec3[1] = z2d[iptr3];

    iptr4 = (i-1)*CONST_NDIM*CONST_NDIM;
    iptr5 = (i-1)*CONST_NDIM;
    
    coef_i = 2*coef_e[i];

    for(int ii=0; ii<2; ii++) {
      for(int jj=0; jj<2; jj++) {
        a[iptr4+2*ii+jj] = a[iptr4+2*ii+jj] - coef_i*mat[ii][jj];
        b[iptr4+2*ii+jj] = b[iptr4+2*ii+jj] + 2*coef_i*mat[ii][jj];
        c[iptr4+2*ii+jj] = c[iptr4+2*ii+jj] - coef_i*mat[ii][jj];
      }
      d[iptr5+ii] = d[iptr5+ii] + coef_e[i]*(vec1[ii]+vec3[ii]-2*vec2[ii]);
    }
  }

  // left bdry
  // float boundry
  for(int ii=0; ii<2; ii++) {
    for(int jj=0; jj<2; jj++) {
      // modify i=0
      b[ii*2+jj] = b[ii*2+jj] + a[ii*2+jj];
    }
  }
  // right bdry
  // float boundry
  size_t iptr = (nx-3)*CONST_NDIM*CONST_NDIM;
  for(int ii=0; ii<2; ii++) {
    for(int jj=0; jj<2; jj++) {
      // modify i=n-1
      b[iptr+ii*2+jj] = b[iptr+ii*2+jj] + c[iptr+ii*2+jj];
    }
  }

  return 0;
}


int
assign_coords(float *xz, float *x2d, float *z2d, int nx, int nz, int k)
{
  size_t iptr,iptr1,iptr2;
  size_t iptr3,iptr4,iptr5;
  for(int i=1; i<nx-1; i++)
  {
    iptr  =  k*nx + i;
    iptr1 = (k-1)*nx + i;
    iptr2 = (i-1)*CONST_NDIM;
    x2d[iptr] = x2d[iptr1] + xz[iptr2];
    z2d[iptr] = z2d[iptr1] + xz[iptr2+1];
  }

  // left
  // floating boundary
  iptr  = k*nx+0;       // (0,k)
  iptr1 = (k-1)*nx+0;   // (0,k-1)
  iptr2 = k*nx+1;       // (1,k)
  iptr3 = (k-1)*nx+1;   // (1,k-1)
  iptr4 = k*nx+2;       // (2,k)
  iptr5 = (k-1)*nx+2;   // (2,k-1)

  x2d[iptr] = x2d[iptr1] + x2d[iptr2]-x2d[iptr3];
  z2d[iptr] = z2d[iptr1] + z2d[iptr2]-z2d[iptr3];

  iptr  = k*nx+(nx-1);       // (nx-1,k)
  iptr1 = (k-1)*nx+(nx-1);   // (nx-1,k-1)
  iptr2 = k*nx+(nx-2);       // (nx-2,k)
  iptr3 = (k-1)*nx+(nx-2);   // (nx-2,k-1)
  iptr4 = k*nx+(nx-3);       // (nx-3,k)
  iptr5 = (k-1)*nx+(nx-3);   // (nx-3,k-1)

  x2d[iptr] = x2d[iptr1] + x2d[iptr2]-x2d[iptr3];
  z2d[iptr] = z2d[iptr1] + z2d[iptr2]-z2d[iptr3];

  return 0;
}


/*
  solve block tridiagonal linear system equaltion
  using thomas method
  Due to the sparse coefficient matrix of the tridiagonal equation,
  the computational complexity is proportional to n,
  rather than the n^3 of Gaussian elimination 

  [b1 c1        ]
  |a2 b2 c2     |
  |   a3 b3 c3  |
  [             ]

  a = square matrix element of lower diagonal
  b = square matrix element of main diagonal
  c = square matrix element of up diagonal
  d = right hand item, each element size is n*1

    [G1 0  0       ]
    |a2 G2 0       |
    |   a3 G3      |
  L=|              |
    |              |
    |              |
    [         an Gn]
                    
    [I1 D1 0            ]
    |0  I2 D2           |
    |   0  I3           |
  U=|                   |
    |                   |
    |          In-1 Dn-1|
    [          0    In  ]

  LU*x = d
  Ly=d
  Ux=y
*/
int
thomas_block(int n, float *a, float *b, float *c, float *d,
             float *xz, float *D, float *y)
{
  float mat_a[2][2], mat_b[2][2], mat_c[2][2], vec_d[2];
  float mat_G[2][2], mat_D[2][2], vec_y[2], vec_xz[2];
  float mat1[2][2], vec1[2], vec2[2];
  size_t iptr1,iptr2,iptr3,iptr4;

  // i=0
  for(int ii=0; ii<2; ii++) {
    for(int jj=0; jj<2; jj++) {
      mat_G[ii][jj] = b[ii*2+jj];
      mat_c[ii][jj] = c[ii*2+jj];
    }
    vec_d[ii] = d[ii];
  }

  mat_invert2x2(mat_G);
  mat_mul2x2(mat_G,mat_c,mat_D);
  mat_mul2x1(mat_G,vec_d,vec_y);
  for(int ii=0; ii<2; ii++) { 
    for(int jj=0; jj<2; jj++) {
      D[ii*2+jj] = mat_D[ii][jj];
    }
    y[ii] = vec_y[ii];
  }

  // i=1~n-1
  for(int i=1; i<n; i++)
  {
    iptr1 = i*CONST_NDIM*CONST_NDIM;
    iptr2 = i*CONST_NDIM;
    iptr3 = (i-1)*CONST_NDIM*CONST_NDIM;
    iptr4 = (i-1)*CONST_NDIM;
    for(int ii=0; ii<2; ii++) { 
      for(int jj=0; jj<2; jj++) {
        mat_a[ii][jj] = a[iptr1+ii*2+jj];
        mat_b[ii][jj] = b[iptr1+ii*2+jj];
        mat_c[ii][jj] = c[iptr1+ii*2+jj];
        mat_D[ii][jj] = D[iptr3+ii*2+jj];
      }
      vec_d[ii] = d[iptr2+ii];
      vec_y[ii] = y[iptr4+ii];
    }
    mat_mul2x2(mat_a,mat_D,mat1);  // a(i)*D(i-1)
    mat_sub2x2(mat_b,mat1,mat_G);  // G(i) = b(i)-a(i)*D(i-1)
    mat_invert2x2(mat_G);          // inv(G(i))
    mat_mul2x2(mat_G,mat_c,mat_D); // D(i) = inv(G(i))*c(i)
    mat_mul2x1(mat_a,vec_y,vec1);  // a(i) * y(i-1)
    vec_sub2x1(vec_d,vec1,vec2);   // d(i) - a(i) * y(i-1)
    mat_mul2x1(mat_G,vec2,vec_y);  // y(i) = inv(G(i))*(d(i) - a(i) * y(i-1))

    for(int ii=0; ii<2; ii++) { 
      for(int jj=0; jj<2; jj++) {
        D[iptr1+ii*2+jj] = mat_D[ii][jj]; 
      }
      y[iptr2+ii] = vec_y[ii];
    }
  }
  
  // i=n-1
  iptr2 = (n-1)*CONST_NDIM;
  for(int ii=0; ii<2; ii++) { 
    xz[iptr2+ii] = y[iptr2+ii];
  }

  for(int i=n-2; i>=0; i--)
  {
    iptr1 = i*CONST_NDIM*CONST_NDIM;
    iptr2 = i*CONST_NDIM;
    iptr4 = (i+1)*CONST_NDIM;
    
    for(int ii=0; ii<2; ii++) { 
      for(int jj=0; jj<2; jj++) {
        mat_D[ii][jj] = D[iptr1+ii*2+jj]; 
      }
      vec_xz[ii] = xz[iptr4+ii];
    }
    mat_mul2x1(mat_D,vec_xz,vec1);  // D(i)*xz(i+1)
    for(int ii=0; ii<2; ii++) 
    { 
      xz[iptr2+ii] = y[iptr2+ii] - vec1[ii]; // xz(i) = y(i) - D(i)*xz(i+1) 
    }
  }
  
  return 0;
}
