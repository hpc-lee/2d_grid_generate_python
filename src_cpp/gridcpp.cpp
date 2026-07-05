// C++ accelerated grid generation kernels (pybind11).
//
// Parabolic:  thomas / predict_point / update_point
// Hyperbolic: cal_smooth_coef / cal_matrix / modify_matrix / thomas_block
// Elliptic:   update_SOR / interp_inner_source / compute_residual

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;

// =========================================================================
// parabolic operators (ported from numba grid_generation.py)
// =========================================================================

// Thomas tridiagonal solver, operates on columns of var_th (stride=7).
// a[i]=vth[7*i+0], b[i]=vth[7*i+1], c[i]=vth[7*i+2],
// d_x[i]=vth[7*i+3], d_z[i]=vth[7*i+4], u_x[i]=vth[7*i+5], u_z[i]=vth[7*i+6].
static inline void thomas_strided(float *vth, int n)
{
    for (int i = 1; i < n; ++i)
    {
        float factor = vth[7 * i + 0] / vth[7 * (i - 1) + 1];   // a[i]/b[i-1]
        vth[7 * i + 1] -= factor * vth[7 * (i - 1) + 2];          // b[i] -= factor*c[i-1]
        vth[7 * i + 3] -= factor * vth[7 * (i - 1) + 3];          // d_x[i]
        vth[7 * i + 4] -= factor * vth[7 * (i - 1) + 4];          // d_z[i]
    }
    vth[7 * (n - 1) + 5] = vth[7 * (n - 1) + 3] / vth[7 * (n - 1) + 1];  // u_x[n-1]
    vth[7 * (n - 1) + 6] = vth[7 * (n - 1) + 4] / vth[7 * (n - 1) + 1];  // u_z[n-1]
    for (int i = n - 2; i >= 0; --i)
    {
        vth[7 * i + 5] = (vth[7 * i + 3] - vth[7 * i + 2] * vth[7 * (i + 1) + 5]) / vth[7 * i + 1];
        vth[7 * i + 6] = (vth[7 * i + 4] - vth[7 * i + 2] * vth[7 * (i + 1) + 6]) / vth[7 * i + 1];
    }
}

// predict_point: predict k+1/k layer points from k-1 layer, writes x_pre/z_pre (in-place).
// Pointwise logic matches numba grid_generation.py predict_point.
void predict_point_cpp(FArr x2d, FArr z2d, int nx, int nz, int k, int t2b,
                      float coef, FArr step_len, FArr x_pre, FArr z_pre)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    const float *slen = step_len.data();
    float *xpre = x_pre.mutable_data();  // (nx, 2)
    float *zpre = z_pre.mutable_data();

    float zt = (1.0f * k) / (nz - 1);
    float cs = std::exp(-coef * zt);
    int sign1 = (t2b == 1) ? 1 : -1;

    for (int i = 1; i < nx - 1; ++i)
    {
        float vt_x = 0.5f * (x[(k - 1) * nx + i + 1] - x[(k - 1) * nx + i - 1]);
        float vt_z = 0.5f * (z[(k - 1) * nx + i + 1] - z[(k - 1) * nx + i - 1]);
        float len_vt = std::sqrt(vt_x * vt_x + vt_z * vt_z);
        if (len_vt == 0.0f)
        {
            throw std::runtime_error(
                "Tangential vector length is zero; cannot normalize.");
        }
        float vn_x = sign1 * vt_z / len_vt;
        float vn_z = -sign1 * vt_x / len_vt;

        float R_x = x[(k - 1) * nx + i] - x[(nz - 1) * nx + i];
        float R_z = z[(k - 1) * nx + i] - z[(nz - 1) * nx + i];
        float R = std::sqrt(R_x * R_x + R_z * R_z);

        float R1 = slen[nz - 1] - slen[k - 1];
        float r1 = slen[k + 1] - slen[k - 1];
        float r2 = slen[k] - slen[k - 1];
        float c1 = (R1 != 0.0f) ? r1 / R1 : 0.0f;
        float c2 = (r1 != 0.0f) ? r2 / r1 : 0.0f;

        float x0 = x[(k - 1) * nx + i] + vn_x * c1 * R;
        float z0 = z[(k - 1) * nx + i] + vn_z * c1 * R;
        float xs = x[(k - 1) * nx + i] + c1 * (x[(nz - 1) * nx + i] - x[(k - 1) * nx + i]);
        float zs = z[(k - 1) * nx + i] + c1 * (z[(nz - 1) * nx + i] - z[(k - 1) * nx + i]);

        xpre[2 * i + 1] = cs * x0 + (1.0f - cs) * xs;   // x_pre[i, 1] = xk1
        zpre[2 * i + 1] = cs * z0 + (1.0f - cs) * zs;
        xpre[2 * i + 0] = x[(k - 1) * nx + i] + c2 * (xpre[2 * i + 1] - x[(k - 1) * nx + i]);  // x_pre[i,0] = xk0
        zpre[2 * i + 0] = z[(k - 1) * nx + i] + c2 * (zpre[2 * i + 1] - z[(k - 1) * nx + i]);
    }

    // geometric symmetry bdry (mirror endpoints i=0 and i=nx-1, for k0 and k1)
    xpre[2 * 0 + 0] = 2.0f * xpre[2 * 1 + 0] - xpre[2 * 2 + 0];
    zpre[2 * 0 + 0] = 2.0f * zpre[2 * 1 + 0] - zpre[2 * 2 + 0];
    xpre[2 * 0 + 1] = 2.0f * xpre[2 * 1 + 1] - xpre[2 * 2 + 1];
    zpre[2 * 0 + 1] = 2.0f * zpre[2 * 1 + 1] - zpre[2 * 2 + 1];
    xpre[2 * (nx - 1) + 0] = 2.0f * xpre[2 * (nx - 2) + 0] - xpre[2 * (nx - 3) + 0];
    zpre[2 * (nx - 1) + 0] = 2.0f * zpre[2 * (nx - 2) + 0] - zpre[2 * (nx - 3) + 0];
    xpre[2 * (nx - 1) + 1] = 2.0f * xpre[2 * (nx - 2) + 1] - xpre[2 * (nx - 3) + 1];
    zpre[2 * (nx - 1) + 1] = 2.0f * zpre[2 * (nx - 2) + 1] - zpre[2 * (nx - 3) + 1];
}

// update_point: update layer k coords via Thomas from x_pre/z_pre (in-place writes x2d/z2d).
// Pointwise logic matches numba grid_generation.py update_point.
void update_point_cpp(FArr x2d, FArr z2d, FArr var_th, int nx, int k,
                      FArr x_pre, FArr z_pre)
{
    float *x = x2d.mutable_data();
    float *z = z2d.mutable_data();
    float *vth = var_th.mutable_data();       // (nx-2, 7)
    const float *xpre = x_pre.data();          // (nx, 2)
    const float *zpre = z_pre.data();
    int n = nx - 2;

    for (int i = 1; i < nx - 1; ++i)
    {
        int idx = i - 1;  // var_th row index (0-based)
        float x_xi = 0.5f * (xpre[2 * (i + 1) + 0] - xpre[2 * (i - 1) + 0]);  // xk0[i+1]-xk0[i-1]
        float z_xi = 0.5f * (zpre[2 * (i + 1) + 0] - zpre[2 * (i - 1) + 0]);
        float x_zt = 0.5f * (xpre[2 * i + 1] - x[(k - 1) * nx + i]);           // xk1[i]-x[k-1,i]
        float z_zt = 0.5f * (zpre[2 * i + 1] - z[(k - 1) * nx + i]);
        float temp_x = xpre[2 * i + 1] + x[(k - 1) * nx + i];                  // xk1[i]+x[k-1,i]
        float temp_z = zpre[2 * i + 1] + z[(k - 1) * nx + i];

        float x_xizt = 0.25f * (xpre[2 * (i + 1) + 1] + x[(k - 1) * nx + i - 1]
                                - x[(k - 1) * nx + i + 1] - xpre[2 * (i - 1) + 1]);
        float z_xizt = 0.25f * (zpre[2 * (i + 1) + 1] + z[(k - 1) * nx + i - 1]
                                - z[(k - 1) * nx + i + 1] - zpre[2 * (i - 1) + 1]);

        float g11 = x_xi * x_xi + z_xi * z_xi;
        float g22 = x_zt * x_zt + z_zt * z_zt;
        float g12 = x_xi * x_zt + z_xi * z_zt;

        vth[7 * idx + 0] = g22;                       // a
        vth[7 * idx + 1] = -2.0f * (g22 + g11);        // b
        vth[7 * idx + 2] = g22;                        // c
        vth[7 * idx + 3] = -g11 * temp_x + 2.0f * g12 * x_xizt;  // d_x
        vth[7 * idx + 4] = -g11 * temp_z + 2.0f * g12 * z_xizt;  // d_z
    }

    // boundary modify: d_x[0]-=a[0]*xk0[0]; d_x[n-1]-=c[n-1]*xk0[nx-1]
    vth[7 * 0 + 3] -= vth[7 * 0 + 0] * xpre[2 * 0 + 0];
    vth[7 * 0 + 4] -= vth[7 * 0 + 0] * zpre[2 * 0 + 0];
    vth[7 * (n - 1) + 3] -= vth[7 * (n - 1) + 2] * xpre[2 * (nx - 1) + 0];
    vth[7 * (n - 1) + 4] -= vth[7 * (n - 1) + 2] * zpre[2 * (nx - 1) + 0];

    thomas_strided(vth, n);

    for (int i = 1; i < nx - 1; ++i)
    {
        int idx = i - 1;
        x[k * nx + i] = vth[7 * idx + 5];  // u_x
        z[k * nx + i] = vth[7 * idx + 6];  // u_z
    }
}

// =========================================================================
// hyperbolic operators (ported from src_c/hyperbolic.c, match C-version golden)
// =========================================================================

static const float HPI = 3.14159265358979323846f;

// 2x2 inversion (in-place), matches lib_math.c mat_invert2x2: skip inversion when |det|<1e-8.
static inline void inv2x2(float *m)  // m[0]=00 m[1]=01 m[2]=10 m[3]=11
{
    float det = m[0] * m[3] - m[1] * m[2];
    if (std::fabs(det) < 1e-8f)
        return;  // singular: keep original (consistent with C; damping avoids triggering in practice)
    float inv_det = 1.0f / det;
    float a = m[0], b = m[1], c = m[2], d = m[3];
    m[0] = d * inv_det;
    m[1] = -b * inv_det;
    m[2] = -c * inv_det;
    m[3] = a * inv_det;
}

// cal_smooth_coef: compute smoothing coefficient coef_e (in-place). Ported from hyperbolic.c cal_smooth_coef.
void cal_smooth_coef_cpp(float coef, FArr x2d, FArr z2d, int nx, int nz,
                         int k, int t2b, FArr coef_e)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    float *ce = coef_e.mutable_data();

    float S = std::sqrt((1.0f * k) / (nz - 1));
    int k1 = (k == 1) ? 2 : k;

    for (int i = 1; i < nx - 1; ++i)
    {
        float x_xi = 0.5f * (x[(k1 - 1) * nx + i + 1] - x[(k1 - 1) * nx + i - 1]);
        float z_xi = 0.5f * (z[(k1 - 1) * nx + i + 1] - z[(k1 - 1) * nx + i - 1]);
        float x_zt = x[(k1 - 1) * nx + i] - x[(k1 - 2) * nx + i];
        float z_zt = z[(k1 - 1) * nx + i] - z[(k1 - 2) * nx + i];
        float xi_len = std::sqrt(x_xi * x_xi + z_xi * z_xi);
        float zt_len = std::sqrt(x_zt * x_zt + z_zt * z_zt);
        float N_xi = zt_len / xi_len;

        float x_xi_plus = x[(k1 - 2) * nx + i + 1] - x[(k1 - 2) * nx + i];
        float z_xi_plus = z[(k1 - 2) * nx + i + 1] - z[(k1 - 2) * nx + i];
        float xi_plus1 = std::sqrt(x_xi_plus * x_xi_plus + z_xi_plus * z_xi_plus);
        float x_xi_minus = x[(k1 - 2) * nx + i - 1] - x[(k1 - 2) * nx + i];
        float z_xi_minus = z[(k1 - 2) * nx + i - 1] - z[(k1 - 2) * nx + i];
        float xi_minus1 = std::sqrt(x_xi_minus * x_xi_minus + z_xi_minus * z_xi_minus);

        float xp2 = x[(k1 - 1) * nx + i + 1] - x[(k1 - 1) * nx + i];
        float zp2 = z[(k1 - 1) * nx + i + 1] - z[(k1 - 1) * nx + i];
        float xi_plus2 = std::sqrt(xp2 * xp2 + zp2 * zp2);
        float xm2 = x[(k1 - 1) * nx + i - 1] - x[(k1 - 1) * nx + i];
        float zm2 = z[(k1 - 1) * nx + i - 1] - z[(k1 - 1) * nx + i];
        float xi_minus2 = std::sqrt(xm2 * xm2 + zm2 * zm2);

        float d1 = xi_plus1 + xi_minus1;
        float d2 = xi_plus2 + xi_minus2;
        float delta = d1 / d2;
        float delta_mdfy = std::fmax(std::pow(delta, 2.0f / S), 0.01f);

        float x_plus = xp2 / xi_plus2;
        float z_plus = zp2 / xi_plus2;
        float x_minus = xm2 / xi_minus2;
        float z_minus = zm2 / xi_minus2;

        float dot = x_plus * x_minus + z_plus * z_minus;
        float det = x_plus * z_minus - z_plus * x_minus;
        float theta = std::atan2(-det, dot);
        if (theta < 0)
            theta += 2.0f * HPI;
        if (t2b == 0)
            theta = 2.0f * HPI - theta;
        float alpha = (theta < HPI) ? 1.0f / (1.0f - std::pow(std::cos(theta / 2.0f), 2.0f)) : 1.0f;
        ce[i] = coef * N_xi * S * delta_mdfy * alpha;
    }
}

// cal_matrix: assemble block-tridiagonal matrix a/b/c/d and area (in-place). Ported from hyperbolic.c cal_matrix.
void cal_matrix_cpp(FArr x2d, FArr z2d, int nx, int k, FArr step,
                    FArr a, FArr b, FArr c, FArr d, FArr area)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    const float *sp = step.data();
    float *pa = a.mutable_data();
    float *pb = b.mutable_data();
    float *pc = c.mutable_data();
    float *pd = d.mutable_data();
    float *ar = area.mutable_data();  // (nx, 2)

    for (int i = 1; i < nx - 1; ++i)
    {
        float x_xi0 = 0.5f * (x[(k - 1) * nx + i + 1] - x[(k - 1) * nx + i - 1]);
        float z_xi0 = 0.5f * (z[(k - 1) * nx + i + 1] - z[(k - 1) * nx + i - 1]);
        float diff_plus_x = x[(k - 1) * nx + i + 1] - x[(k - 1) * nx + i];
        float diff_plus_z = z[(k - 1) * nx + i + 1] - z[(k - 1) * nx + i];
        float diff_minus_x = x[(k - 1) * nx + i - 1] - x[(k - 1) * nx + i];
        float diff_minus_z = z[(k - 1) * nx + i - 1] - z[(k - 1) * nx + i];
        float arc_plus = std::sqrt(diff_plus_x * diff_plus_x + diff_plus_z * diff_plus_z);
        float arc_minus = std::sqrt(diff_minus_x * diff_minus_x + diff_minus_z * diff_minus_z);
        float arc_len = 0.5f * (arc_plus + arc_minus);

        if (k == 1)
        {
            ar[2 * i + 0] = arc_len * sp[k - 1];
            ar[2 * i + 1] = ar[2 * i + 0];
        }
        else
        {
            ar[2 * i + 0] = ar[2 * i + 1];
            ar[2 * i + 1] = arc_len * sp[k - 1];
        }

        float temp = x_xi0 * x_xi0 + z_xi0 * z_xi0;
        float area0 = ar[2 * i + 0];
        float x_zt0 = -z_xi0 * area0 / temp;
        float z_zt0 = x_xi0 * area0 / temp;

        // A = [[x_zt0, z_zt0],[z_zt0, -x_zt0]]
        float A[4] = {x_zt0, z_zt0, z_zt0, -x_zt0};
        // B = [[x_xi0+1e-7, z_xi0],[-z_xi0, x_xi0+1e-7]]
        float B[4] = {x_xi0 + 1e-7f, z_xi0, -z_xi0, x_xi0 + 1e-7f};
        inv2x2(B);  // B <- inv(B)
        // mat = B * A
        float mat[4];
        mat[0] = B[0] * A[0] + B[1] * A[2];
        mat[1] = B[0] * A[1] + B[1] * A[3];
        mat[2] = B[2] * A[0] + B[3] * A[2];
        mat[3] = B[2] * A[1] + B[3] * A[3];
        // vec = [0, area1]; vec_d = B * vec
        float area1 = ar[2 * i + 1];
        float vec_d0 = B[1] * area1;
        float vec_d1 = B[3] * area1;

        float *ai = pa + (i - 1) * 4;
        float *bi = pb + (i - 1) * 4;
        float *ci = pc + (i - 1) * 4;
        // a = -0.5*mat, b = I, c = 0.5*mat
        ai[0] = -0.5f * mat[0]; ai[1] = -0.5f * mat[1];
        ai[2] = -0.5f * mat[2]; ai[3] = -0.5f * mat[3];
        bi[0] = 1.0f; bi[1] = 0.0f; bi[2] = 0.0f; bi[3] = 1.0f;
        ci[0] = 0.5f * mat[0]; ci[1] = 0.5f * mat[1];
        ci[2] = 0.5f * mat[2]; ci[3] = 0.5f * mat[3];
        pd[2 * (i - 1) + 0] = vec_d0;
        pd[2 * (i - 1) + 1] = vec_d1;
    }
}

// modify_matrix: add dissipation + boundary modification to a/b/c/d (in-place). Ported from hyperbolic.c.
void modify_matrix_cpp(FArr x2d, FArr z2d, int nx, int k,
                       FArr a, FArr b, FArr c, FArr d, FArr coef_e)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    const float *ce = coef_e.data();
    float *pa = a.mutable_data();
    float *pb = b.mutable_data();
    float *pc = c.mutable_data();
    float *pd = d.mutable_data();
    int n = nx - 2;

    for (int i = 1; i < nx - 1; ++i)
    {
        int idx = i - 1;
        float coef_i = 2.0f * ce[i];
        float dx = x[(k - 1) * nx + i - 1] + x[(k - 1) * nx + i + 1] - 2.0f * x[(k - 1) * nx + i];
        float dz = z[(k - 1) * nx + i - 1] + z[(k - 1) * nx + i + 1] - 2.0f * z[(k - 1) * nx + i];

        float *ai = pa + idx * 4;
        float *bi = pb + idx * 4;
        float *ci = pc + idx * 4;
        // a -= coef_i*I ; b += 2*coef_i*I ; c -= coef_i*I
        ai[0] -= coef_i; ai[3] -= coef_i;
        bi[0] += 2.0f * coef_i; bi[3] += 2.0f * coef_i;
        ci[0] -= coef_i; ci[3] -= coef_i;
        // d += coef_e[i] * (prev + next - 2*curr)
        pd[2 * idx + 0] += ce[i] * dx;
        pd[2 * idx + 1] += ce[i] * dz;
    }

    // floating boundary: b[0] += a[0]; b[n-1] += c[n-1]
    float *b0 = pb;
    float *a0 = pa;
    b0[0] += a0[0]; b0[1] += a0[1]; b0[2] += a0[2]; b0[3] += a0[3];
    float *bn = pb + (n - 1) * 4;
    float *cn = pc + (n - 1) * 4;
    bn[0] += cn[0]; bn[1] += cn[1]; bn[2] += cn[2]; bn[3] += cn[3];
}

// thomas_block: block-tridiagonal Thomas solver (in-place writes xz). Ported from hyperbolic.c thomas_block.
void thomas_block_cpp(int n, FArr a, FArr b, FArr c, FArr d,
                      FArr xz, FArr D, FArr y)
{
    float *pa = a.mutable_data();
    float *pb = b.mutable_data();
    float *pc = c.mutable_data();
    float *pd = d.mutable_data();
    float *pxz = xz.mutable_data();
    float *pD = D.mutable_data();
    float *py = y.mutable_data();

    // i=0: G=inv(b[0]); D[0]=G*c[0]; y[0]=G*d[0]
    float G[4] = {pb[0], pb[1], pb[2], pb[3]};
    inv2x2(G);
    pD[0] = G[0] * pc[0] + G[1] * pc[2];
    pD[1] = G[0] * pc[1] + G[1] * pc[3];
    pD[2] = G[2] * pc[0] + G[3] * pc[2];
    pD[3] = G[2] * pc[1] + G[3] * pc[3];
    py[0] = G[0] * pd[0] + G[1] * pd[1];
    py[1] = G[2] * pd[0] + G[3] * pd[1];

    for (int i = 1; i < n; ++i)
    {
        float *ai = pa + i * 4;
        float *bi = pb + i * 4;
        float *ci = pc + i * 4;
        float *Di = pD + i * 4;
        float *Dim1 = pD + (i - 1) * 4;
        // aD = a[i]*D[i-1]
        float aD[4];
        aD[0] = ai[0] * Dim1[0] + ai[1] * Dim1[2];
        aD[1] = ai[0] * Dim1[1] + ai[1] * Dim1[3];
        aD[2] = ai[2] * Dim1[0] + ai[3] * Dim1[2];
        aD[3] = ai[2] * Dim1[1] + ai[3] * Dim1[3];
        // G = b[i] - aD
        float Gc[4] = {bi[0] - aD[0], bi[1] - aD[1], bi[2] - aD[2], bi[3] - aD[3]};
        inv2x2(Gc);
        // D[i] = G * c[i]
        Di[0] = Gc[0] * ci[0] + Gc[1] * ci[2];
        Di[1] = Gc[0] * ci[1] + Gc[1] * ci[3];
        Di[2] = Gc[2] * ci[0] + Gc[3] * ci[2];
        Di[3] = Gc[2] * ci[1] + Gc[3] * ci[3];
        // ay = a[i]*y[i-1]; y[i] = G*(d[i]-ay)
        float ay0 = ai[0] * py[2 * (i - 1) + 0] + ai[1] * py[2 * (i - 1) + 1];
        float ay1 = ai[2] * py[2 * (i - 1) + 0] + ai[3] * py[2 * (i - 1) + 1];
        float r0 = pd[2 * i + 0] - ay0;
        float r1 = pd[2 * i + 1] - ay1;
        py[2 * i + 0] = Gc[0] * r0 + Gc[1] * r1;
        py[2 * i + 1] = Gc[2] * r0 + Gc[3] * r1;
    }

    // back substitution: xz[n-1]=y[n-1]; xz[i]=y[i]-D[i]*xz[i+1]
    pxz[2 * (n - 1) + 0] = py[2 * (n - 1) + 0];
    pxz[2 * (n - 1) + 1] = py[2 * (n - 1) + 1];
    for (int i = n - 2; i >= 0; --i)
    {
        float *Di = pD + i * 4;
        float xz_next0 = pxz[2 * (i + 1) + 0];
        float xz_next1 = pxz[2 * (i + 1) + 1];
        float Dx0 = Di[0] * xz_next0 + Di[1] * xz_next1;
        float Dx1 = Di[2] * xz_next0 + Di[3] * xz_next1;
        pxz[2 * i + 0] = py[2 * i + 0] - Dx0;
        pxz[2 * i + 1] = py[2 * i + 1] - Dx1;
    }
}

// =========================================================================
// elliptic operators
//   update_SOR: ported from numba grid_utils.py (pointwise Gauss-Seidel, no C version)
//   interp_inner_source / compute_residual: ported from src_c/elliptic.c (match C-version golden)
// Scope: local grid (nz, nx) with ghost, pointwise k=1..nz-2, i=1..nx-2.
// =========================================================================

// update_SOR: Gauss-Seidel pointwise update of x2d_tmp/z2d_tmp (in-place).
// Ported from grid_utils.py update_SOR (numba version). Note x2d_tmp[k,i-1]/[k-1,i]
// uses already-updated values of the current sweep (Gauss-Seidel sequential
// dependency), so it must run pointwise serial and cannot be vectorized.
void update_SOR_cpp(FArr x2d, FArr z2d, FArr x2d_tmp, FArr z2d_tmp,
                    int nx, int nz, FArr P, FArr Q, float omega)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    float *xt = x2d_tmp.mutable_data();
    float *zt = z2d_tmp.mutable_data();
    const float *p = P.data();
    const float *q = Q.data();

    for (int k = 1; k < nz - 1; ++k)
    {
        for (int i = 1; i < nx - 1; ++i)
        {
            size_t c = (size_t)k * nx + i;
            float x_xi = 0.5f * (x[c + 1] - xt[c - 1]);
            float z_xi = 0.5f * (z[c + 1] - zt[c - 1]);
            float x_zt = 0.5f * (x[c + nx] - xt[c - nx]);
            float z_zt = 0.5f * (z[c + nx] - zt[c - nx]);
            float x_xizt = 0.25f * (x[c + nx + 1] + xt[c - nx - 1] - xt[c - nx + 1] - x[c + nx - 1]);
            float z_xizt = 0.25f * (z[c + nx + 1] + zt[c - nx - 1] - zt[c - nx + 1] - z[c + nx - 1]);

            float g11 = x_xi * x_xi + z_xi * z_xi;
            float g22 = x_zt * x_zt + z_zt * z_zt;
            float g12 = x_xi * x_zt + z_xi * z_zt;
            float denom = g22 + g11;
            float coef = (denom != 0.0f) ? 0.5f / denom : 0.0f;

            float xnew = coef * (g22 * (x[c + 1] + xt[c - 1]) +
                                 g11 * (x[c + nx] + xt[c - nx]) -
                                 2.0f * g12 * x_xizt +
                                 g22 * p[c] * x_xi + g11 * q[c] * x_zt);
            xt[c] = omega * xnew + (1.0f - omega) * x[c];

            float znew = coef * (g22 * (z[c + 1] + zt[c - 1]) +
                                 g11 * (z[c + nx] + zt[c - nx]) -
                                 2.0f * g12 * z_xizt +
                                 g22 * p[c] * z_xi + g11 * q[c] * z_zt);
            zt[c] = omega * znew + (1.0f - omega) * z[c];
        }
    }
}

// interp_inner_source: interpolate inner source terms (in-place writes P/Q). Ported from elliptic.c.
void interp_inner_source_cpp(FArr P, FArr P_x1, FArr P_x2, FArr P_z1, FArr P_z2,
                             FArr Q, FArr Q_x1, FArr Q_x2, FArr Q_z1, FArr Q_z2,
                             int nx, int nz, int gni1, int gnk1,
                             int total_nx, int total_nz, FArr coef, FArr weight)
{
    float *p = P.mutable_data();
    float *q = Q.mutable_data();
    const float *px1 = P_x1.data(), *px2 = P_x2.data();
    const float *pz1 = P_z1.data(), *pz2 = P_z2.data();
    const float *qx1 = Q_x1.data(), *qx2 = Q_x2.data();
    const float *qz1 = Q_z1.data(), *qz2 = Q_z2.data();
    const float *cf = coef.data();
    const float *wt = weight.data();

    for (int k = 1; k < nz - 1; ++k)
    {
        for (int i = 1; i < nx - 1; ++i)
        {
            int gnk = gnk1 + k - 1;
            int gni = gni1 + i;
            float xi = (1.0f * gni) / (total_nx - 1);
            float c0 = 1.0f - xi, c1 = xi;
            float r0 = std::exp(-cf[0] * xi);
            float r1 = std::exp(-cf[1] * (1.0f - xi));
            size_t iptr = (size_t)k * nx + i;
            p[iptr] = wt[0] * (c0 * px1[gnk] + c1 * px2[gnk]);
            q[iptr] = wt[0] * (r0 * qx1[gnk] + r1 * qx2[gnk]);
        }
    }
    for (int k = 1; k < nz - 1; ++k)
    {
        for (int i = 1; i < nx - 1; ++i)
        {
            int gnk = gnk1 + k;
            int gni = gni1 + i - 1;
            float zt = (1.0f * gnk) / (total_nz - 1);
            float c0 = 1.0f - zt, c1 = zt;
            float r0 = std::exp(-cf[2] * zt);
            float r1 = std::exp(-cf[3] * (1.0f - zt));
            size_t iptr = (size_t)k * nx + i;
            p[iptr] += wt[1] * (r0 * pz1[gni] + r1 * pz2[gni]);
            q[iptr] += wt[1] * (c0 * qz1[gni] + c1 * qz2[gni]);
        }
    }
}

// compute_residual: residual local_max[2] (in-place). Ported from elliptic.c.
void compute_residual_cpp(FArr x2d, FArr z2d, FArr x2d_tmp, FArr z2d_tmp,
                          FArr local_max, int nx, int nz)
{
    const float *x = x2d.data();
    const float *z = z2d.data();
    const float *xt = x2d_tmp.data();
    const float *zt = z2d_tmp.data();
    float *lm = local_max.mutable_data();

    float max_resi = 0.0f, max_resk = 0.0f;
    for (int k = 1; k < nz - 1; ++k)
    {
        for (int i = 1; i < nx - 1; ++i)
        {
            size_t c = (size_t)k * nx + i;
            size_t c1 = (size_t)k * nx + i + 1;
            size_t c2 = (size_t)(k + 1) * nx + i;
            float dx = xt[c] - x[c], dz = zt[c] - z[c];
            float dif1 = std::sqrt(dx * dx + dz * dz);
            dx = x[c1] - x[c]; dz = z[c1] - z[c];
            float dif2 = std::sqrt(dx * dx + dz * dz);
            dx = x[c2] - x[c]; dz = z[c2] - z[c];
            float dif3 = std::sqrt(dx * dx + dz * dz);
            float resi = dif1 / dif2;
            float resk = dif1 / dif3;
            max_resi = std::fmax(max_resi, resi);
            max_resk = std::fmax(max_resk, resk);
        }
    }
    lm[0] = max_resi;
    lm[1] = max_resk;
}

PYBIND11_MODULE(gridcpp, m)
{
    m.doc() = "C++ accelerated grid generation kernels (parabolic + hyperbolic)";
    m.def("predict_point_cpp", &predict_point_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("nx"), py::arg("nz"),
          py::arg("k"), py::arg("t2b"), py::arg("coef"), py::arg("step_len"),
          py::arg("x_pre"), py::arg("z_pre"),
          "Predict k+1/k layer points from k-1 layer (in-place writes x_pre/z_pre)");
    m.def("update_point_cpp", &update_point_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("var_th"), py::arg("nx"),
          py::arg("k"), py::arg("x_pre"), py::arg("z_pre"),
          "Update layer k coords via Thomas (in-place writes x2d/z2d)");
    m.def("cal_smooth_coef_cpp", &cal_smooth_coef_cpp,
          py::arg("coef"), py::arg("x2d"), py::arg("z2d"), py::arg("nx"),
          py::arg("nz"), py::arg("k"), py::arg("t2b"), py::arg("coef_e"),
          "Hyperbolic smoothing coefficients (in-place writes coef_e)");
    m.def("cal_matrix_cpp", &cal_matrix_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("nx"), py::arg("k"),
          py::arg("step"), py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("d"), py::arg("area"),
          "Hyperbolic matrix assembly (in-place writes a/b/c/d/area)");
    m.def("modify_matrix_cpp", &modify_matrix_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("nx"), py::arg("k"),
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
          py::arg("coef_e"),
          "Hyperbolic matrix modification: dissipation + boundary (in-place)");
    m.def("thomas_block_cpp", &thomas_block_cpp,
          py::arg("n"), py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("d"), py::arg("xz"), py::arg("D"), py::arg("y"),
          "Hyperbolic block tridiagonal Thomas solver (in-place writes xz)");
    m.def("update_SOR_cpp", &update_SOR_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("x2d_tmp"), py::arg("z2d_tmp"),
          py::arg("nx"), py::arg("nz"), py::arg("P"), py::arg("Q"), py::arg("omega"),
          "Elliptic Gauss-Seidel SOR update (in-place writes x2d_tmp/z2d_tmp)");
    m.def("interp_inner_source_cpp", &interp_inner_source_cpp,
          py::arg("P"), py::arg("P_x1"), py::arg("P_x2"), py::arg("P_z1"),
          py::arg("P_z2"), py::arg("Q"), py::arg("Q_x1"), py::arg("Q_x2"),
          py::arg("Q_z1"), py::arg("Q_z2"), py::arg("nx"), py::arg("nz"),
          py::arg("gni1"), py::arg("gnk1"), py::arg("total_nx"), py::arg("total_nz"),
          py::arg("coef"), py::arg("weight"),
          "Elliptic inner source interpolation (in-place writes P/Q)");
    m.def("compute_residual_cpp", &compute_residual_cpp,
          py::arg("x2d"), py::arg("z2d"), py::arg("x2d_tmp"), py::arg("z2d_tmp"),
          py::arg("local_max"), py::arg("nx"), py::arg("nz"),
          "Elliptic residual (in-place writes local_max[2])");
}
