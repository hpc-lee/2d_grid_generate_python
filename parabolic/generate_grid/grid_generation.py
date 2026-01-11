import numpy as np
import numba
from grid_data import GridData
from common.grid_math import flip_coord_z, flip_step_z


def para_gene(gdcurv: GridData, cfgs: dict):
    nx = gdcurv.nx
    nz = gdcurv.nz
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    step = gdcurv.step
    
    t2b = cfgs['t2b']
    coef = cfgs['coef']
    if t2b == 1:
        flip_coord_z(gdcurv.x2d, gdcurv.z2d)
        flip_step_z(gdcurv.step)
    
    # Allocate space for Thomas method
    var_th = np.zeros(((nx - 2), 7), dtype=np.float32)
    x_pre = np.zeros((nx, 2), dtype=np.float32)
    z_pre = np.zeros((nx, 2), dtype=np.float32)
    
    # Calculate step length
    step_len = np.zeros(nz, dtype=np.float32)
    for k in range(1, nz):
        step_len[k] = step_len[k-1] + step[k-1]
    
    for k in range(1, nz - 1):
        # Predict k+1 layer points
        predict_point(x2d, z2d, nx, nz, k, t2b, coef, step_len, x_pre, z_pre)
        # Based on predicted points, update k layer points
        update_point(x2d, z2d, var_th, nx, k, x_pre, z_pre)
        assign_bdry_coords(x2d, z2d, nx, k)
        print(f"number of layer is {k+1}")

    if t2b == 1:
        flip_coord_z(gdcurv.x2d, gdcurv.z2d)


@numba.jit(nopython=True, nogil=True, cache=True)
def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           d_x: np.ndarray, d_z: np.ndarray, 
           u_x: np.ndarray, u_z: np.ndarray):
    """
    Vectorized Thomas algorithm solver
    """
    n = len(b)
    
    # Forward elimination
    for i in range(1, n):
        factor = a[i] / b[i-1]
        b[i] = b[i] - factor * c[i-1]
        d_x[i] = d_x[i] - factor * d_x[i-1]
        d_z[i] = d_z[i] - factor * d_z[i-1]
    
    # Back substitution
    u_x[n-1] = d_x[n-1] / b[n-1]
    u_z[n-1] = d_z[n-1] / b[n-1]
    
    for i in range(n-2, -1, -1):
        u_x[i] = (d_x[i] - c[i] * u_x[i+1]) / b[i]
        u_z[i] = (d_z[i] - c[i] * u_z[i+1]) / b[i]


@numba.jit(nopython=True, nogil=True, cache=True)
def predict_point(x2d: np.ndarray, z2d: np.ndarray, nx: int,
                  nz: int, k: int, t2b: int, coef: float,
                  step_len: np.ndarray, x_pre: np.ndarray,
                  z_pre: np.ndarray) -> None:
    """
        k-1 layer points are known
        predict points k+1 and k layer
        NOTE: this predict point only used
        by calculate matrix or coefficient
        not the final point
        cal k-1 layer point unit normal vector
        vt -> vector tangential
        vn -> vector normal
    """

    # cal switching factor
    zt = k / (nz - 1)
    cs = np.exp(-coef * zt)
    
    # t2b top bdry to bottom bdry
    sign1 = 1 if t2b == 1 else -1
    
    # Calculate tangential vectors for all i at once
    vt_x = 0.5 * (x2d[k-1, 2:nx] - x2d[k-1, 0:nx-2])
    vt_z = 0.5 * (z2d[k-1, 2:nx] - z2d[k-1, 0:nx-2])
    
    # Calculate normal vectors
    len_vt = np.sqrt(vt_x**2 + vt_z**2)
    # Check if any length is zero and raise an error
    if np.any(len_vt == 0):
        raise ValueError("Tangential vector length is zero at one or more"
                         "points. Cannot normalize to get normal vector.")

    vn_x = np.zeros_like(vt_x)
    vn_z = np.zeros_like(vt_z)
    vn_x = sign1 * vt_z / len_vt
    vn_z = -sign1 * vt_x / len_vt
    
    # Calculate inner points
    R_x = x2d[k-1, 1:nx-1] - x2d[nz-1, 1:nx-1]
    R_z = z2d[k-1, 1:nx-1] - z2d[nz-1, 1:nx-1]
    R = np.sqrt(R_x**2 + R_z**2)
    
    # Calculate clustering factors
    R1 = step_len[nz-1] - step_len[k-1]
    r1 = step_len[k+1] - step_len[k-1]
    r2 = step_len[k] - step_len[k-1]
    
    c1 = r1 / R1 if R1 != 0 else 0
    c2 = r2 / r1 if r1 != 0 else 0
    
    # Calculate normal points
    x0 = x2d[k-1, 1:nx-1] + vn_x * c1 * R
    z0 = z2d[k-1, 1:nx-1] + vn_z * c1 * R
    
    # Calculate linear distance points
    xs = x2d[k-1, 1:nx-1] + c1 * (x2d[nz-1, 1:nx-1] - x2d[k-1, 1:nx-1])
    zs = z2d[k-1, 1:nx-1] + c1 * (z2d[nz-1, 1:nx-1] - z2d[k-1, 1:nx-1])
    
    # Assign to arrays
    x_pre[1:nx-1, 1] = cs * x0 + (1 - cs) * xs
    z_pre[1:nx-1, 1] = cs * z0 + (1 - cs) * zs
    x_pre[1:nx-1, 0] = x2d[k-1, 1:nx-1] + c2 * (x_pre[1:nx-1, 1] - x2d[k-1, 1:nx-1])
    z_pre[1:nx-1, 0] = z2d[k-1, 1:nx-1] + c2 * (z_pre[1:nx-1, 1] - z2d[k-1, 1:nx-1])
    
    # Handle boundaries
    # Mirror Symmetry Boundary
    x_pre[0, 0] = 2 * x_pre[1, 0] - x_pre[2, 0]
    z_pre[0, 0] = 2 * z_pre[1, 0] - z_pre[2, 0]
    x_pre[0, 1] = 2 * x_pre[1, 1] - x_pre[2, 1]
    z_pre[0, 1] = 2 * z_pre[1, 1] - z_pre[2, 1]
    
    x_pre[nx-1, 0] = 2 * x_pre[nx-2, 0] - x_pre[nx-3, 0]
    z_pre[nx-1, 0] = 2 * z_pre[nx-2, 0] - z_pre[nx-3, 0]
    x_pre[nx-1, 1] = 2 * x_pre[nx-2, 1] - x_pre[nx-3, 1]
    z_pre[nx-1, 1] = 2 * z_pre[nx-2, 1] - z_pre[nx-3, 1]


@numba.jit(nopython=True, nogil=True, cache=True)
def update_point(x2d: np.ndarray, z2d: np.ndarray,
                 var_th: np.ndarray, nx: int, k: int,
                 x_pre: np.ndarray, z_pre: np.ndarray) -> None:
    """
    Vectorized version of update_point
    """
    a = var_th[:, 0]
    b = var_th[:, 1]
    c = var_th[:, 2]
    d_x = var_th[:, 3]
    d_z = var_th[:, 4]
    u_x = var_th[:, 5]
    u_z = var_th[:, 6]
    
    # Calculate derivatives
    x_xi = 0.5 * (x_pre[2:nx, 0] - x_pre[0:nx-2, 0])
    z_xi = 0.5 * (z_pre[2:nx, 0] - z_pre[0:nx-2, 0])
    
    x_zt = 0.5 * (x_pre[1:nx-1, 1] - x2d[k-1, 1:nx-1])
    z_zt = 0.5 * (z_pre[1:nx-1, 1] - z2d[k-1, 1:nx-1])
    
    temp_x = x_pre[1:nx-1, 1] + x2d[k-1, 1:nx-1]
    temp_z = z_pre[1:nx-1, 1] + z2d[k-1, 1:nx-1]
    
    # Calculate cross derivatives
    x_xizt = 0.25 * (x_pre[2:nx, 1] + x2d[k-1, 0:nx-2] - x2d[k-1, 2:nx] - x_pre[0:nx-2, 1])
    z_xizt = 0.25 * (z_pre[2:nx, 1] + z2d[k-1, 0:nx-2] - z2d[k-1, 2:nx] - z_pre[0:nx-2, 1])
    
    # Calculate metric coefficients
    g11 = x_xi*x_xi + z_xi*z_xi
    g22 = x_zt*x_zt + z_zt*z_zt
    g12 = x_xi*x_zt + z_xi*z_zt
    
    # Update matrix coefficients
    a[:] = g22
    b[:] = -2 * (g22 + g11)
    c[:] = g22
    
    d_x[:] = -g11 * temp_x + 2 * g12 * x_xizt
    d_z[:] = -g11 * temp_z + 2 * g12 * z_xizt
    
    # Modify boundaries
    d_x[0] = d_x[0] - a[0] * x_pre[0, 0]
    d_z[0] = d_z[0] - a[0] * z_pre[0, 0]
    
    d_x[nx-3] = d_x[nx-3] - c[nx-3] * x_pre[nx-1, 0]
    d_z[nx-3] = d_z[nx-3] - c[nx-3] * z_pre[nx-1, 0]
    
    # Solve using Thomas algorithm
    thomas(a, b, c, d_x, d_z, u_x, u_z)
    
    # Update coordinates
    x2d[k, 1:nx-1] = u_x[:]
    z2d[k, 1:nx-1] = u_z[:]


@numba.jit(nopython=True, nogil=True, cache=True)
def assign_bdry_coords(x2d: np.ndarray, z2d: np.ndarray, 
                       nx: int, k: int) -> None:
    # Geometric symmetry boundary
    x2d[k, 0] = 2 * x2d[k, 1] - x2d[k, 2]
    z2d[k, 0] = 2 * z2d[k, 1] - z2d[k, 2]
    
    x2d[k, nx-1] = 2 * x2d[k, nx-2] - x2d[k, nx-3]
    z2d[k, nx-1] = 2 * z2d[k, nx-2] - z2d[k, nx-3]

