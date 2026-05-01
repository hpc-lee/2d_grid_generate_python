import sys
import ctypes
import numpy as np
from pathlib import Path
from grid_data import GridData


def grid_sample(gdcurv: GridData, coefs: list,
                use_c: bool = False, lib=None) -> GridData:
    nx = gdcurv.nx
    nz = gdcurv.nz
    coef_x = coefs[0]
    coef_z = coefs[1]
    if coef_x < 1 or coef_z < 1:
        print("ERROR: coef must be >= 1")
        sys.exit(1)
    nx_new = (nx-1) * coef_x + 1
    nz_new = (nz-1) * coef_z + 1
    if nx_new < nx or nz_new < nz:
        print("ERROR: only support upsample, nx_new must >= nx")
        sys.exit(1)

    gdcurv_new = GridData(nx_new, nz_new)
    sample_interp(gdcurv_new, gdcurv, use_c=use_c, lib=lib)
    return gdcurv_new


def sample_interp(gdcurv_new: GridData, gdcurv: GridData,
                  use_c: bool = False, lib=None):
    """Bilinear interpolation: first z-direction, then x-direction."""
    if use_c and lib is not None:
        _sample_interp_c(gdcurv_new, gdcurv, lib)
    else:
        _sample_interp_py(gdcurv_new, gdcurv)


def _sample_interp_c(gdcurv_new: GridData, gdcurv: GridData, lib):
    """C backend for sample_interp — tight loops, fp32 arithmetic."""
    # NumPy arrays must be C-contiguous and float32 for ctypes
    x2d = np.ascontiguousarray(gdcurv.x2d, dtype=np.float32)
    z2d = np.ascontiguousarray(gdcurv.z2d, dtype=np.float32)
    x2d_new = np.ascontiguousarray(gdcurv_new.x2d, dtype=np.float32)
    z2d_new = np.ascontiguousarray(gdcurv_new.z2d, dtype=np.float32)

    lib.sample_interp_c(
        x2d, z2d, x2d_new, z2d_new,
        ctypes.c_int(gdcurv.nx), ctypes.c_int(gdcurv.nz),
        ctypes.c_int(gdcurv_new.nx), ctypes.c_int(gdcurv_new.nz),
    )

    # Copy results back
    gdcurv_new.x2d[:] = x2d_new
    gdcurv_new.z2d[:] = z2d_new


def _sample_interp_py(gdcurv_new: GridData, gdcurv: GridData):
    """Python/NumPy implementation — vectorized column/row interpolation."""
    nx = gdcurv.nx
    nz = gdcurv.nz
    nx_new = gdcurv_new.nx
    nz_new = gdcurv_new.nz

    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    x2d_new = gdcurv_new.x2d
    z2d_new = gdcurv_new.z2d

    # First pass: interpolate along z-direction
    u = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    r_z = np.linspace(0.0, 1.0, nz_new, dtype=np.float32)
    m_z = np.searchsorted(u, r_z, side='right') - 1
    m_z = np.clip(m_z, 0, nz-2)

    for i in range(nx):
        x_col = x2d[:, i]
        z_col = z2d[:, i]
        ratio = (r_z - u[m_z]) / (u[m_z+1] - u[m_z])
        x2d_new[:, i] = x_col[m_z] + (x_col[m_z+1] - x_col[m_z]) * ratio
        z2d_new[:, i] = z_col[m_z] + (z_col[m_z+1] - z_col[m_z]) * ratio

    # Second pass: interpolate along x-direction
    v = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    r_x = np.linspace(0.0, 1.0, nx_new, dtype=np.float32)
    m_x = np.searchsorted(v, r_x, side='right') - 1
    m_x = np.clip(m_x, 0, nx-2)

    for k_new in range(nz_new):
        x_temp = x2d_new[k_new, :nx].copy()
        z_temp = z2d_new[k_new, :nx].copy()
        ratio = (r_x - v[m_x]) / (v[m_x+1] - v[m_x])
        x2d_new[k_new, :] = x_temp[m_x] + (x_temp[m_x+1] - x_temp[m_x]) * ratio
        z2d_new[k_new, :] = z_temp[m_x] + (z_temp[m_x+1] - z_temp[m_x]) * ratio
