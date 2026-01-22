import numpy as np
import numba
from typing import Tuple
from .io_operetions import quality_export


def grid_quality_check(gdcurv: 'GridData', cfgs: dict) -> None:
    """Perform grid quality checks"""
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    ni = gdcurv.ni
    nk = gdcurv.nk

    var = np.zeros((nk, ni), dtype=np.float32)
    grid_export_dir = cfgs['grid_export_dir']
    if cfgs['check_orth'] == 1:
        quality_name = "orth"
        cal_orth(var, x2d, z2d, ni, nk)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_jac'] == 1:
        quality_name = "jacobi"
        cal_jacobi(var, x2d, z2d, ni, nk)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_ratio'] == 1:
        quality_name = "ratio"
        cal_ratio(var, x2d, z2d, ni, nk)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_step_xi'] == 1:
        quality_name = "step_xi"
        cal_step_x(var, x2d, z2d, ni)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_step_zt'] == 1:
        quality_name = "step_zt"
        cal_step_z(var, x2d, z2d, nk)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_smooth_xi'] == 1:
        quality_name = "smooth_xi"
        cal_smooth_x(var, x2d, z2d, ni)
        quality_export(gdcurv, var, grid_export_dir, quality_name)
    
    if cfgs['check_smooth_zt'] == 1:
        quality_name = "smooth_zt"
        cal_smooth_z(var, x2d, z2d, nk)
        quality_export(gdcurv, var, grid_export_dir, quality_name)


def cal_orth(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
             ni: int, nk: int) -> None:
    
    trans = 180 / np.pi  # arc to angle
    
    # Calculate derivatives using vectorized operations
    x_xi = x2d[:, 1:ni] - x2d[:, 0:ni-1]  # dx/dxi
    z_xi = z2d[:, 1:ni] - z2d[:, 0:ni-1]  # dz/dxi
    
    x_zt = x2d[1:nk, :] - x2d[0:nk-1, :]  # dx/dzt
    z_zt = z2d[1:nk, :] - z2d[0:nk-1, :]  # dz/dzt
    
    # Calculate dot product and lengths
    dot = (x_xi[:-1, :] * x_zt[:, :-1]) + (z_xi[:-1, :] * z_zt[:, :-1])
    len_xi = np.sqrt(x_xi[:-1, :]**2 + z_xi[:-1, :]**2)
    len_zt = np.sqrt(x_zt[:, :-1]**2 + z_zt[:, :-1]**2)
    
    # Calculate cosine and angle
    cos_angle = np.divide(dot, len_xi * len_zt, out=np.zeros_like(dot), where=(len_xi * len_zt) != 0)
    # Clip cosine to valid range for acos
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * trans
    var[:nk-1, :ni-1] = 90 - np.abs(angle - 90)
    
    # Handle boundaries
    var[:, ni-1] = var[:, ni-2]  # i = ni-1
    var[nk-1, :] = var[nk-2, :]  # k = nk-1


def cal_jacobi(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
               ni: int, nk: int):
    
    # Calculate derivatives using vectorized operations
    x_xi = x2d[:, 1:ni] - x2d[:, 0:ni-1]  # dx/dxi
    z_xi = z2d[:, 1:ni] - z2d[:, 0:ni-1]  # dz/dxi
    
    x_zt = x2d[1:nk, :] - x2d[0:nk-1, :]  # dx/dzt
    z_zt = z2d[1:nk, :] - z2d[0:nk-1, :]  # dz/dzt
    
    # Calculate Jacobian
    jacobian = (x_xi[:-1, :] * z_zt[:, :-1]) - (z_xi[:-1, :] * x_zt[:, :-1])
    var[:-1, :-1] = jacobian
    
    # Handle boundaries
    var[:, ni-1] = var[:, ni-2]  # i = ni-1
    var[nk-1, :] = var[nk-2, :]  # k = nk-1


def cal_ratio(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
              ni: int, nk: int):
    # Calculate derivatives using vectorized operations
    x_xi = x2d[:, 1:ni] - x2d[:, 0:ni-1]  # dx/dxi
    z_xi = z2d[:, 1:ni] - z2d[:, 0:ni-1]  # dz/dxi
    
    x_zt = x2d[1:nk, :] - x2d[0:nk-1, :]  # dx/dzt
    z_zt = z2d[1:nk, :] - z2d[0:nk-1, :]  # dz/dzt
    
    # Calculate lengths
    len_xi = np.sqrt(x_xi[:-1, :]**2 + z_xi[:-1, :]**2)
    len_zt = np.sqrt(x_zt[:, :-1]**2 + z_zt[:, :-1]**2)
    
    # Calculate ratios
    r1 = np.divide(len_xi, len_zt, out=np.zeros_like(len_xi), where=len_zt != 0)
    r2 = np.divide(len_zt, len_xi, out=np.zeros_like(len_zt), where=len_xi != 0)
    
    var[:nk-1, :ni-1] = np.maximum(r1, r2)
    
    # Handle boundaries
    var[:, ni-1] = var[:, ni-2]  # i = ni-1
    var[nk-1, :] = var[nk-2, :]  # k = nk-1


def cal_step_x(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
               ni: int):
    # Calculate step length in x direction
    x_xi = x2d[:, 1:ni] - x2d[:, 0:ni-1]
    z_xi = z2d[:, 1:ni] - z2d[:, 0:ni-1]
    
    step_lengths = np.sqrt(x_xi**2 + z_xi**2)
    var[:, :ni-1] = step_lengths
    
    # Handle boundary
    var[:, ni-1] = var[:, ni-2]  # i = ni-1


def cal_step_z(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
               nk: int):
    # Calculate step length in z direction
    x_zt = x2d[1:nk, :] - x2d[0:nk-1, :]
    z_zt = z2d[1:nk, :] - z2d[0:nk-1, :]
    
    step_lengths = np.sqrt(x_zt**2 + z_zt**2)
    var[:nk-1, :] = step_lengths
    
    # Handle boundary
    var[nk-1, :] = var[nk-2, :]  # k = nk-1


def cal_smooth_x(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
                 ni: int):
    # Calculate differences in x direction
    x_xi1 = x2d[:, 1:ni-1] - x2d[:, 0:ni-2]  # dx/dxi from left
    z_xi1 = z2d[:, 1:ni-1] - z2d[:, 0:ni-2]  # dz/dxi from left
    
    x_xi2 = x2d[:, 2:ni] - x2d[:, 1:ni-1]    # dx/dxi from right
    z_xi2 = z2d[:, 2:ni] - z2d[:, 1:ni-1]    # dz/dxi from right
    
    # Calculate lengths
    len_xi1 = np.sqrt(x_xi1**2 + z_xi1**2)
    len_xi2 = np.sqrt(x_xi2**2 + z_xi2**2)
    
    # Calculate ratios
    r1 = np.divide(len_xi1, len_xi2, out=np.zeros_like(len_xi1), where=len_xi2 != 0)
    r2 = np.divide(len_xi2, len_xi1, out=np.zeros_like(len_xi2), where=len_xi1 != 0)
    
    var[:, 1:ni-1] = np.maximum(r1, r2)
    
    # Handle boundaries
    var[:, 0] = var[:, 1]    # i = 0
    var[:, ni-1] = var[:, ni-2]  # i = ni-1


def cal_smooth_z(var: np.ndarray, x2d: np.ndarray, z2d: np.ndarray,
                 nk: int):
    # Calculate differences in z direction
    x_zt1 = x2d[1:nk-1, :] - x2d[0:nk-2, :]  # dx/dzt from below
    z_zt1 = z2d[1:nk-1, :] - z2d[0:nk-2, :]  # dz/dzt from below
    
    x_zt2 = x2d[2:nk, :] - x2d[1:nk-1, :]    # dx/dzt from above
    z_zt2 = z2d[2:nk, :] - z2d[1:nk-1, :]    # dz/dzt from above
    
    # Calculate lengths
    len_zt1 = np.sqrt(x_zt1**2 + z_zt1**2)
    len_zt2 = np.sqrt(x_zt2**2 + z_zt2**2)
    
    # Calculate ratios
    r1 = np.divide(len_zt1, len_zt2, out=np.zeros_like(len_zt1), where=len_zt2 != 0)
    r2 = np.divide(len_zt2, len_zt1, out=np.zeros_like(len_zt2), where=len_zt1 != 0)
    
    var[1:nk-1, :] = np.maximum(r1, r2)
    
    # Handle boundaries
    var[0, :] = var[1, :]      # k = 0
    var[nk-1, :] = var[nk-2, :]  # k = nk-1

@numba.jit(nopython=True, nogil=True, cache=True)
def dist_point2line_vectorized(x0: np.ndarray, z0: np.ndarray,
                               x1: np.ndarray, z1: np.ndarray,
                               x2: np.ndarray, z2: np.ndarray
                               ) -> np.ndarray:
    A = z2 - z1
    B = x1 - x2
    C = -z1 * B - x1 * A
    
    numerator = np.abs(A * x0 + B * z0 + C)
    denominator = np.sqrt(A * A + B * B)
    
    # Check for zero denominators
    if np.any(denominator <  1e-10):
        # Find indices where denominator is zero
        zero_indices = np.where(denominator < 1e-10)
        raise ValueError(
            f"Found {len(zero_indices[0])} points with zero denominator. "
            "This indicates coincident points p1 and p2 at these locations."
        )
    
    return numerator / denominator


def cal_min_dist(gdcurv: 'GridData') -> Tuple[int, int, float]:
    """
    Calculate the effective grid step size for determining the maximum
    stable time step in the finite difference method.
    """
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    ni = gdcurv.ni
    nk = gdcurv.nk
    k_indices = slice(1, nk-1)
    i_indices = slice(1, ni-1)
    
    #  Current point coordinates
    x0 = x2d[k_indices, i_indices]  # (nk-2, ni-2)
    z0 = z2d[k_indices, i_indices]
    
    #  Adjacent point coordinates
    x_left = x2d[k_indices, i_indices.start-1:i_indices.stop-1]
    z_left = z2d[k_indices, i_indices.start-1:i_indices.stop-1]
    x_right = x2d[k_indices, i_indices.start+1:i_indices.stop+1]
    z_right = z2d[k_indices, i_indices.start+1:i_indices.stop+1]
    x_up = x2d[k_indices.start+1:k_indices.stop+1, i_indices]
    z_up = z2d[k_indices.start+1:k_indices.stop+1, i_indices]
    x_down = x2d[k_indices.start-1:k_indices.stop-1, i_indices]
    z_down = z2d[k_indices.start-1:k_indices.stop-1, i_indices]
    
    # Calculate distances in four directions
    d1 = dist_point2line_vectorized(x0, z0, x_left, z_left, x_up, z_up)     
    d2 = dist_point2line_vectorized(x0, z0, x_right, z_right, x_up, z_up)   
    d3 = dist_point2line_vectorized(x0, z0, x_left, z_left, x_down, z_down) 
    d4 = dist_point2line_vectorized(x0, z0, x_right, z_right, x_down, z_down) 

    # locate min distance and index
    min_val = [np.min(d1), np.min(d2), np.min(d3), np.min(d4)]
    mat_idx = np.argmin(min_val) 
    dL_min = min(min_val)

    matrices = [d1, d2, d3, d4]
    target_mat = matrices[mat_idx]
    indx_k, indx_i = np.unravel_index(np.argmin(target_mat), target_mat.shape)
    # NOTE: need add 1,  target_mat.shape = (nk-2, ni-2)
    indx_i += 1
    indx_k += 1
    
    return int(indx_i), int(indx_k), float(dL_min)