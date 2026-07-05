import sys
import numpy as np
from typing import Tuple


class GridData:
    def __init__(self, nx=0, nz=0):
        self.nx = nx
        self.nz = nz

        # File Output in MPI Parallel Simulations
        self.fname_part = "px0_pz0" 
        # global_index_of_first_physical_points
        self.gni1 = 0
        self.gnk1 = 0
        # ni and nk are variables only required for MPI parallelization
        # with ghost points, which are unnecessary here; however, 
        # they are assigned for unifying data export across different methods.
        self.ni = nx
        self.nk = nz
        
        self.x2d = np.zeros((nz, nx), dtype=np.float32)
        self.z2d = np.zeros((nz, nx), dtype=np.float32)
        
        self.step = np.zeros((nz-1), dtype=np.float32)


def grid_init_set(cfgs: dict) ->GridData :
    geometry_file = cfgs['geometry_input_file']
    step_file = cfgs['step_input_file']

    nx = cfgs['number_of_grid_points_x']
    nz = cfgs['number_of_grid_points_z']
    gdcurv = GridData(nx, nz)
    
    # Read step lengths
    try:
        with open(step_file, 'r') as fp:
            steps = [line.strip() for line in fp
                     if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        print(f"read step file error: {e}")

    if (len(steps) != nz-1):
        print(f"ERROR: step data length {len(steps)} does not match nz-1={nz-1}")
        sys.exit(1)
    gdcurv.step = np.array(steps, dtype=np.float32)

    # Read geometry file
    try:
        with open(geometry_file, "r") as fp:
            bdry_coords = [list(map(float, line.split())) 
                           for line in fp 
                           if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        print(f"read geometry file error: {e}")

    
    if (len(bdry_coords) != 2*nx):
        print(f"ERROR: grid direction is z, bdry file point number is wrong!"
              f"bdry file point is {len(bdry_coords)/2} does not match nx={nx}")
        sys.exit(1)
    bdry_coords = np.array(bdry_coords, dtype=np.float32)
    gdcurv.x2d[0, :] = bdry_coords[:nx, 0]
    gdcurv.z2d[0, :] = bdry_coords[:nx, 1]
    gdcurv.x2d[nz-1, :] = bdry_coords[nx:, 0]
    gdcurv.z2d[nz-1, :] = bdry_coords[nx:, 1]

    return gdcurv


    
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


def cal_min_dist(gdcurv: GridData) -> Tuple[int, int, float]:
    """
    Compute the minimum step size (or minimal spacing) of grid points.
    """
    x2d = gdcurv.x2d  # (nz, nx)
    z2d = gdcurv.z2d  # (nz, nx)
    nx, nz = gdcurv.nx, gdcurv.nz
    
    k_indices = slice(1, nz-1)
    i_indices = slice(1, nx-1)
    
    #  Current point coordinates
    x0 = x2d[k_indices, i_indices]  # (nz-2, nx-2)
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
    # NOTE: need add 1,  target_mat.shape = (nz-2, nx-2)
    indx_i += 1
    indx_k += 1
    
    return int(indx_i), int(indx_k), float(dL_min)

def cfgs_print(cfgs: dict):
    print(f"number of total grid points x is {cfgs['number_of_grid_points_x']}")
    print(f"number of total grid points z is {cfgs['number_of_grid_points_z']}")
    
    print(f"input geometry file is \n {cfgs['geometry_input_file']}")
    print(f"input step file is \n {cfgs['step_input_file']}")
    print(f"export grid dir is \n {cfgs['grid_export_dir']}")
    print("-------------------------------------------------------")
    
    if cfgs['check_orth'] == 1:
        print("------- check grid orthogonality-------")
    if cfgs['check_jac'] == 1:
        print("------- check grid jacobi-------")
    if cfgs['check_ratio'] == 1:
        print("------- check grid ratio-------")
    if cfgs['check_step_xi'] == 1:
        print("------- check grid step xi direction-------")
    if cfgs['check_step_zt'] == 1:
        print("------- check grid step zt direction-------")
    if cfgs['check_smooth_xi'] == 1:
        print("------- check grid smooth xi direction-------")
    if cfgs['check_smooth_zt'] == 1:
        print("------- check grid smooth zt direction-------")
    
    print(f"coef is {cfgs['coef']}")
    if cfgs['t2b'] == 1:
        print("top(bdry_2) to bottom(bdry_1)")
    else:
        print("bottom(bdry_1) to top(bdry_2)")