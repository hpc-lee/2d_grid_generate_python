import numpy as np
import numba
from grid_data import GridData
from common.grid_math import inv_2x2, matmul_2x2, matmul_2x2_vec
from common.grid_math import inv_2x2_batch, matmul_2x2_batch
from common.grid_math import matmul_2x2_vec_batch, flip_coord_z
from common.algebra import zt_arc_stretch


def hyper_gene(gdcurv: GridData, cfgs: dict):
    nx = gdcurv.nx
    nz = gdcurv.nz
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    step = gdcurv.step
    
    t2b = cfgs['t2b']
    coef = cfgs['coef']
    flag_stretch = cfgs['flag_stretch']
    
    n = nx - 2
    coef_e = np.zeros(nx, dtype=np.float32)
    area = np.zeros((nx,2), dtype=np.float32)
    # cal_matrix Intermediate variable
    A = np.zeros((n, 2, 2), dtype=np.float32)  
    B = np.zeros((n, 2, 2), dtype=np.float32)  
    vec = np.zeros((n, 2), dtype=np.float32)
    # Allocate space for Thomas method
    a = np.zeros((n, 2, 2), dtype=np.float32)  
    b = np.zeros((n, 2, 2), dtype=np.float32)  
    c = np.zeros((n, 2, 2), dtype=np.float32)  
    d = np.zeros((n, 2), dtype=np.float32)  
    xz = np.zeros((n, 2), dtype=np.float32)  
    # Thomas Intermediate variable
    D = np.zeros((n, 2, 2), dtype=np.float32)  
    y = np.zeros((n, 2), dtype=np.float32)     

    # Generate k=1 layer coordinates to calculate smooth coefficients
    # Note: k=1 layer coordinates will be regenerated here
    k = 1    
    cal_matrix(x2d,z2d,nx,k,step,a,b,c,d,area,A,B,vec)
    modify_matrix(x2d,z2d,nx,k,a,b,c,d,coef_e)
    ThomasBlock(a,b,c,d,xz,D,y)
    update_coords(xz,x2d,z2d,nx,k)

    for k in range(1, nz): 
        cal_smooth_coef(coef,x2d,z2d,nx,nz,k,t2b,coef_e)
        cal_matrix(x2d,z2d,nx,k,step,a,b,c,d,area,A,B,vec)
        modify_matrix(x2d,z2d,nx,k,a,b,c,d,coef_e)
        ThomasBlock(a,b,c,d,xz,D,y)
        update_coords(xz,x2d,z2d,nx,k)
        print(f"number of layer is {k+1}")

    if t2b == 1:
        flip_coord_z(gdcurv.x2d, gdcurv.z2d)

    if flag_stretch == 1:
        zt_arc_stretch(gdcurv)


@numba.jit(nopython=True, nogil=True, cache=True )
def ThomasBlock(a: np.ndarray, b: np.ndarray, c: np.ndarray,
          d: np.ndarray, xz: np.ndarray, D: np.ndarray, y: np.ndarray) -> None:
    """
    Solving Block Tridiagonal Linear Systems Using the Thomas Algorithm
    """
    n = len(a)
    b0_inv = inv_2x2(b[0])
    D[0] = matmul_2x2(b0_inv, c[0])
    y[0] = matmul_2x2_vec(b0_inv, d[0])
    
    for i in range(1, n):
        aD = matmul_2x2(a[i], D[i-1])
        G = b[i] - aD
        G_inv = inv_2x2(G)
        
        D[i] = matmul_2x2(G_inv, c[i])
        ay = matmul_2x2_vec(a[i], y[i-1])
        y[i] = matmul_2x2_vec(G_inv, d[i] - ay)
    
    # Backward Process 
    xz[n-1] = y[n-1]
    for i in range(n-2, -1, -1):
        Dxz = matmul_2x2_vec(D[i], xz[i+1])
        xz[i] = y[i] - Dxz

@numba.jit(nopython=True, nogil=True, cache=True)
def cal_smooth_coef(coef: float, x2d: np.ndarray, 
                    z2d: np.ndarray, nx: int, nz: int, 
                    k: int, t2b: int, coef_e: np.ndarray) -> None:
    """
    Calculate smoothing coefficients
    """
    S = np.sqrt(k / (nz - 1))
    
    # Determine k1 based on k value
    k1 = 2 if k == 1 else k
    
    i_indices = np.arange(1, nx-1)  # i from 1 to nx-2
    
    # Calculate derivatives
    # x_xi, z_xi (xi derivatives at k1-1 layer)
    x_xi = 0.5 * (x2d[k1-1, i_indices + 1] - x2d[k1-1, i_indices - 1])
    z_xi = 0.5 * (z2d[k1-1, i_indices + 1] - z2d[k1-1, i_indices - 1])
    
    # x_zt, z_zt (zt derivatives: current layer - previous layer)
    x_zt = x2d[k1-1, i_indices] - x2d[k1-2, i_indices]
    z_zt = z2d[k1-1, i_indices] - z2d[k1-2, i_indices]
    
    # Lengths
    xi_len = np.sqrt(x_xi**2 + z_xi**2)
    zt_len = np.sqrt(x_zt**2 + z_zt**2)
    N_xi = zt_len / xi_len
    
    # Calculate xi lengths at k1-2 layer
    x_xi_plus_1 = x2d[k1-2, i_indices + 1] - x2d[k1-2, i_indices]
    z_xi_plus_1 = z2d[k1-2, i_indices + 1] - z2d[k1-2, i_indices]
    xi_plus1 = np.sqrt(x_xi_plus_1**2 + z_xi_plus_1**2)
    
    x_xi_minus_1 = x2d[k1-2, i_indices - 1] - x2d[k1-2, i_indices]
    z_xi_minus_1 = z2d[k1-2, i_indices - 1] - z2d[k1-2, i_indices]
    xi_minus1 = np.sqrt(x_xi_minus_1**2 + z_xi_minus_1**2)
    
    # Calculate xi lengths at k1-1 layer
    x_xi_plus_2 = x2d[k1-1, i_indices + 1] - x2d[k1-1, i_indices]
    z_xi_plus_2 = z2d[k1-1, i_indices + 1] - z2d[k1-1, i_indices]
    xi_plus2 = np.sqrt(x_xi_plus_2**2 + z_xi_plus_2**2)
    
    x_xi_minus_2 = x2d[k1-1, i_indices - 1] - x2d[k1-1, i_indices]
    z_xi_minus_2 = z2d[k1-1, i_indices - 1] - z2d[k1-1, i_indices]
    xi_minus2 = np.sqrt(x_xi_minus_2**2 + z_xi_minus_2**2)
    
    # Delta calculation
    d1 = xi_plus1 + xi_minus1
    d2 = xi_plus2 + xi_minus2
    delta = d1 / d2
    delta_mdfy = np.maximum(np.power(delta, 2/S), 0.01)
    
    # Normalization
    x_plus =  x_xi_plus_2 / xi_plus2
    z_plus =  z_xi_plus_2 / xi_plus2
    x_minus = x_xi_minus_2 / xi_minus2
    z_minus = z_xi_minus_2 / xi_minus2
    
    # Dot and det products
    dot = x_plus * x_minus + z_plus * z_minus
    det = x_plus * z_minus - z_plus * x_minus
    
    # Calculate angle theta
    # cal two normal vector clockwise angle.
    # the method from website
    # from plus vector to minus vector
    # z axis upward, so is -det
    theta = np.arctan2(-det, dot)
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    
    # Adjust theta based on t2b
    if t2b == 0:
        theta = 2*np.pi - theta
    
    # Calculate alpha
    alpha = np.where(theta < np.pi,
                     1.0 / (1 - np.power(np.cos(theta/2), 2)),
                     1.0)  # alpha = 1 when theta >= np.pi
    
    # Calculate final coefficient
    coef_e[i_indices] = coef * N_xi * S * delta_mdfy * alpha


@numba.jit(nopython=True, nogil=True, cache=True)
def cal_matrix(x2d: np.ndarray, z2d: np.ndarray, nx: int, 
               k: int, step: np.ndarray, a: np.ndarray,
               b: np.ndarray, c: np.ndarray, d: np.ndarray, 
               area: np.ndarray, A: np.ndarray, 
               B: np.ndarray, vec: np.ndarray) -> None:
    i_indices = np.arange(1, nx-1)
    
    x_next = x2d[k-1, i_indices + 1]
    z_next = z2d[k-1, i_indices + 1]
    x_curr = x2d[k-1, i_indices]
    z_curr = z2d[k-1, i_indices]
    x_prev = x2d[k-1, i_indices - 1]
    z_prev = z2d[k-1, i_indices - 1]
    
    x_xi0 = 0.5 * (x_next - x_prev)
    z_xi0 = 0.5 * (z_next - z_prev)
    
    diff_plus_x = x_next - x_curr
    diff_plus_z = z_next - z_curr
    diff_minus_x = x_prev - x_curr
    diff_minus_z = z_prev - z_curr

    arc_plus = np.sqrt(diff_plus_x**2 + diff_plus_z**2)
    arc_minus = np.sqrt(diff_minus_x**2 + diff_minus_z**2)
    arc_len = 0.5 * (arc_plus + arc_minus)
    
    # update area
    # arc_length -> area
    # area(:, 0) = A0  k-1 layer area 
    # area(:, 1) = A1  k layer area
    new_area_values = arc_len * step[k-1]
    if k == 1:
    # assume k=0 area equal k=1 area 
        area[i_indices, 0] = new_area_values  
        area[i_indices, 1] = new_area_values   
    else:
        area[i_indices, 0] = area[i_indices, 1] # A0 = A1 
        area[i_indices, 1] = new_area_values    # A1 = curr area
    
    temp = x_xi0**2 + z_xi0**2
    temp = np.maximum(temp, 1e-10)
    area0 = area[i_indices, 0]
    x_zt0 = -z_xi0 * area0 / temp
    z_zt0 = x_xi0 * area0 / temp
    
    # add damping factor, maybe inv(B) singular
    damping = 1e-7
    
    A[:, 0, 0] = x_zt0
    A[:, 0, 1] = z_zt0
    A[:, 1, 0] = z_zt0
    A[:, 1, 1] = -x_zt0
    
    B[:, 0, 0] = x_xi0 + damping
    B[:, 0, 1] = z_xi0
    B[:, 1, 0] = -z_xi0
    B[:, 1, 1] = x_xi0 + damping

    B_inv = inv_2x2_batch(B)
    mat = matmul_2x2_batch(B_inv, A)  # (n_points, 2, 2)
    
    vec[:, 1] = area[i_indices, 1]  
    d[:] = matmul_2x2_vec_batch(B_inv, vec)  # (n_points, 2, 2)
    
    a[:] = -0.5 * mat
    b[:] = np.eye(2)[None, :, :]
    c[:] = 0.5 * mat


@numba.jit(nopython=True, nogil=True, cache=True)
def modify_matrix(x2d: np.ndarray, z2d: np.ndarray, nx: int, k: int, 
                  a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                  d: np.ndarray, coef_e: np.ndarray) -> None:
    """
    Modifying Matrix Coefficients by Adding Dissipation Terms
    Modifying Matrix Coefficients with Boundary Conditions
    """
    # Second-order dissipation operator
    i_indices = np.arange(1, nx-1)

    coords_prev = np.empty((len(i_indices), 2), dtype=np.float32)
    coords_curr = np.empty((len(i_indices), 2), dtype=np.float32)
    coords_next = np.empty((len(i_indices), 2), dtype=np.float32)
    coords_prev[:,0] = x2d[k-1, i_indices - 1]
    coords_prev[:,1] = z2d[k-1, i_indices - 1]
    coords_curr[:,0] = x2d[k-1, i_indices]
    coords_curr[:,1] = z2d[k-1, i_indices]
    coords_next[:,0] = x2d[k-1, i_indices + 1]
    coords_next[:,1] = z2d[k-1, i_indices + 1]
    
    coef_i = 2 * coef_e 
    
    d[:] += coef_e[i_indices, None] * (coords_prev + coords_next - 2 * coords_curr)
    
    eye_2x2 = np.eye(2)
    
    a[:] -= coef_i[i_indices, None, None] * eye_2x2
    b[:] += 2 * coef_i[i_indices, None, None] * eye_2x2
    c[:] -= coef_i[i_indices, None, None] * eye_2x2

    # Modify boundary conditions
    # Only use float boundary
    b[0] += a[0]
    b[-1] += c[-1]

@numba.jit(nopython=True, nogil=True, cache=True)
def update_coords(xz: np.ndarray, x2d: np.ndarray, z2d: np.ndarray, 
                  nx: int, k: int) -> None:
    """
    Assign coordinates based on solution
    """
    # Interior points: i from 1 to nx-2
    i_indices = np.arange(1, nx-1)
    
    x2d[k, i_indices] = x2d[k-1, i_indices] + xz[(i_indices-1), 0]
    z2d[k, i_indices] = z2d[k-1, i_indices] + xz[(i_indices-1), 1]
    
    # Left boundary (i=0): floating boundary
    x2d[k, 0] = x2d[k-1, 0] + (x2d[k, 1] - x2d[k-1, 1])
    z2d[k, 0] = z2d[k-1, 0] + (z2d[k, 1] - z2d[k-1, 1])
    
    # Right boundary (i=nx-1): floating boundary (with epsilon=0)
    x2d[k, nx-1] = x2d[k-1, nx-1] + (x2d[k, nx-2] - x2d[k-1, nx-2])
    z2d[k, nx-1] = z2d[k-1, nx-1] + (z2d[k, nx-2] - z2d[k-1, nx-2])
