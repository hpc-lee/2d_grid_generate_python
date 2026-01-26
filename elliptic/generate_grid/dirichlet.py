import numpy as np
import numba
from mpi4py import MPI
from mympi import grid_comm_optimized
from grid_utils import update_SOR, interp_inner_source
from grid_utils import compute_residual, source


def diri_gene(gdcurv: 'GridData', cfgs: dict, mympi: 'MPIclass', lib=None):
    """
    Dirichlet boundary condition grid generation
    """
    err_threshold = cfgs['method']["dirichlet"]['iter_err']
    max_iter = int(cfgs['method']["dirichlet"]['max_iter'])
    coef = cfgs['method']["dirichlet"]['coef']
    weight = cfgs['method']["dirichlet"]['weight']
    coef = np.array(coef, dtype=np.float32)
    weight = np.array(weight, dtype=np.float32)

    nx = gdcurv.nx
    nz = gdcurv.nz
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    total_nx = gdcurv.total_nx
    total_nz = gdcurv.total_nz

    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    x2d_tmp = gdcurv.x2d_tmp
    z2d_tmp = gdcurv.z2d_tmp

    comm = mympi.comm
    myid = mympi.myid
    neighid = mympi.neighid

    src = source()
    src.init_src(gdcurv)
    
    p_x1 = np.zeros((nz, 2), dtype=np.float32)
    p_x2 = np.zeros((nz, 2), dtype=np.float32)
    p_z1 = np.zeros((nx, 2), dtype=np.float32)
    p_z2 = np.zeros((nx, 2), dtype=np.float32)
    g11_x1 = np.zeros(nz, dtype=np.float32)
    g11_x2 = np.zeros(nz, dtype=np.float32)
    g22_z1 = np.zeros(nx, dtype=np.float32)
    g22_z2 = np.zeros(nx, dtype=np.float32)

    ghost_point_cal(x2d, z2d, nx, nz, p_x1, p_x2, p_z1, p_z2,
                    g11_x1, g11_x2, g22_z1, g22_z2, neighid)

    set_src_diri(x2d, z2d, p_x1, p_x2, p_z1, p_z2, 
                 g11_x1, g11_x2, g22_z1, g22_z2, 
                 gdcurv, src, mympi)
    if (cfgs['execute_C_code'] == 1):
        lib.interp_inner_source_c(src.P, src.P_x1, src.P_x2, 
                              src.P_z1, src.P_z2, src.Q, 
                              src.Q_x1, src.Q_x2, src.Q_z1, 
                              src.Q_z2, nx, nz, gni1, gnk1,
                              total_nx, total_nz, coef, 
                              weight)
    else:
        interp_inner_source(src.P, src.P_x1, src.P_x2, 
                            src.P_z1, src.P_z2, src.Q, 
                            src.Q_x1, src.Q_x2, src.Q_z1, 
                            src.Q_z2, nx, nz, gni1, gnk1,
                            total_nx, total_nz, coef, 
                            weight)

    # Copy coordinates
    x2d_tmp[:] = x2d[:]
    z2d_tmp[:] = z2d[:]

    # SOR coefficient (Gauss-Seidel when w=1.0)
    omega = 1.0

    local_max = np.zeros(2, dtype=np.float32) 
    global_max = np.zeros(2, dtype=np.float32) 

    for n_iter in range(1, max_iter + 1):
        # Update grid using SOR
        update_SOR(x2d, z2d, x2d_tmp, z2d_tmp, nx, nz, src.P, src.Q, omega)
        # Boundary exchange
        grid_comm_optimized(mympi, x2d_tmp, z2d_tmp)

        if (cfgs['execute_C_code'] == 1):
            lib.compute_residual_c(x2d, z2d, x2d_tmp, z2d_tmp, local_max, nx, nz)
        else:
            compute_residual(x2d, z2d, x2d_tmp, z2d_tmp, local_max, nx, nz)

        # Global reduction of maximum errors
        comm.Allreduce(local_max, global_max, op=MPI.MAX)
        max_resi = global_max[0]
        max_resk = global_max[1]
        # Convergence check (break early if converged)
        if max_resi < err_threshold and max_resk < err_threshold:
            if myid == 0:
                print(f"Converged at iteration {n_iter}")
                print(f"Final errors: max_resi={max_resi:.6e}, max_resk={max_resk:.6e}")
            break
        
        # Status reporting (rank 0 only)
        if myid == 0 and n_iter % 1 == 0:
            print(f"Iter {n_iter:4d}: max_resi={max_resi:.3e}, max_resk={max_resk:.3e}")
        
        # Efficient array swap using memory views
        x2d, x2d_tmp = x2d_tmp, x2d
        z2d, z2d_tmp = z2d_tmp, z2d
        
        # Update source terms with new grid
        set_src_diri(x2d, z2d, p_x1, p_x2, p_z1, p_z2, 
                    g11_x1, g11_x2, g22_z1, g22_z2, 
                    gdcurv, src, mympi)
        if (cfgs['execute_C_code'] == 1):
            lib.interp_inner_source_c(src.P, src.P_x1, src.P_x2, 
                                  src.P_z1, src.P_z2, src.Q, 
                                  src.Q_x1, src.Q_x2, src.Q_z1, 
                                  src.Q_z2, nx, nz, gni1, gnk1,
                                  total_nx, total_nz, coef, 
                                  weight)
        else:
            interp_inner_source(src.P, src.P_x1, src.P_x2, 
                                src.P_z1, src.P_z2, src.Q, 
                                src.Q_x1, src.Q_x2, src.Q_z1, 
                                src.Q_z2, nx, nz, gni1, gnk1,
                                total_nx, total_nz, coef, 
                                weight)
    else:
        max_resi = global_max[0]
        max_resk = global_max[1]
        if myid == 0:
            print(f"MAX ITERATIONS REACHED ({max_iter})")
            print(f"Final errors: max_resi={max_resi:.6e}, max_resk={max_resk:.6e}")

    # Update grid pointers
    gdcurv.x2d = x2d
    gdcurv.z2d = z2d


def ghost_point_cal(x2d: np.ndarray, z2d: np.ndarray, nx: int, 
              nz: int, p_x1: np.ndarray, p_x2: np.ndarray, 
              p_z1: np.ndarray, p_z2: np.ndarray, 
              g11_x1: np.ndarray, g11_x2: np.ndarray, 
              g22_z1: np.ndarray, g22_z2: np.ndarray, 
              neighid: np.ndarray) -> None:
  # Small epsilon to avoid division by zero in normal vector calculation
    EPS = 1e-12

    if neighid[0] == MPI.PROC_NULL:
        # Valid range: z-direction k ∈ [1, nz-2] (exclude boundary points)
        k_range = slice(1, nz-1)
        
        # Extract core grid points (physical positions)
        x_inner = x2d[k_range, 1]     # Inner grid point (k, 1) → adjacent to ghost point
        z_inner = z2d[k_range, 1]     
        x_bdry = x2d[k_range, 0]      # boundary point position (k, 0)
        z_bdry = z2d[k_range, 0]      
        x_zt_plus = x2d[2:, 0]      # z+1 direction point (k+1, 0)
        z_zt_plus = z2d[2:, 0]      
        x_zt_minus = x2d[:-2, 0]      # z-1 direction point (k-1, 0)
        z_zt_minus = z2d[:-2, 0]       

        # 1. Calculate xi-direction gradient (bdry → inner point)
        grad_xi_x = x_inner - x_bdry
        grad_xi_z = z_inner - z_bdry
        
        # 2. Calculate zt-direction central gradient (tangent in z-direction)
        grad_zt_x = 0.5 * (x_zt_plus - x_zt_minus)
        grad_zt_z = 0.5 * (z_zt_plus - z_zt_minus)
        
        # 3. Calculate normal vector (orthogonal to zt direction)
        vn_xi = grad_zt_z                  # xi component of normal vector
        vn_zt = -grad_zt_x                 # zt component of normal vector
        vn_len = np.hypot(vn_xi, vn_zt) + EPS  
        
        # 4. Normalize normal vector
        vn_xi_norm = vn_xi / vn_len
        vn_zt_norm = vn_zt / vn_len
        
        # 5. Project gradient onto normal vector direction
        proj_dot = grad_xi_x * vn_xi_norm + grad_xi_z * vn_zt_norm
        proj_x = proj_dot * vn_xi_norm
        proj_z = proj_dot * vn_zt_norm
        
        # 6. Update ghost point coords and g11 parameter (offset opposite to normal vector)
        p_x1[k_range, 0] = x_bdry - proj_x
        p_x1[k_range, 1] = z_bdry - proj_z
        g11_x1[k_range] = proj_x**2 + proj_z**2

    if neighid[1] == MPI.PROC_NULL:
        # Valid range: z-direction k ∈ [1, nz-2]
        k_range = slice(1, nz-1)
        
        # Extract core grid points
        x_inner = x2d[k_range, nx-2]   # Inner grid point (k, nx-2)
        z_inner = z2d[k_range, nx-2]       
        x_bdry = x2d[k_range, nx-1]    # Boundary point position (k, nx-1)
        z_bdry = z2d[k_range, nx-1]       
        x_zt_plus = x2d[2:, nx-1]     # z+1 direction point (k+1, nx-1)
        z_zt_plus = z2d[2:, nx-1]       
        x_zt_minus = x2d[:-2, nx-1]     # z-1 direction point (k-1, nx-1)
        z_zt_minus = z2d[:-2, nx-1]       

        # Gradient calculation (same logic as left boundary, different grid positions)
        grad_xi_x = x_bdry - x_inner
        grad_xi_z = z_bdry - z_inner
        grad_zt_x = 0.5 * (x_zt_plus - x_zt_minus)
        grad_zt_z = 0.5 * (z_zt_plus - z_zt_minus)
        
        # Normal vector calculation
        vn_xi = grad_zt_z
        vn_zt = -grad_zt_x
        vn_len = np.hypot(vn_xi, vn_zt) + EPS
        vn_xi_norm = vn_xi / vn_len
        vn_zt_norm = vn_zt / vn_len
        
        # Projection calculation
        proj_dot = grad_xi_x * vn_xi_norm + grad_xi_z * vn_zt_norm
        proj_x = proj_dot * vn_xi_norm
        proj_z = proj_dot * vn_zt_norm
        
        # Update ghost point (offset along normal vector)
        p_x2[k_range, 0] = x_bdry + proj_x
        p_x2[k_range, 1] = z_bdry + proj_z
        g11_x2[k_range] = proj_x**2 + proj_z**2

    if neighid[2] == MPI.PROC_NULL:
        # Valid range: x-direction i ∈ [1, nx-2]
        i_range = slice(1, nx-1)
        
        # Extract core grid points
        x_inner = x2d[1, i_range]       # Inner grid point (1, i)
        z_inner = z2d[1, i_range]       
        x_bdry = x2d[0, i_range]        # Boundary point position (0, i)
        z_bdry = z2d[0, i_range]       
        x_xi_plus = x2d[0, 2:]        # x+1 direction point (0, i+1)
        z_xi_plus = z2d[0, 2:]       
        x_xi_minus = x2d[0, :-2]        # x-1 direction point (0, i-1)
        z_xi_minus = z2d[0, :-2]       

        # 1. Calculate zt-direction gradient (bdry→ inner point)
        grad_zt_x = x_inner - x_bdry
        grad_zt_z = z_inner - z_bdry
        
        # 2. Calculate xi-direction central gradient (tangent in x-direction)
        grad_xi_x = 0.5 * (x_xi_plus - x_xi_minus)
        grad_xi_z = 0.5 * (z_xi_plus - z_xi_minus)
        
        # 3. Calculate normal vector (orthogonal to xi direction)
        vn_xi = -grad_xi_z
        vn_zt = grad_xi_x
        vn_len = np.hypot(vn_xi, vn_zt) + EPS
        
        # 4. Normalize normal vector
        vn_xi_norm = vn_xi / vn_len
        vn_zt_norm = vn_zt / vn_len
        
        # 5. Project gradient onto normal vector direction
        proj_dot = grad_zt_x * vn_xi_norm + grad_zt_z * vn_zt_norm
        proj_x = proj_dot * vn_xi_norm
        proj_z = proj_dot * vn_zt_norm
        
        # 6. Update ghost point coords and g22 parameter
        p_z1[i_range, 0] = x_bdry - proj_x
        p_z1[i_range, 1] = z_bdry - proj_z
        g22_z1[i_range] = proj_x**2 + proj_z**2

    if neighid[3] == MPI.PROC_NULL:
        # Valid range: x-direction i ∈ [1, nx-2]
        i_range = slice(1, nx-1)
        
        # Extract core grid points
        x_inner = x2d[nz-2, i_range]    # Inner grid point (nz-2, i)
        z_inner = z2d[nz-2, i_range]       
        x_bdry = x2d[nz-1, i_range]    # Boundary point position (nz-1, i)
        z_bdry = z2d[nz-1, i_range]       
        x_xi_plus = x2d[nz-1, 2:]     # x+1 direction point (nz-1, i+1)
        z_xi_plus = z2d[nz-1, 2:]       
        x_xi_minus = x2d[nz-1, :-2]     # x-1 direction point (nz-1, i-1)
        z_xi_minus = z2d[nz-1, :-2]       

        # Gradient calculation
        grad_zt_x = x_bdry - x_inner
        grad_zt_z = z_bdry - z_inner
        grad_xi_x = 0.5 * (x_xi_plus - x_xi_minus)
        grad_xi_z = 0.5 * (z_xi_plus - z_xi_minus)
        
        # Normal vector calculation
        vn_xi = -grad_xi_z
        vn_zt = grad_xi_x
        vn_len = np.hypot(vn_xi, vn_zt) + EPS
        vn_xi_norm = vn_xi / vn_len
        vn_zt_norm = vn_zt / vn_len
        
        # Projection calculation
        proj_dot = grad_zt_x * vn_xi_norm + grad_zt_z * vn_zt_norm
        proj_x = proj_dot * vn_xi_norm
        proj_z = proj_dot * vn_zt_norm
        
        # Update ghost point
        p_z2[i_range, 0] = x_bdry + proj_x
        p_z2[i_range, 1] = z_bdry + proj_z
        g22_z2[i_range] = proj_x**2 + proj_z**2


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_x1_diri(x2d, z2d, p_x1, g11_x1, gnk1, nx, nz, EPS, P_x1_loc, Q_x1_loc):
    k_start = 1
    k_end = nz - 1
    k_range_slice = slice(k_start, k_end)
    gnk_range = gnk1 + np.arange(nz - 2)
    
    bdry_point_x = x2d[k_range_slice, 0]
    bdry_point_z = z2d[k_range_slice, 0]
    bdry_zt_plus_x = x2d[2:nz, 0]
    bdry_zt_plus_z = z2d[2:nz, 0]
    bdry_zt_minus_x = x2d[0:nz-2, 0]
    bdry_zt_minus_z = z2d[0:nz-2, 0]
    inner_point_x = x2d[k_range_slice, 1]
    inner_point_z = z2d[k_range_slice, 1]

    grad_zt_x = 0.5 * (bdry_zt_plus_x - bdry_zt_minus_x)
    grad_zt_z = 0.5 * (bdry_zt_plus_z - bdry_zt_minus_z)
    
    grad_ztzt_x = bdry_zt_plus_x + bdry_zt_minus_x - 2 * bdry_point_x
    grad_ztzt_z = bdry_zt_plus_z + bdry_zt_minus_z - 2 * bdry_point_z
    
    grad_xi_x = inner_point_x - bdry_point_x
    grad_xi_z = inner_point_z - bdry_point_z
    
    grad_xixi_x = p_x1[k_range_slice, 0] + inner_point_x - 2 * bdry_point_x
    grad_xixi_z = p_x1[k_range_slice, 1] + inner_point_z - 2 * bdry_point_z
    
    g22 = grad_zt_x**2 + grad_zt_z**2 + EPS
    
    term1_P = (grad_xi_x * grad_xixi_x + grad_xi_z * grad_xixi_z) / (g11_x1[k_range_slice] + EPS)
    term2_P = (grad_xi_x * grad_ztzt_x + grad_xi_z * grad_ztzt_z) / g22
    P_x1_loc[gnk_range] = -(term1_P + term2_P)
    
    term1_Q = (grad_zt_x * grad_xixi_x + grad_zt_z * grad_xixi_z) / (g11_x1[k_range_slice] + EPS)
    term2_Q = (grad_zt_x * grad_ztzt_x + grad_zt_z * grad_ztzt_z) / g22
    Q_x1_loc[gnk_range] = -(term1_Q + term2_Q)

@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_x2_diri(x2d, z2d, p_x2, g11_x2, gnk1, nx, nz, EPS, P_x2_loc, Q_x2_loc):
    k_range_slice = slice(1, nz-1)
    gnk_range = gnk1 + np.arange(nz - 2)
    
    bdry_point_x = x2d[k_range_slice, nx-1]
    bdry_point_z = z2d[k_range_slice, nx-1]
    bdry_zt_plus_x = x2d[2:nz, nx-1]
    bdry_zt_plus_z = z2d[2:nz, nx-1]
    bdry_zt_minus_x = x2d[0:nz-2, nx-1]
    bdry_zt_minus_z = z2d[0:nz-2, nx-1]
    inner_point_x = x2d[k_range_slice, nx-2]
    inner_point_z = z2d[k_range_slice, nx-2]

    grad_zt_x = 0.5 * (bdry_zt_plus_x - bdry_zt_minus_x)
    grad_zt_z = 0.5 * (bdry_zt_plus_z - bdry_zt_minus_z)
    
    grad_ztzt_x = bdry_zt_plus_x + bdry_zt_minus_x - 2 * bdry_point_x
    grad_ztzt_z = bdry_zt_plus_z + bdry_zt_minus_z - 2 * bdry_point_z
    
    grad_xi_x = bdry_point_x - inner_point_x
    grad_xi_z = bdry_point_z - inner_point_z
    
    grad_xixi_x = p_x2[k_range_slice, 0] + inner_point_x - 2 * bdry_point_x
    grad_xixi_z = p_x2[k_range_slice, 1] + inner_point_z - 2 * bdry_point_z
    
    g22 = grad_zt_x**2 + grad_zt_z**2 + EPS
    
    term1_P = (grad_xi_x * grad_xixi_x + grad_xi_z * grad_xixi_z) / (g11_x2[k_range_slice] + EPS)
    term2_P = (grad_xi_x * grad_ztzt_x + grad_xi_z * grad_ztzt_z) / g22
    P_x2_loc[gnk_range] = -(term1_P + term2_P)
    
    term1_Q = (grad_zt_x * grad_xixi_x + grad_zt_z * grad_xixi_z) / (g11_x2[k_range_slice] + EPS)
    term2_Q = (grad_zt_x * grad_ztzt_x + grad_zt_z * grad_ztzt_z) / g22
    Q_x2_loc[gnk_range] = -(term1_Q + term2_Q)


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_z1_diri(x2d, z2d, p_z1, g22_z1, gni1, nx, nz, EPS, P_z1_loc, Q_z1_loc):
    i_start = 1
    i_end = nx - 1
    i_range_slice = slice(i_start, i_end)
    gni_range = gni1 + np.arange(nx - 2)
    
    bdry_point_x = x2d[0, i_range_slice]
    bdry_point_z = z2d[0, i_range_slice]
    bdry_xi_plus_x = x2d[0, 2:nx]
    bdry_xi_plus_z = z2d[0, 2:nx]
    bdry_xi_minus_x = x2d[0, 0:nx-2]
    bdry_xi_minus_z = z2d[0, 0:nx-2]
    inner_point_x = x2d[1, i_range_slice]
    inner_point_z = z2d[1, i_range_slice]

    grad_xi_x = 0.5 * (bdry_xi_plus_x - bdry_xi_minus_x)
    grad_xi_z = 0.5 * (bdry_xi_plus_z - bdry_xi_minus_z)
    
    grad_xixi_x = bdry_xi_plus_x + bdry_xi_minus_x - 2 * bdry_point_x
    grad_xixi_z = bdry_xi_plus_z + bdry_xi_minus_z - 2 * bdry_point_z
    
    grad_zt_x = inner_point_x - bdry_point_x
    grad_zt_z = inner_point_z - bdry_point_z
    
    grad_ztzt_x = p_z1[i_range_slice, 0] + inner_point_x - 2 * bdry_point_x
    grad_ztzt_z = p_z1[i_range_slice, 1] + inner_point_z - 2 * bdry_point_z
    
    g11 = grad_xi_x**2 + grad_xi_z**2 + EPS
    
    term1_P = (grad_xi_x * grad_xixi_x + grad_xi_z * grad_xixi_z) / g11
    term2_P = (grad_xi_x * grad_ztzt_x + grad_xi_z * grad_ztzt_z) / (g22_z1[i_range_slice] + EPS)
    P_z1_loc[gni_range] = -(term1_P + term2_P)
    
    term1_Q = (grad_zt_x * grad_xixi_x + grad_zt_z * grad_xixi_z) / g11
    term2_Q = (grad_zt_x * grad_ztzt_x + grad_zt_z * grad_ztzt_z) / (g22_z1[i_range_slice] + EPS)
    Q_z1_loc[gni_range] = -(term1_Q + term2_Q)


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_z2_diri(x2d, z2d, p_z2, g22_z2, gni1, nx, nz, EPS, P_z2_loc, Q_z2_loc):
    i_range_slice = slice(1, nx-1)
    gni_range = gni1 + np.arange(nx - 2)
    
    bdry_point_x = x2d[nz-1, i_range_slice]
    bdry_point_z = z2d[nz-1, i_range_slice]
    bdry_xi_plus_x = x2d[nz-1, 2:nx]
    bdry_xi_plus_z = z2d[nz-1, 2:nx]
    bdry_xi_minus_x = x2d[nz-1, 0:nx-2]
    bdry_xi_minus_z = z2d[nz-1, 0:nx-2]
    inner_point_x = x2d[nz-2, i_range_slice]
    inner_point_z = z2d[nz-2, i_range_slice]

    grad_xi_x = 0.5 * (bdry_xi_plus_x - bdry_xi_minus_x)
    grad_xi_z = 0.5 * (bdry_xi_plus_z - bdry_xi_minus_z)
    
    grad_xixi_x = bdry_xi_plus_x + bdry_xi_minus_x - 2 * bdry_point_x
    grad_xixi_z = bdry_xi_plus_z + bdry_xi_minus_z - 2 * bdry_point_z
    
    grad_zt_x = bdry_point_x - inner_point_x
    grad_zt_z = bdry_point_z - inner_point_z
    
    grad_ztzt_x = p_z2[i_range_slice, 0] + inner_point_x - 2 * bdry_point_x
    grad_ztzt_z = p_z2[i_range_slice, 1] + inner_point_z - 2 * bdry_point_z
    
    g11 = grad_xi_x**2 + grad_xi_z**2 + EPS
    
    term1_P = (grad_xi_x * grad_xixi_x + grad_xi_z * grad_xixi_z) / g11
    term2_P = (grad_xi_x * grad_ztzt_x + grad_xi_z * grad_ztzt_z) / (g22_z2[i_range_slice] + EPS)
    P_z2_loc[gni_range] = -(term1_P + term2_P)
    
    term1_Q = (grad_zt_x * grad_xixi_x + grad_zt_z * grad_xixi_z) / g11
    term2_Q = (grad_zt_x * grad_ztzt_x + grad_zt_z * grad_ztzt_z) / (g22_z2[i_range_slice] + EPS)
    Q_z2_loc[gni_range] = -(term1_Q + term2_Q)


def set_src_diri(x2d: np.ndarray, z2d: np.ndarray, 
                 p_x1: np.ndarray, p_x2: np.ndarray, 
                 p_z1: np.ndarray, p_z2: np.ndarray,
                 g11_x1: np.ndarray, g11_x2: np.ndarray, 
                 g22_z1: np.ndarray, g22_z2: np.ndarray,
                 gdcurv, src, mympi) -> None:
    """
    Set Dirichlet boundary source terms with vectorized calculation
    """
    nx = gdcurv.nx
    nz = gdcurv.nz
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    
    topocomm = mympi.topocomm
    neighid = mympi.neighid
    
    EPS = 1e-12

    # -------------------------- Left Boundary (x1, i=0) Calculation --------------------------
    if neighid[0] == MPI.PROC_NULL:
        calc_x1_diri(x2d, z2d, p_x1, g11_x1, gnk1, nx, nz, EPS, src.P_x1_loc, src.Q_x1_loc)

    # -------------------------- Right Boundary (x2, i=nx-1) Calculation --------------------------
    if neighid[1] == MPI.PROC_NULL:
        calc_x2_diri(x2d, z2d, p_x2, g11_x2, gnk1, nx, nz, EPS, src.P_x2_loc, src.Q_x2_loc)

    # -------------------------- Bottom Boundary (z1, k=0) Calculation --------------------------
    if neighid[2] == MPI.PROC_NULL:
        calc_z1_diri(x2d, z2d, p_z1, g22_z1, gni1, nx, nz, EPS, src.P_z1_loc, src.Q_z1_loc)

    # -------------------------- Top Boundary (z2, k=nz-1) Calculation --------------------------
    if neighid[3] == MPI.PROC_NULL:
        calc_z2_diri(x2d, z2d, p_z2, g22_z2, gni1, nx, nz, EPS, src.P_z2_loc, src.Q_z2_loc)

    # -------------------------- MPI Allreduce (Sum Operation) --------------------------
    # Keep original MPI communication logic (critical for parallel execution)
    topocomm.Allreduce(src.P_x1_loc, src.P_x1, op=MPI.SUM)
    topocomm.Allreduce(src.Q_x1_loc, src.Q_x1, op=MPI.SUM)
    topocomm.Allreduce(src.P_x2_loc, src.P_x2, op=MPI.SUM)
    topocomm.Allreduce(src.Q_x2_loc, src.Q_x2, op=MPI.SUM)
    topocomm.Allreduce(src.P_z1_loc, src.P_z1, op=MPI.SUM)
    topocomm.Allreduce(src.Q_z1_loc, src.Q_z1, op=MPI.SUM)
    topocomm.Allreduce(src.P_z2_loc, src.P_z2, op=MPI.SUM)
    topocomm.Allreduce(src.Q_z2_loc, src.Q_z2, op=MPI.SUM)