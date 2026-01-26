import numpy as np
from mpi4py import MPI
import numba
from mympi import grid_comm_optimized
from grid_utils import update_SOR, interp_inner_source
from grid_utils import compute_residual, source


def higen_gene(gdcurv: 'GridData', cfgs: dict, 
               mympi: 'MPIclass', lib=None):
    """
    Dirichlet boundary condition grid generation
    """
    err_threshold = cfgs['method']["higenstock"]['iter_err']
    max_iter = int(cfgs['method']["higenstock"]['max_iter'])
    coef = cfgs['method']["higenstock"]['coef']
    weight = cfgs['method']["higenstock"]['weight']
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
    
    dx1 = np.zeros(nz, dtype=np.float32)
    dx2 = np.zeros(nz, dtype=np.float32)
    dz1 = np.zeros(nx, dtype=np.float32)
    dz2 = np.zeros(nx, dtype=np.float32)

    dist_cal(x2d, z2d, nx, nz, dx1, dx2, dz1, dz2, neighid)

    set_src_higen(x2d, z2d, gdcurv, src, dx1, dx2,
                  dz1, dz2, mympi)

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
        set_src_higen(x2d, z2d, gdcurv, src, dx1, dx2,
                      dz1, dz2, mympi)
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


def dist_cal(x2d: np.ndarray, z2d: np.ndarray, nx: int, nz: int,
             dx1: np.ndarray, dx2: np.ndarray, dz1: np.ndarray, dz2: np.ndarray,
             neighid: np.ndarray) -> None:
    """
    Calculate the distance projection values for the four boundaries (x1/x2/z1/z2) of the grid
    """
    EPS = 1e-12

    if neighid[0] == MPI.PROC_NULL:
        k_range = slice(1, nz-1)
        
        x_xi0 = x2d[k_range, 1] - x2d[k_range, 0]
        z_xi0 = z2d[k_range, 1] - z2d[k_range, 0]
        
        x_zt = 0.5 * (x2d[2:, 0] - x2d[:-2, 0])  
        z_zt = 0.5 * (z2d[2:, 0] - z2d[:-2, 0])
        
        vn_xi = z_zt
        vn_zt = -x_zt
        len_vn = np.hypot(vn_xi, vn_zt) + EPS  
        vn_xi0 = vn_xi / len_vn
        vn_zt0 = vn_zt / len_vn
        
        dx1[k_range] = x_xi0 * vn_xi0 + z_xi0 * vn_zt0

    if neighid[1] == MPI.PROC_NULL:
        k_range = slice(1, nz-1)
        
        x_xi0 = x2d[k_range, nx-1] - x2d[k_range, nx-2]
        z_xi0 = z2d[k_range, nx-1] - z2d[k_range, nx-2]
        
        x_zt = 0.5 * (x2d[2:, nx-1] - x2d[:-2, nx-1])
        z_zt = 0.5 * (z2d[2:, nx-1] - z2d[:-2, nx-1])
        
        vn_xi = z_zt
        vn_zt = -x_zt
        len_vn = np.hypot(vn_xi, vn_zt) + EPS
        vn_xi0 = vn_xi / len_vn
        vn_zt0 = vn_zt / len_vn
        
        dx2[k_range] = x_xi0 * vn_xi0 + z_xi0 * vn_zt0

    if neighid[2] == MPI.PROC_NULL:
        i_range = slice(1, nx-1)
        
        x_zt0 = x2d[1, i_range] - x2d[0, i_range]
        z_zt0 = z2d[1, i_range] - z2d[0, i_range]
        
        x_xi = 0.5 * (x2d[0, 2:] - x2d[0, :-2])
        z_xi = 0.5 * (z2d[0, 2:] - z2d[0, :-2])
        
        vn_xi = -z_xi
        vn_zt = x_xi
        len_vn = np.hypot(vn_xi, vn_zt) + EPS
        vn_xi0 = vn_xi / len_vn
        vn_zt0 = vn_zt / len_vn
        
        dz1[i_range] = x_zt0 * vn_xi0 + z_zt0 * vn_zt0

    if neighid[3] == MPI.PROC_NULL:
        i_range = slice(1, nx-1)
        
        x_zt0 = x2d[nz-1, i_range] - x2d[nz-2, i_range]
        z_zt0 = z2d[nz-1, i_range] - z2d[nz-2, i_range]
        
        x_xi = 0.5 * (x2d[nz-1, 2:] - x2d[nz-1, :-2])
        z_xi = 0.5 * (z2d[nz-1, 2:] - z2d[nz-1, :-2])
        
        vn_xi = -z_xi
        vn_zt = x_xi
        len_vn = np.hypot(vn_xi, vn_zt) + EPS
        vn_xi0 = vn_xi / len_vn
        vn_zt0 = vn_zt / len_vn
        
        dz2[i_range] = x_zt0 * vn_xi0 + z_zt0 * vn_zt0


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_x1_boundary(x2d, z2d, dx1, gnk1, nx, nz, a, theta0, EPS, Q_x1_loc, P_x1_loc):
    k_start = 1
    k_end = nz - 1
    gnk_range = gnk1 + np.arange(nz - 2)
    
    bdry_point_x = x2d[k_start:k_end, 0]
    bdry_point_z = z2d[k_start:k_end, 0]
    bdry_zt_plus_x = x2d[2:nz, 0]
    bdry_zt_plus_z = z2d[2:nz, 0]
    bdry_zt_minus_x = x2d[0:nz-2, 0]
    bdry_zt_minus_z = z2d[0:nz-2, 0]
    inner_point_x = x2d[k_start:k_end, 1]
    inner_point_z = z2d[k_start:k_end, 1]

    x_zt = 0.5 * (bdry_zt_plus_x - bdry_zt_minus_x)
    z_zt = 0.5 * (bdry_zt_plus_z - bdry_zt_minus_z)
    x_xi = inner_point_x - bdry_point_x
    z_xi = inner_point_z - bdry_point_z
    
    dot = x_xi * x_zt + z_xi * z_zt
    len_xi = np.hypot(x_xi, z_xi) + EPS
    len_zt = np.hypot(x_zt, z_zt) + EPS
    cos_theta = dot / (len_xi * len_zt)
    
    theta = np.arccos(cos_theta)
    dif_theta = (theta0 - theta) / theta0
    dif_dis = (dx1[k_start:k_end] - len_xi) / (dx1[k_start:k_end] + EPS)
    
    Q_x1_loc[gnk_range] -= a * np.tanh(dif_theta)
    P_x1_loc[gnk_range] += a * np.tanh(dif_dis)


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_x2_boundary(x2d, z2d, dx2, gnk1, nx, nz, a, theta0, EPS, Q_x2_loc, P_x2_loc):
    k_start = 1
    k_end = nz - 1
    gnk_range = gnk1 + np.arange(nz - 2)
    
    bdry_point_x = x2d[k_start:k_end, nx-1]
    bdry_point_z = z2d[k_start:k_end, nx-1]
    bdry_zt_plus_x = x2d[2:nz, nx-1]
    bdry_zt_plus_z = z2d[2:nz, nx-1]
    bdry_zt_minus_x = x2d[0:nz-2, nx-1]
    bdry_zt_minus_z = z2d[0:nz-2, nx-1]
    inner_point_x = x2d[k_start:k_end, nx-2]
    inner_point_z = z2d[k_start:k_end, nx-2]

    x_zt = 0.5 * (bdry_zt_plus_x - bdry_zt_minus_x)
    z_zt = 0.5 * (bdry_zt_plus_z - bdry_zt_minus_z)
    x_xi = bdry_point_x - inner_point_x
    z_xi = bdry_point_z - inner_point_z
    
    dot = x_xi * x_zt + z_xi * z_zt
    len_xi = np.hypot(x_xi, z_xi) + EPS
    len_zt = np.hypot(x_zt, z_zt) + EPS
    cos_theta = dot / (len_xi * len_zt)
    
    theta = np.arccos(cos_theta)
    dif_theta = (theta0 - theta) / theta0
    dif_dis = (dx2[k_start:k_end] - len_xi) / (dx2[k_start:k_end] + EPS)
    
    Q_x2_loc[gnk_range] += a * np.tanh(dif_theta)
    P_x2_loc[gnk_range] -= a * np.tanh(dif_dis)


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_z1_boundary(x2d, z2d, dz1, gni1, nx, nz, a, theta0, EPS, P_z1_loc, Q_z1_loc):
    i_start = 1
    i_end = nx - 1
    gni_range = gni1 + np.arange(nx - 2)
    
    bdry_point_x = x2d[0, i_start:i_end]
    bdry_point_z = z2d[0, i_start:i_end]
    bdry_xi_plus_x = x2d[0, 2:nx]
    bdry_xi_plus_z = z2d[0, 2:nx]
    bdry_xi_minus_x = x2d[0, 0:nx-2]
    bdry_xi_minus_z = z2d[0, 0:nx-2]
    inner_point_x = x2d[1, i_start:i_end]
    inner_point_z = z2d[1, i_start:i_end]

    x_xi = 0.5 * (bdry_xi_plus_x - bdry_xi_minus_x)
    z_xi = 0.5 * (bdry_xi_plus_z - bdry_xi_minus_z)
    x_zt = inner_point_x - bdry_point_x
    z_zt = inner_point_z - bdry_point_z
    
    dot = x_xi * x_zt + z_xi * z_zt
    len_xi = np.hypot(x_xi, z_xi) + EPS
    len_zt = np.hypot(x_zt, z_zt) + EPS
    cos_theta = dot / (len_xi * len_zt)
    
    theta = np.arccos(cos_theta)
    dif_theta = (theta0 - theta) / theta0
    dif_dis = (dz1[i_start:i_end] - len_zt) / (dz1[i_start:i_end] + EPS)
    
    P_z1_loc[gni_range] -= a * np.tanh(dif_theta)
    Q_z1_loc[gni_range] += a * np.tanh(dif_dis)


@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def calc_z2_boundary(x2d, z2d, dz2, gni1, nx, nz, a, theta0, EPS, P_z2_loc, Q_z2_loc):
    i_start = 1
    i_end = nx - 1
    gni_range = gni1 + np.arange(nx - 2)
    
    bdry_point_x = x2d[nz-1, i_start:i_end]
    bdry_point_z = z2d[nz-1, i_start:i_end]
    bdry_xi_plus_x = x2d[nz-1, 2:nx]
    bdry_xi_plus_z = z2d[nz-1, 2:nx]
    bdry_xi_minus_x = x2d[nz-1, 0:nx-2]
    bdry_xi_minus_z = z2d[nz-1, 0:nx-2]
    inner_point_x = x2d[nz-2, i_start:i_end]
    inner_point_z = z2d[nz-2, i_start:i_end]

    x_xi = 0.5 * (bdry_xi_plus_x - bdry_xi_minus_x)
    z_xi = 0.5 * (bdry_xi_plus_z - bdry_xi_minus_z)
    x_zt = bdry_point_x - inner_point_x
    z_zt = bdry_point_z - inner_point_z
    
    dot = x_xi * x_zt + z_xi * z_zt
    len_xi = np.hypot(x_xi, z_xi) + EPS
    len_zt = np.hypot(x_zt, z_zt) + EPS
    cos_theta = dot / (len_xi * len_zt)
    
    theta = np.arccos(cos_theta)
    dif_theta = (theta0 - theta) / theta0
    dif_dis = (dz2[i_start:i_end] - len_zt) / (dz2[i_start:i_end] + EPS)
    
    P_z2_loc[gni_range] += a * np.tanh(dif_theta)
    Q_z2_loc[gni_range] -= a * np.tanh(dif_dis)


def set_src_higen(x2d: np.ndarray, z2d: np.ndarray, 
                  gdcurv, src, dx1: np.ndarray, dx2: np.ndarray,
                  dz1: np.ndarray, dz2: np.ndarray, mympi) -> None:
    """
    Set high-order boundary source terms (higen) with vectorized calculation
    Calculate and update source terms for four grid boundaries (x1/x2/z1/z2)
    """
    nx = gdcurv.nx
    nz = gdcurv.nz
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    
    topocomm = mympi.topocomm
    neighid = mympi.neighid
    
    theta0 = np.pi / 2
    a = 0.1  # 
    EPS = 1e-12  

    # -------------------------- Left Boundary (x1, xi=0) Calculation --------------------------
    if neighid[0] == MPI.PROC_NULL:
        calc_x1_boundary(x2d, z2d, dx1, gnk1, nx, nz, a, theta0, EPS, src.Q_x1_loc, src.P_x1_loc)

    # -------------------------- Right Boundary (x2, xi=1) Calculation --------------------------
    if neighid[1] == MPI.PROC_NULL:
        calc_x2_boundary(x2d, z2d, dx2, gnk1, nx, nz, a, theta0, EPS, src.Q_x2_loc, src.P_x2_loc)

    # -------------------------- Bottom Boundary (z1, zt=0) Calculation --------------------------
    if neighid[2] == MPI.PROC_NULL:
        calc_z1_boundary(x2d, z2d, dz1, gni1, nx, nz, a, theta0, EPS, src.P_z1_loc, src.Q_z1_loc)

    # -------------------------- Top Boundary (z2, zt=1) Calculation --------------------------
    if neighid[3] == MPI.PROC_NULL:
        calc_z2_boundary(x2d, z2d, dz2, gni1, nx, nz, a, theta0, EPS, src.P_z2_loc, src.Q_z2_loc)

    # -------------------------- MPI Communication --------------------------
    # Keep original MPI communication logic (critical for parallel execution)
    topocomm.Allreduce(src.P_x1_loc, src.P_x1, op=MPI.SUM)
    topocomm.Allreduce(src.Q_x1_loc, src.Q_x1, op=MPI.SUM)
    topocomm.Allreduce(src.P_x2_loc, src.P_x2, op=MPI.SUM)
    topocomm.Allreduce(src.Q_x2_loc, src.Q_x2, op=MPI.SUM)
    topocomm.Allreduce(src.P_z1_loc, src.P_z1, op=MPI.SUM)
    topocomm.Allreduce(src.Q_z1_loc, src.Q_z1, op=MPI.SUM)
    topocomm.Allreduce(src.P_z2_loc, src.P_z2, op=MPI.SUM)
    topocomm.Allreduce(src.Q_z2_loc, src.Q_z2, op=MPI.SUM)