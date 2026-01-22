from mpi4py import MPI
import numpy as np
import numba


def linear_tfi(gdcurv, bdry, mympi) -> None:
    """
    Perform linear TFI (Transfinite Interpolation) to initialize interior grid points,
    and assign boundary values to ghost cells if on domain boundary.
    
    Parameters:
        gdcurv: GD instance with x2d, z2d allocated as (nz, nx) arrays
        bdry: Bdry instance with bx1, bx2 (nz_all, 2), bz1, bz2 (nx_all, 2)
        mympi: MyMPI instance with neighid = [left, right, down, up]
    """
      # Local grid indices (interior, 1-based)
    ni1, ni2 = gdcurv.ni1, gdcurv.ni2  # x interior range
    nk1, nk2 = gdcurv.nk1, gdcurv.nk2  # z interior range
    
    # Local grid size (including ghost cells)
    nx, nz = gdcurv.nx, gdcurv.nz
    
    # Global start indices (interior, no ghost)
    gni1 = gdcurv.gni1  # global x start for current rank
    gnk1 = gdcurv.gnk1  # global z start for current rank
    
    # Grid arrays (reference to avoid copy)
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d

    nx_all = bdry.nx_all  # total global x points
    nz_all = bdry.nz_all  # total global z points
    
    # Boundary coordinates: [:,0] = x, [:,1] = z
    bx1 = bdry.bx1  # left x-boundary (nz_all, 2)
    bx2 = bdry.bx2  # right x-boundary (nz_all, 2)
    bz1 = bdry.bz1  # bottom z-boundary (nx_all, 2)
    bz2 = bdry.bz2  # top z-boundary (nx_all, 2)

    # Create meshgrid for local interior indices
    # Local interior indices (k: z, i: x)
    k_local = np.arange(nk1, nk2 + 1, dtype=np.int32)  # shape [nz_interior]
    i_local = np.arange(ni1, ni2 + 1, dtype=np.int32)  # shape [nx_interior]
    k_mesh, i_mesh = np.meshgrid(k_local, i_local, indexing='ij')  # [nz_interior, nx_interior]

    # Compute global indices (vectorized)
    g_i = gni1 + i_mesh  # global x indices for interpolation
    g_k = gnk1 + k_mesh  # global z indices for interpolation

    # Normalized coordinates (0~1, vectorized)
    xi = g_i / (nx_all - 1)  
    zt = g_k / (nz_all - 1)  

    # Precompute interpolation weights (vectorized)
    a0 = 1.0 - xi  # weight for left boundary (bx1)
    a1 = xi        # weight for right boundary (bx2)
    c0 = 1.0 - zt  # weight for bottom boundary (bz1)
    c1 = zt        # weight for top boundary (bz2)

    # U term (interpolation along x-boundaries bx1/bx2)
    # Bound check g_k to avoid out-of-bounds (vectorized mask)
    g_k_mask = (g_k >= 0) & (g_k < nz_all)
    U_x = np.zeros_like(g_i, dtype=np.float32)  
    U_z = np.zeros_like(g_i, dtype=np.float32)
    U_x[g_k_mask] = a0[g_k_mask] * bx1[g_k[g_k_mask], 0] + a1[g_k_mask] * bx2[g_k[g_k_mask], 0]
    U_z[g_k_mask] = a0[g_k_mask] * bx1[g_k[g_k_mask], 1] + a1[g_k_mask] * bx2[g_k[g_k_mask], 1]

    # W term (interpolation along z-boundaries bz1/bz2)
    # Bound check g_i to avoid out-of-bounds (vectorized mask)
    g_i_mask = (g_i >= 0) & (g_i < nx_all)
    W_x = np.zeros_like(g_i, dtype=np.float32) 
    W_z = np.zeros_like(g_i, dtype=np.float32)
    W_x[g_i_mask] = c0[g_i_mask] * bz1[g_i[g_i_mask], 0] + c1[g_i_mask] * bz2[g_i[g_i_mask], 0]
    W_z[g_i_mask] = c0[g_i_mask] * bz1[g_i[g_i_mask], 1] + c1[g_i_mask] * bz2[g_i[g_i_mask], 1]

    # UW term (bilinear interpolation of four corners - CONSTANT, precompute once)
    # Corners: (x=0,z=0), (x=0,z=nz_all-1), (x=nx_all-1,z=0), (x=nx_all-1,z=nz_all-1)
    UW_x = (
        a0 * c0 * bx1[0, 0] +
        a0 * c1 * bx1[-1, 0] +
        a1 * c0 * bx2[0, 0] +
        a1 * c1 * bx2[-1, 0]
    )
    UW_z = (
        a0 * c0 * bx1[0, 1] +
        a0 * c1 * bx1[-1, 1] +
        a1 * c0 * bx2[0, 1] +
        a1 * c1 * bx2[-1, 1]
    )

    # TFI core formula (U + W - UW)
    x2d[nk1:nk2+1, ni1:ni2+1] = (U_x - UW_x + W_x)
    z2d[nk1:nk2+1, ni1:ni2+1] = (U_z - UW_z + W_z)

    # --------------------------
    # 4. Assign boundary values to ghost cells (vectorized)
    # --------------------------
    neigh = mympi.neighid  # [left, right, down, up]

    # Left boundary (x=0 ghost cell) → use bx1
    if neigh[0] == MPI.PROC_NULL:
        # Create global z indices for all local z (including ghosts)
        g_k_ghost = gnk1 + np.arange(nz, dtype=np.int32)
        g_k_ghost_mask = (g_k_ghost >= 0) & (g_k_ghost < nz_all)
        # Assign to left ghost column (x=0) with bounds check (float32)
        x2d[g_k_ghost_mask, 0] = bx1[g_k_ghost[g_k_ghost_mask], 0]
        z2d[g_k_ghost_mask, 0] = bx1[g_k_ghost[g_k_ghost_mask], 1]

    # Right boundary (x=nx_local-1 ghost cell) → use bx2
    if neigh[1] == MPI.PROC_NULL:
        g_k_ghost = gnk1 + np.arange(nz, dtype=np.int32)
        g_k_ghost_mask = (g_k_ghost >= 0) & (g_k_ghost < nz_all)
        x2d[g_k_ghost_mask, nx - 1] = bx2[g_k_ghost[g_k_ghost_mask], 0]
        z2d[g_k_ghost_mask, nx - 1] = bx2[g_k_ghost[g_k_ghost_mask], 1]

    # Bottom boundary (z=0 ghost cell) → use bz1
    if neigh[2] == MPI.PROC_NULL:
        g_i_ghost = gni1 + np.arange(nx, dtype=np.int32)
        g_i_ghost_mask = (g_i_ghost >= 0) & (g_i_ghost < nx_all)
        x2d[0, g_i_ghost_mask] = bz1[g_i_ghost[g_i_ghost_mask], 0]
        z2d[0, g_i_ghost_mask] = bz1[g_i_ghost[g_i_ghost_mask], 1]

    # Top boundary (z=nz_local-1 ghost cell) → use bz2
    if neigh[3] == MPI.PROC_NULL:
        g_i_ghost = gni1 + np.arange(nx, dtype=np.int32)
        g_i_ghost_mask = (g_i_ghost >= 0) & (g_i_ghost < nx_all)
        x2d[nz - 1, g_i_ghost_mask] = bz2[g_i_ghost[g_i_ghost_mask], 0]
        z2d[nz - 1, g_i_ghost_mask] = bz2[g_i_ghost[g_i_ghost_mask], 1]

@numba.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def update_SOR(x2d: np.ndarray, z2d: np.ndarray, 
               x2d_tmp: np.ndarray, z2d_tmp: np.ndarray, 
               nx: int, nz: int, P: np.ndarray, 
               Q: np.ndarray, omega: float) -> None:
    """
    Gauss-Seidel (G-S) Iteration Notes:
    1. Core logic can only be implemented
     with for loops due to point-by-point sequential dependencies
    2. Accelerate via Numba JIT compilation to balance numerical consistency 
    """

    for k in range(1, nz - 1):
        for i in range(1, nx - 1):
            x_xi = 0.5 * (x2d[k, i+1] - x2d_tmp[k, i-1])
            z_xi = 0.5 * (z2d[k, i+1] - z2d_tmp[k, i-1])

            x_zt = 0.5 * (x2d[k+1, i] - x2d_tmp[k-1, i])
            z_zt = 0.5 * (z2d[k+1, i] - z2d_tmp[k-1, i])

            x_xizt = 0.25 * (
                x2d[k+1, i+1] + x2d_tmp[k-1, i-1] - 
                x2d_tmp[k-1, i+1] - x2d[k+1, i-1]
            )
            z_xizt = 0.25 * (
                z2d[k+1, i+1] + z2d_tmp[k-1, i-1] - 
                z2d_tmp[k-1, i+1] - z2d[k+1, i-1]
            )

            g11 = x_xi * x_xi + z_xi * z_xi  
            g22 = x_zt * x_zt + z_zt * z_zt  
            g12 = x_xi * x_zt + z_xi * z_zt  

            denom = g22 + g11
            coef = 0.5 / denom if denom != 0 else 0.0  

            x2d_tmp[k, i] = coef * (
                g22 * (x2d[k, i+1] + x2d_tmp[k, i-1]) + 
                g11 * (x2d[k+1, i] + x2d_tmp[k-1, i]) - 
                2 * g12 * x_xizt + 
                g22 * P[k, i] * x_xi + 
                g11 * Q[k, i] * x_zt
            )
            x2d_tmp[k, i] = omega * x2d_tmp[k, i] + (1 - omega) * x2d[k, i]

            z2d_tmp[k, i] = coef * (
                g22 * (z2d[k, i+1] + z2d_tmp[k, i-1]) + 
                g11 * (z2d[k+1, i] + z2d_tmp[k-1, i]) - 
                2 * g12 * z_xizt + 
                g22 * P[k, i] * z_xi + 
                g11 * Q[k, i] * z_zt
            )
            z2d_tmp[k, i] = omega * z2d_tmp[k, i] + (1 - omega) * z2d[k, i]


class source:
    def __init__(self):
        self.P_x1_loc = None
        self.Q_x1_loc = None
        self.P_x2_loc = None
        self.Q_x2_loc = None
        self.P_z1_loc = None
        self.Q_z1_loc = None
        self.P_z2_loc = None
        self.Q_z2_loc = None
        self.P_x1 = None
        self.Q_x1 = None
        self.P_x2 = None
        self.Q_x2 = None
        self.P_z1 = None
        self.Q_z1 = None
        self.P_z2 = None
        self.Q_z2 = None
        self.P = None
        self.Q = None

    def init_src(self, gdcurv) -> None:
        nx = gdcurv.nx
        nz = gdcurv.nz
        total_nx = gdcurv.total_nx
        total_nz = gdcurv.total_nz

        # Exclude 2 boundary points (1D array: np.zeros directly)
        self.P_x1_loc = np.zeros(total_nz-2, dtype=np.float32)
        self.Q_x1_loc = np.zeros(total_nz-2, dtype=np.float32)
        self.P_x2_loc = np.zeros(total_nz-2, dtype=np.float32)
        self.Q_x2_loc = np.zeros(total_nz-2, dtype=np.float32)
        self.P_z1_loc = np.zeros(total_nx-2, dtype=np.float32)
        self.Q_z1_loc = np.zeros(total_nx-2, dtype=np.float32)
        self.P_z2_loc = np.zeros(total_nx-2, dtype=np.float32)
        self.Q_z2_loc = np.zeros(total_nx-2, dtype=np.float32)

        self.P_x1 = np.zeros(total_nz-2, dtype=np.float32)
        self.Q_x1 = np.zeros(total_nz-2, dtype=np.float32)
        self.P_x2 = np.zeros(total_nz-2, dtype=np.float32)
        self.Q_x2 = np.zeros(total_nz-2, dtype=np.float32)
        self.P_z1 = np.zeros(total_nx-2, dtype=np.float32)
        self.Q_z1 = np.zeros(total_nx-2, dtype=np.float32)
        self.P_z2 = np.zeros(total_nx-2, dtype=np.float32)
        self.Q_z2 = np.zeros(total_nx-2, dtype=np.float32)

        self.P = np.zeros((nz, nx), dtype=np.float32)
        self.Q = np.zeros((nz, nx), dtype=np.float32)


def interp_inner_source(P: np.ndarray, P_x1: np.ndarray, P_x2: np.ndarray, 
                        P_z1: np.ndarray, P_z2: np.ndarray, Q: np.ndarray, 
                        Q_x1: np.ndarray, Q_x2: np.ndarray, Q_z1: np.ndarray, 
                        Q_z2: np.ndarray, nx: int, nz: int, gni1: int, gnk1: int,
                        total_nx: int, total_nz: int,  coef: np.ndarray, 
                        weight: np.ndarray) -> None:
    """
    Interpolate inner source
    """
    # Valid range: k ∈ [1, nz-2], i ∈ [1, nx-2] (exclude boundary points)
    k_start = 1
    k_end = nz - 1  # exclusive → covers k=1 to k=nz-2
    i_start = 1
    i_end = nx - 1  # exclusive → covers i=1 to i=nx-2
    
    k_vals = np.arange(k_start, k_end, dtype=np.int32)
    i_vals = np.arange(i_start, i_end, dtype=np.int32)
    k_grid, i_grid = np.meshgrid(k_vals, i_vals, indexing='ij')  # critical: 'ij' for (k,i) order

    # Calculate gni (index for xi direction interpolation) → shape: (nz-2, nx-2)
    gni_xi = gni1 + i_grid
    # Normalized xi coordinate (0 ≤ xi ≤ 1)
    xi = gni_xi / (total_nx - 1)
    
    # Interpolation coefficients (vectorized)
    c0_xi = 1.0 - xi
    c1_xi = xi
    r0_xi = np.exp(-coef[0] * xi)
    r1_xi = np.exp(-coef[1] * (1.0 - xi))
    
    # Calculate gnk (index for x1/x2 boundary sources) → shape: (nz-2, nx-2)
    gnk_xi = gnk1 + k_grid - 1
    
    # Vectorized interpolation for P/Q (xi direction)
    # Shape match: src.P_x1[gnk_xi] → (nz-2, nx-2) (broadcast from 1D to 2D)
    P[k_start:k_end, i_start:i_end] = weight[0] * (
        c0_xi * P_x1[gnk_xi] + c1_xi * P_x2[gnk_xi]
    )
    Q[k_start:k_end, i_start:i_end] = weight[0] * (
        r0_xi * Q_x1[gnk_xi] + r1_xi * Q_x2[gnk_xi]
    )

    # Calculate gnk (index for zt direction interpolation) → shape: (nz-2, nx-2)
    gnk_zt = gnk1 + k_grid
    # Normalized zt coordinate (0 ≤ zt ≤ 1)
    zt = gnk_zt / (total_nz - 1)
    
    # Interpolation coefficients (vectorized)
    c0_zt = 1.0 - zt
    c1_zt = zt
    r0_zt = np.exp(-coef[2] * zt)
    r1_zt = np.exp(-coef[3] * (1.0 - zt))
    
    # Calculate gni (index for z1/z2 boundary sources) → shape: (nz-2, nx-2)
    gni_zt = gni1 + i_grid - 1
    # Ensure gni_zt is integer index
    gni_zt = gni_zt
    
    # Vectorized interpolation for P/Q (zt direction, accumulate to xi result)
    P[k_start:k_end, i_start:i_end] += weight[1] * (
        r0_zt * P_z1[gni_zt] + r1_zt * P_z2[gni_zt]
    )
    Q[k_start:k_end, i_start:i_end] += weight[1] * (
        c0_zt * Q_z1[gni_zt] + c1_zt * Q_z2[gni_zt]
    )



def compute_residual(x2d: np.ndarray, z2d: np.ndarray, x2d_tmp: np.ndarray, 
                     z2d_tmp: np.ndarray, local_max: np.ndarray, 
                     nx: int, nz: int) -> None:

    # 1. Calculate displacement magnitude between iterations
    internal_slice = (slice(1, nz-1), slice(1, nx-1)) 
    dx = x2d_tmp[internal_slice] - x2d[internal_slice]
    dz = z2d_tmp[internal_slice] - z2d[internal_slice]
    displacement = np.hypot(dx, dz)
    
    # 2. Calculate horizontal metric (i+1 - i)
    dx_h = x2d[internal_slice[0], internal_slice[1].start+1:internal_slice[1].stop+1] - x2d[internal_slice]
    dz_h = z2d[internal_slice[0], internal_slice[1].start+1:internal_slice[1].stop+1] - z2d[internal_slice]
    metric_h = np.hypot(dx_h, dz_h)
    metric_h = np.where(metric_h > 1e-12, metric_h, 1e-12)  # Avoid division by zero
    
    # 3. Calculate vertical metric (k+1 - k)
    dx_v = x2d[internal_slice[0].start+1:internal_slice[0].stop+1, internal_slice[1]] - x2d[internal_slice]
    dz_v = z2d[internal_slice[0].start+1:internal_slice[0].stop+1, internal_slice[1]] - z2d[internal_slice]
    metric_v = np.hypot(dx_v, dz_v)
    metric_v = np.where(metric_v > 1e-12, metric_v, 1e-12)  # Avoid division by zero
    
    # 4. Compute normalized residuals
    res_i = displacement / metric_h
    res_k = displacement / metric_v

    local_max[0] = np.max(res_i)
    local_max[1] = np.max(res_k)
    
