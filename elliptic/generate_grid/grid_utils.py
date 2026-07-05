import numpy as np


def linear_tfi(gdcurv, bdry) -> None:
    """
    Perform linear TFI (Transfinite Interpolation) to initialize the full grid
    including ghost cells. Ghosts are filled directly by TFI (internal ghosts
    equal the neighbor's physical boundary; external ghosts equal the global
    boundary via corner consistency), so no MPI boundary exchange is needed
    afterwards.

    Parameters:
        gdcurv: GD instance with x2d, z2d allocated as (nz, nx) arrays
        bdry: Bdry instance with bx1, bx2 (nz_all, 2), bz1, bz2 (nx_all, 2)
    """
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

    # Generate full grid including ghost cells: TFI fills ghosts directly
    # (internal ghosts = neighbor physical boundary, external ghosts = global
    # boundary), so no MPI boundary exchange is needed afterwards.
    k_local = np.arange(0, nz, dtype=np.int32)
    i_local = np.arange(0, nx, dtype=np.int32)
    k_mesh, i_mesh = np.meshgrid(k_local, i_local, indexing='ij')

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

    # TFI core formula (U + W - UW) — fills the entire grid including ghost
    # cells. External ghosts (g_i=0 or nx_all-1, etc.) reduce to the global
    # boundary values via corner consistency; internal ghosts equal the
    # neighbor's physical boundary (same global point, same formula).
    x2d[:, :] = (U_x - UW_x + W_x)
    z2d[:, :] = (U_z - UW_z + W_z)


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
