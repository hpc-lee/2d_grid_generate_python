import numpy as np


class SimpleGridData:
    """Grid data container for single-partition (non-MPI) grid generation methods."""

    def __init__(self, nx: int = 0, nz: int = 0):
        self.nx = nx
        self.nz = nz
        self.ni = nx
        self.nk = nz

        self.x2d = np.zeros((nz, nx), dtype=np.float32)
        self.z2d = np.zeros((nz, nx), dtype=np.float32)

        # File output in MPI-partitioned re-export
        self.fname_part = "px0_pz0"
        # Global index of first physical points (for unified I/O)
        self.gni1 = 0
        self.gnk1 = 0

        # Step lengths for marching methods (parabolic/hyperbolic only)
        self.step = np.zeros(max(nz - 1, 0), dtype=np.float32)
