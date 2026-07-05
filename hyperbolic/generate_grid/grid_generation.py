import numpy as np
import sys
from pathlib import Path
from grid_data import GridData
from common.grid_math import flip_coord_z
from common.algebra import zt_arc_stretch

# C++ accelerated module (pybind11), located at src_cpp/gridcpp.so
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_cpp"))
try:
    import gridcpp
except ImportError:
    gridcpp = None


def hyper_gene_cpp(gdcurv: GridData, cfgs: dict) -> None:
    """C++ accelerated: cal_smooth_coef/cal_matrix/modify_matrix/thomas_block via
    gridcpp(C++), update_coords shared pure-Python version. Python keeps for-k dispatch loop."""
    if gridcpp is None:
        raise RuntimeError("gridcpp module not loaded, build src_cpp/gridcpp.so first")
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
    area = np.zeros((nx, 2), dtype=np.float32)
    a = np.zeros((n, 2, 2), dtype=np.float32)
    b = np.zeros((n, 2, 2), dtype=np.float32)
    c = np.zeros((n, 2, 2), dtype=np.float32)
    d = np.zeros((n, 2), dtype=np.float32)
    xz = np.zeros((n, 2), dtype=np.float32)
    D = np.zeros((n, 2, 2), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)

    # k=1 preparation layer
    gridcpp.cal_matrix_cpp(x2d, z2d, nx, 1, step, a, b, c, d, area)
    gridcpp.modify_matrix_cpp(x2d, z2d, nx, 1, a, b, c, d, coef_e)
    gridcpp.thomas_block_cpp(n, a, b, c, d, xz, D, y)
    update_coords(xz, x2d, z2d, nx, 1)

    for k in range(1, nz):
        if k % 20 == 0 or k == nz - 1:
            print(f"layer {k}/{nz-1}")
        gridcpp.cal_smooth_coef_cpp(coef, x2d, z2d, nx, nz, k, t2b, coef_e)
        gridcpp.cal_matrix_cpp(x2d, z2d, nx, k, step, a, b, c, d, area)
        gridcpp.modify_matrix_cpp(x2d, z2d, nx, k, a, b, c, d, coef_e)
        gridcpp.thomas_block_cpp(n, a, b, c, d, xz, D, y)
        update_coords(xz, x2d, z2d, nx, k)

    if t2b == 1:
        flip_coord_z(gdcurv.x2d, gdcurv.z2d)

    if flag_stretch == 1:
        zt_arc_stretch(gdcurv)


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
