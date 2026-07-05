import numpy as np
import sys
from pathlib import Path
from grid_data import GridData
from common.grid_math import flip_coord_z, flip_step_z

# C++ accelerated module (pybind11), located at src_cpp/gridcpp.so
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_cpp"))
try:
    import gridcpp
except ImportError:
    gridcpp = None


def assign_bdry_coords(x2d: np.ndarray, z2d: np.ndarray,
                       nx: int, k: int) -> None:
    # Geometric symmetry boundary
    x2d[k, 0] = 2 * x2d[k, 1] - x2d[k, 2]
    z2d[k, 0] = 2 * z2d[k, 1] - z2d[k, 2]

    x2d[k, nx-1] = 2 * x2d[k, nx-2] - x2d[k, nx-3]
    z2d[k, nx-1] = 2 * z2d[k, nx-2] - z2d[k, nx-3]


def para_gene_cpp(gdcurv: GridData, cfgs: dict) -> None:
    """C++ accelerated grid generation: predict_point/update_point via gridcpp(C++),
    assign_bdry_coords shared pure-Python version. Python keeps for-k dispatch loop."""
    if gridcpp is None:
        raise RuntimeError("gridcpp module not loaded, build src_cpp/gridcpp.so first")
    nx = gdcurv.nx
    nz = gdcurv.nz
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    step = gdcurv.step
    t2b = cfgs['t2b']
    coef = cfgs['coef']
    if t2b == 1:
        flip_coord_z(x2d, z2d)
        flip_step_z(step)

    var_th = np.zeros((nx - 2, 7), dtype=np.float32)
    x_pre = np.zeros((nx, 2), dtype=np.float32)
    z_pre = np.zeros((nx, 2), dtype=np.float32)
    step_len = np.zeros(nz, dtype=np.float32)
    for k in range(1, nz):
        step_len[k] = step_len[k-1] + step[k-1]

    for k in range(1, nz - 1):
        if k % 20 == 0 or k == nz - 2:
            print(f"layer {k}/{nz-2}")
        gridcpp.predict_point_cpp(x2d, z2d, nx, nz, k, t2b, coef,
                                  step_len, x_pre, z_pre)
        gridcpp.update_point_cpp(x2d, z2d, var_th, nx, k, x_pre, z_pre)
        assign_bdry_coords(x2d, z2d, nx, k)

    if t2b == 1:
        flip_coord_z(x2d, z2d)
