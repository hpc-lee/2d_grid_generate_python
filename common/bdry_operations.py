import numpy as np


def extend_abs_layer(bdry: np.ndarray, dh: float,
                     npoints: int, num_pml: int) -> np.ndarray:
    coef = 0.7
    for i in range(num_pml-1, -1, -1):
        dz = bdry[i + 2, 1] - bdry[i + 1, 1]
        slope = coef * dz / dh

        bdry[i, 0] = bdry[i + 1, 0] - dh
        bdry[i, 1] = bdry[i + 1, 1] - slope * dh

    for i in range(npoints - num_pml, npoints):
        dz = bdry[i - 1, 1] - bdry[i - 2, 1]
        slope = coef * dz / dh

        bdry[i, 0] = bdry[i - 1, 0] + dh
        bdry[i, 1] = bdry[i - 1, 1] + slope * dh

    return bdry


def arc_strech(A: float, bdry: np.ndarray) -> np.ndarray:
    npoints = bdry.shape[0]
    x_0 = bdry[:, 0]
    z_0 = bdry[:, 1]

    s = np.zeros(npoints)
    for i in range(1, npoints):
        ds = np.sqrt((x_0[i] - x_0[i-1])**2 + (z_0[i] - z_0[i-1])**2)
        s[i] = s[i-1] + ds

    u = np.zeros(npoints)
    for i in range(1, npoints):
        u[i] = s[i] / s[-1]

    for i in range(1, npoints - 1):
        xi = i / (npoints - 1)
        r = (np.exp(A * xi) - 1) / (np.exp(A) - 1)

        n = 0
        for m in range(npoints - 1):
            if r >= u[m] and r < u[m + 1]:
                n = m
                break

        bdry[i, 0] = (x_0[n] + (x_0[n + 1] - x_0[n]) * (r - u[n])
                      / (u[n + 1] - u[n]))
        bdry[i, 1] = (z_0[n] + (z_0[n + 1] - z_0[n]) * (r - u[n])
                      / (u[n + 1] - u[n]))

    return bdry