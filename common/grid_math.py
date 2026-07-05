import numpy as np


def flip_coord_z(x2d: np.ndarray, z2d: np.ndarray) -> None:
    """Flip z direction"""
    x2d[:] = x2d[::-1, :]
    z2d[:] = z2d[::-1, :]


def flip_step_z(step: np.ndarray) -> None:
    """Flip step array in z direction"""
    step[:] = step[::-1]
