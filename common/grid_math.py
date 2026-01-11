import numba
import numpy as np


# 2x2 inverse
@numba.jit(nopython=True, nogil=True, cache=True)
def inv_2x2(mat):
    det = mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]
    det = det if abs(det) > 1e-10 else 1e-10
    inv = np.empty((2,2), dtype=mat.dtype)
    inv[0,0] =  mat[1,1] / det
    inv[0,1] = -mat[0,1] / det
    inv[1,0] = -mat[1,0] / det
    inv[1,1] =  mat[0,0] / det
    return inv

# 2x2 2x2
@numba.jit(nopython=True, nogil=True, cache=True)
def matmul_2x2(a_mat, b_mat):
    res = np.empty((2,2), dtype=a_mat.dtype)
    res[0,0] = a_mat[0,0]*b_mat[0,0] + a_mat[0,1]*b_mat[1,0]
    res[0,1] = a_mat[0,0]*b_mat[0,1] + a_mat[0,1]*b_mat[1,1]
    res[1,0] = a_mat[1,0]*b_mat[0,0] + a_mat[1,1]*b_mat[1,0]
    res[1,1] = a_mat[1,0]*b_mat[0,1] + a_mat[1,1]*b_mat[1,1]
    return res

# 2x2 2x1
@numba.jit(nopython=True, nogil=True, cache=True)
def matmul_2x2_vec(mat, vec):
    res = np.empty(2, dtype=mat.dtype)
    res[0] = mat[0,0]*vec[0] + mat[0,1]*vec[1]
    res[1] = mat[1,0]*vec[0] + mat[1,1]*vec[1]
    return res

@numba.jit(nopython=True, nogil=True, cache=True)
def inv_2x2_batch(mat_batch):
    """
    Batch inversion for 2x2 matrices (vectorized implementation).
    """
    n = mat_batch.shape[0]

    inv_batch = np.empty((n, 2, 2), dtype=mat_batch.dtype)
    
    det = mat_batch[:,0,0] * mat_batch[:,1,1] - mat_batch[:,0,1] * mat_batch[:,1,0]
    det = np.where(np.abs(det) > 1e-10, det, 1e-10)
    
    inv_batch[:,0,0] =  mat_batch[:,1,1] / det
    inv_batch[:,0,1] = -mat_batch[:,0,1] / det
    inv_batch[:,1,0] = -mat_batch[:,1,0] / det
    inv_batch[:,1,1] =  mat_batch[:,0,0] / det
    
    return inv_batch

@numba.jit(nopython=True, nogil=True, cache=True)
def matmul_2x2_batch(a_batch, b_batch):
    n = a_batch.shape[0]
    assert a_batch.shape == (n, 2, 2) and b_batch.shape == (n, 2, 2), \
        "Input batches must have shape (N, 2, 2)"
    
    # Preallocate output array (match input dtype for consistency)
    res_batch = np.empty((n, 2, 2), dtype=a_batch.dtype)
    
    # Vectorized batch multiplication (no Python loops)
    res_batch[:, 0, 0] = a_batch[:, 0, 0] * b_batch[:, 0, 0] + a_batch[:, 0, 1] * b_batch[:, 1, 0]
    res_batch[:, 0, 1] = a_batch[:, 0, 0] * b_batch[:, 0, 1] + a_batch[:, 0, 1] * b_batch[:, 1, 1]
    res_batch[:, 1, 0] = a_batch[:, 1, 0] * b_batch[:, 0, 0] + a_batch[:, 1, 1] * b_batch[:, 1, 0]
    res_batch[:, 1, 1] = a_batch[:, 1, 0] * b_batch[:, 0, 1] + a_batch[:, 1, 1] * b_batch[:, 1, 1]
    
    return res_batch


@numba.jit(nopython=True, nogil=True, cache=True)
def matmul_2x2_vec_batch(mat_batch, vec_batch):
    n = mat_batch.shape[0]
    assert mat_batch.shape == (n, 2, 2) and vec_batch.shape == (n, 2), \
        "Matrices must be (N,2,2) and vectors must be (N,2)"
    
    # Preallocate output array (preserve input dtype)
    res_batch = np.empty((n, 2), dtype=mat_batch.dtype)
    
    # Vectorized batch multiplication (compiled to SIMD instructions)
    res_batch[:, 0] = mat_batch[:, 0, 0] * vec_batch[:, 0] + mat_batch[:, 0, 1] * vec_batch[:, 1]
    res_batch[:, 1] = mat_batch[:, 1, 0] * vec_batch[:, 0] + mat_batch[:, 1, 1] * vec_batch[:, 1]
    
    return res_batch


@numba.jit(nopython=True, nogil=True, cache=True)
def flip_coord_z(x2d: np.ndarray, z2d: np.ndarray) -> None:
    """Flip z direction"""
    x2d[:] = x2d[::-1, :]
    z2d[:] = z2d[::-1, :]


@numba.jit(nopython=True, nogil=True, cache=True)
def flip_step_z(step: np.ndarray) -> None: 
    """Flip step array in z direction"""
    step[:] = step[::-1]