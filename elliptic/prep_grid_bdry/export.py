import numpy as np

def export_bdry(bx1: np.ndarray, bx2: np.ndarray, 
                bz1: np.ndarray, bz2: np.ndarray, 
                nx: int, nz: int, file_dir: str) -> None:
    with open(file_dir, 'w') as fd:
        fd.write(f"# bx1 coords, nz is {nz}\n") 
        for i in range(nz):
            fd.write(f'{bx1[i, 0]:.9e} {bx1[i, 1]:.9e}\n')
        fd.write(f"# bx2 coords, nz is {nz}\n") 
        for i in range(nz):
            fd.write(f'{bx2[i, 0]:.9e} {bx2[i, 1]:.9e}\n')
        fd.write(f"# bz1 coords, nx is {nx}\n") 
        for i in range(nx):
            fd.write(f'{bz1[i, 0]:.9e} {bz1[i, 1]:.9e}\n')
        fd.write(f"# bz2 coords, nx is {nx}\n") 
        for i in range(nx):
            fd.write(f'{bz2[i, 0]:.9e} {bz2[i, 1]:.9e}\n')
