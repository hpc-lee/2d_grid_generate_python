import numpy as np


def export_bdry(bz1: np.ndarray, bz2: np.ndarray, nx: int,
                file_dir: str) -> None:
    with open(file_dir, 'w') as fd:
        fd.write(f"# bz1 coords, nx is {nx}\n") 
        for i in range(nx):
            fd.write(f'{bz1[i, 0]:.9e} {bz1[i, 1]:.9e}\n')
        
        fd.write(f"# bz2 coords, nx is {nx}\n")
        for i in range(nx):
            fd.write(f'{bz2[i, 0]:.9e} {bz2[i, 1]:.9e}\n')

