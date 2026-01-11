import numpy as np


def export_bdry(bz: np.ndarray, nx: int,
                file_dir: str) -> None:
    with open(file_dir, 'w') as fd:
        fd.write(f"# bz coords, nx is {nx}\n") 
        for i in range(nx):
            fd.write(f'{bz[i, 0]:.9e} {bz[i, 1]:.9e}\n')
