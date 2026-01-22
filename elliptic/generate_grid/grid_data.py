import sys
import numpy as np
from pathlib import Path
from mpi4py import MPI


class GridData:
    def __init__(self):
        # Local grid dimensions (excluding ghost points)
        self.ni = 0
        self.nk = 0
        
        # Total local dimensions (including 2 ghost points: +2)
        self.nx = 0
        self.nz = 0
        
        # Local index range (1-based indexing for inner points, like Fortran/C legacy)
        self.ni1 = 0
        self.ni2 = 0
        self.nk1 = 0
        self.nk2 = 0
        
        # Global index range (excluding ghost points, 0-based or consistent with global grid)
        self.gni1 = 0
        self.gnk1 = 0
        self.gni2 = 0
        self.gnk2 = 0
        # Total global grid size (including boundaries, i.e., total_nx = global Nx)
        self.total_nx = 0
        self.total_nz = 0
        
        # Data arrays (will be allocated later)
        self.x2d = None      # shape: (nz, nx)
        self.z2d = None      # shape: (nz, nx)
        
        self.x2d_tmp = None
        self.z2d_tmp = None
        
        # I/O related
        self.fname_part = ""


def grid_info_set(mympi, cfgs: dict) -> GridData: 
    """
    Set up grid decomposition info for a 2D domain with ghost cells.
    
    """
    gdcurv = GridData()
    gdcurv.total_nx = cfgs['number_of_grid_points_x']
    gdcurv.total_nz = cfgs['number_of_grid_points_z']

    # ----------------------------
    # X-direction decomposition
    # ----------------------------
    # Effective interior points (exclude 2 boundary points)
    nx_et = gdcurv.total_nx - 2

    nx_avg = nx_et // mympi.nprocx
    nx_left = nx_et % mympi.nprocx

    # Local interior points in x
    ni = nx_avg
    if mympi.topoid[0] < nx_left:
        ni += 1

    # Compute global start index (gni1) for interior points (0-based, excludes ghost)
    if mympi.topoid[0] == 0:
        gni1 = 0
    else:
        gni1 = mympi.topoid[0] * nx_avg
        if nx_left != 0:
            gni1 += min(mympi.topoid[0], nx_left)

    # ----------------------------
    # Z-direction decomposition
    # ----------------------------
    nz_et = gdcurv.total_nz - 2

    nz_avg = nz_et // mympi.nprocz
    nz_left = nz_et % mympi.nprocz

    nk = nz_avg
    if mympi.topoid[1] < nz_left:
        nk += 1

    if mympi.topoid[1] == 0:
        gnk1 = 0
    else:
        gnk1 = mympi.topoid[1] * nz_avg
        if nz_left != 0:
            gnk1 += min(mympi.topoid[1], nz_left)

    # ----------------------------
    # Finalize local grid info
    # ----------------------------

    # These ghost points are actually the four given boundaries.
    nghost = 1
    nx = ni + 2 * nghost  # add 2 ghost points (left & right)
    nz = nk + 2 * nghost  # add 2 ghost points (bottom & top)
    
    gdcurv.ni = ni
    gdcurv.nk = nk
    gdcurv.nx = nx
    gdcurv.nz = nz

    #  ni1: Start index of local grid for current MPI rank
    #  Index range includes ghost points, with ghost layer size = 1
    #  Ghost point index = 0, index of the first physical valid point = 1
    gdcurv.ni1 = nghost
    gdcurv.ni2 = gdcurv.ni1 + ni - 1  # = ni
    gdcurv.nk1 = nghost
    gdcurv.nk2 = gdcurv.nk1 + nk - 1  # = nk

    # Global index range (for interior points only, 0-based)
    gdcurv.gni1 = gni1
    gdcurv.gni2 = gni1 + ni - 1
    gdcurv.gnk1 = gnk1
    gdcurv.gnk2 = gnk1 + nk - 1

    gdcurv.fname_part = f"px{mympi.topoid[0]}_pz{mympi.topoid[1]}" 

    return gdcurv


def grid_info_print(gdcurv: GridData, myid: int) -> None:
    """
    Print grid decomposition info for debugging.
    """
    def fmt(val):
        return f"{val:<10d}"  # left-aligned, width=10, integer

    print("-------------------------------------------------------")
    print("--> grid info:")
    print("-------------------------------------------------------")
    print(f"my rank id is {myid}")
    print(f"file name part is {gdcurv.fname_part}")
    print(f" nx    = {fmt(gdcurv.nx)}")
    print(f" nz    = {fmt(gdcurv.nz)}")
    print(f" ni    = {fmt(gdcurv.ni)}")
    print(f" nk    = {fmt(gdcurv.nk)}")

    print(f" ni1   = {fmt(gdcurv.ni1)}")
    print(f" ni2   = {fmt(gdcurv.ni2)}")
    print(f" nk1   = {fmt(gdcurv.nk1)}")
    print(f" nk2   = {fmt(gdcurv.nk2)}")

    print(f" ni1_to_glob_phys0   = {fmt(gdcurv.gni1)}")
    print(f" ni2_to_glob_phys0   = {fmt(gdcurv.gni2)}")
    print(f" nk1_to_glob_phys0   = {fmt(gdcurv.gnk1)}")
    print(f" nk2_to_glob_phys0   = {fmt(gdcurv.gnk2)}")

    sys.stdout.flush()


def grid_init_set(gdcurv: GridData) -> None:
    nx = gdcurv.nx
    nz = gdcurv.nz
    gdcurv.x2d = np.zeros((nz, nx), dtype=np.float32)
    gdcurv.z2d = np.zeros((nz, nx), dtype=np.float32)
    gdcurv.x2d_tmp = np.zeros((nz, nx), dtype=np.float32)
    gdcurv.z2d_tmp = np.zeros((nz, nx), dtype=np.float32)


class Bdry:
    def __init__(self):
        self.nx_all = 0          # number_of_grid_points_x
        self.nz_all = 0          # number_of_grid_points_z
        self.bx1 = None      # shape (nz, 2) → [[x0, z0], [x1, z1], ...]
        self.bx2 = None      # shape (nz, 2)
        self.bz1 = None      # shape (nx, 2)
        self.bz2 = None      # shape (nx, 2)


def init_bdry(cfgs: dict) -> Bdry:
    """Initialize boundary arrays with proper shapes."""
    bdry = Bdry()
    bdry.nx_all = cfgs['number_of_grid_points_x']
    bdry.nz_all = cfgs['number_of_grid_points_z']

    # Allocate clean, intuitive arrays
    bdry.bx1 = np.zeros((bdry.nz_all, 2), dtype=np.float32)
    bdry.bx2 = np.zeros((bdry.nz_all, 2), dtype=np.float32)
    bdry.bz1 = np.zeros((bdry.nx_all, 2), dtype=np.float32)
    bdry.bz2 = np.zeros((bdry.nx_all, 2), dtype=np.float32)

    return bdry


def read_bdry(myid: int, bdry: Bdry, geometry_file: str) -> None:
    """Read boundary from file into structured arrays."""
    path = Path(geometry_file)
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

    try:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        print(f"read config json file error: {e}")
        sys.exit(1)

    nx_all = bdry.nx_all
    nz_all = bdry.nz_all

    idx = 0
    # Read bx1: nz points
    for k in range(nz_all):
        vals = lines[idx].split()
        bdry.bx1[k, 0] = float(vals[0])  # x
        bdry.bx1[k, 1] = float(vals[1])  # z
        idx += 1

    # Read bx2: nz points
    for k in range(nz_all):
        vals = lines[idx].split()
        bdry.bx2[k, 0] = float(vals[0])
        bdry.bx2[k, 1] = float(vals[1])
        idx += 1

    # Read bz1: nx points
    for i in range(nx_all):
        vals = lines[idx].split()
        bdry.bz1[i, 0] = float(vals[0])  # x
        bdry.bz1[i, 1] = float(vals[1])  # z
        idx += 1

    # Read bz2: nx points
    for i in range(nx_all):
        vals = lines[idx].split()
        bdry.bz2[i, 0] = float(vals[0])
        bdry.bz2[i, 1] = float(vals[1])
        idx += 1

    # Only rank 0 performs consistency check
    if myid == 0:
        check_bdry(bdry)


def check_bdry(bdry: Bdry) -> None:
    """Check corner consistency using intuitive array access."""
    tol = 1e-6

    # (0, 0): bx1[0] vs bz1[0]
    p1 = bdry.bx1[0]      # [x, z]
    p2 = bdry.bz1[0]
    if np.abs(p1 - p2).sum() > tol:
        raise ValueError(f"Corner (0,0) mismatch: bx1[0]={p1}, bz1[0]={p2}")

    # (nx-1, 0): bx2[0] vs bz1[-1]
    p1 = bdry.bx2[0]
    p2 = bdry.bz1[-1]
    if np.abs(p1 - p2).sum() > tol:
        raise ValueError(f"Corner (nx-1,0) mismatch: bx2[0]={p1}, bz1[-1]={p2}")

    # (nx-1, nz-1): bx2[-1] vs bz2[-1]
    p1 = bdry.bx2[-1]
    p2 = bdry.bz2[-1]
    if np.abs(p1 - p2).sum() > tol:
        raise ValueError(f"Corner (nx-1,nz-1) mismatch: bx2[-1]={p1}, bz2[-1]={p2}")

    # (0, nz-1): bx1[-1] vs bz2[0]
    p1 = bdry.bx1[-1]
    p2 = bdry.bz2[0]
    if np.abs(p1 - p2).sum() > tol:
        raise ValueError(f"Corner (0,nz-1) mismatch: bx1[-1]={p1}, bz2[0]={p2}")

    print("Boundary check completed. Boundaries are normal.")


def grid_info_reset(gdcurv: GridData, mympi):
    # Reconfigure grid information for each rank for grid output

    # Check left boundary (neighid[0]) - expand grid if no left neighbor
    if mympi.neighid[0] == MPI.PROC_NULL:
        gdcurv.ni += 1
        gdcurv.ni1 -= 1
    
    # Check right boundary (neighid[1]) - expand grid if no right neighbor  
    if mympi.neighid[1] == MPI.PROC_NULL:
        gdcurv.ni += 1
        gdcurv.ni2 += 1
    
    # Check bottom boundary (neighid[2]) - expand grid if no bottom neighbor
    if mympi.neighid[2] == MPI.PROC_NULL:
        gdcurv.nk += 1
        gdcurv.nk1 -= 1
    
    # Check top boundary (neighid[3]) - expand grid if no top neighbor
    if mympi.neighid[3] == MPI.PROC_NULL:
        gdcurv.nk += 1
        gdcurv.nk2 += 1

    # Adjust global left index if topology exists (topoid[0])
    if mympi.topoid[0] != 0:
        gdcurv.gni1 += 1
    
    # Adjust global bottom index if topology exists (topoid[1])  
    if mympi.topoid[1] != 0:
        gdcurv.gnk1 += 1
    gdcurv.x2d = gdcurv.x2d[gdcurv.nk1:gdcurv.nk2+1, gdcurv.ni1:gdcurv.ni2+1].copy()
    gdcurv.z2d = gdcurv.z2d[gdcurv.nk1:gdcurv.nk2+1, gdcurv.ni1:gdcurv.ni2+1].copy()


def cfgs_print(cfgs: dict):
    print(f"number of total grid points x is {cfgs['number_of_grid_points_x']}")
    print(f"number of total grid points z is {cfgs['number_of_grid_points_z']}")

    print(f"number of mpi procs x is {cfgs['number_of_mpiprocs_x']}")
    print(f"number of mpi procs z is {cfgs['number_of_mpiprocs_z']}")
    
    print(f"input geometry file is \n {cfgs['geometry_input_file']}")
    print(f"export grid dir is \n {cfgs['grid_export_dir']}")
    print("-------------------------------------------------------")
    
    if cfgs['check_orth'] == 1:
        print("------- check grid orthogonality-------")
    if cfgs['check_jac'] == 1:
        print("------- check grid jacobi-------")
    if cfgs['check_ratio'] == 1:
        print("------- check grid ratio-------")
    if cfgs['check_step_xi'] == 1:
        print("------- check grid step xi direction-------")
    if cfgs['check_step_zt'] == 1:
        print("------- check grid step zt direction-------")
    if cfgs['check_smooth_xi'] == 1:
        print("------- check grid smooth xi direction-------")
    if cfgs['check_smooth_zt'] == 1:
        print("------- check grid smooth zt direction-------")
    
    cur_method = cfgs['method']
    print(f"methon is {cfgs['method']}")
    if (cur_method == 'tfi'):
        pass
    else:
        print(f"coef: {cfgs['coef']}")
        print(f"weight:   {cfgs['weight']}")
        print(f"iter_err: {cfgs['iter_err']}")
        print(f"max_iter: {cfgs['max_iter']}")
    
        