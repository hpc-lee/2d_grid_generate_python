import sys
import numpy as np


class GridData:
    def __init__(self, nx=0, nz=0):
        self.nx = nx
        self.nz = nz


        # File Output in MPI Parallel Simulations
        self.fname_part = "px0_pz0" 
        # global_index_of_first_physical_points
        self.gni1 = 0
        self.gnk1 = 0
        # ni and nk are variables only required for MPI parallelization
        # with ghost points, which are unnecessary here; however, 
        # they are assigned for unifying data export across different methods.
        self.ni = nx
        self.nk = nz
        
        self.x2d = np.zeros((nz, nx), dtype=np.float32)
        self.z2d = np.zeros((nz, nx), dtype=np.float32)
        
        self.step = np.zeros((nz-1), dtype=np.float32)


def grid_init_set(cfgs: dict) ->GridData :
    geometry_file = cfgs['geometry_input_file']
    step_file = cfgs['step_input_file']

    nx = cfgs['number_of_grid_points_x']
    nz = cfgs['number_of_grid_points_z']
    gdcurv = GridData(nx, nz)
    
    # Read step lengths
    try:
        with open(step_file, 'r') as fp:
            steps = [line.strip() for line in fp
                     if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        print(f"read step file error: {e}")

    if (len(steps) != nz-1):
        print(f"ERROR: step data length {len(steps)} does not match nz-1={nz-1}")
        sys.exit(1)
    gdcurv.step = np.array(steps, dtype=np.float32)

    # Read geometry file
    try:
        with open(geometry_file, "r") as fp:
            bdry_coords = [list(map(float, line.split())) 
                           for line in fp 
                           if line.strip() and not line.strip().startswith('#')]
    except Exception as e:
        print(f"read geometry file error: {e}")

    
    if (len(bdry_coords) != nx):
        print(f"ERROR: grid direction is z, bdry file point number is wrong!"
              f"bdry file point is {len(bdry_coords)} does not match nx={nx}")
        sys.exit(1)
    bdry_coords = np.array(bdry_coords, dtype=np.float32)
    gdcurv.x2d[0, :] = bdry_coords[:, 0]
    gdcurv.z2d[0, :] = bdry_coords[:, 1]

    return gdcurv


def cfgs_print(cfgs: dict):
    print(f"number of total grid points x is {cfgs['number_of_grid_points_x']}")
    print(f"number of total grid points z is {cfgs['number_of_grid_points_z']}")
    
    print(f"input geometry file is \n {cfgs['geometry_input_file']}")
    print(f"input step file is \n {cfgs['step_input_file']}")
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
    
    print(f"coef is {cfgs['coef']}")
    print(f"flag_stretch is {cfgs['flag_stretch']}")
    if cfgs['t2b'] == 1:
        print("top(bdry_2) to bottom(bdry_1)")
    else:
        print("bottom(bdry_1) to top(bdry_2)")
    
        