import sys
import numpy as np
from common.utils import print_quality_checks
from common.grid_data import SimpleGridData as GridData


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

    
    if (len(bdry_coords) != 2*nx):
        print(f"ERROR: grid direction is z, bdry file point number is wrong!"
              f"bdry file point is {len(bdry_coords)/2} does not match nx={nx}")
        sys.exit(1)
    bdry_coords = np.array(bdry_coords, dtype=np.float32)
    gdcurv.x2d[0, :] = bdry_coords[:nx, 0]
    gdcurv.z2d[0, :] = bdry_coords[:nx, 1]
    gdcurv.x2d[nz-1, :] = bdry_coords[nx:, 0]
    gdcurv.z2d[nz-1, :] = bdry_coords[nx:, 1]

    return gdcurv


def cfgs_print(cfgs: dict):
    print(f"number of total grid points x is {cfgs['number_of_grid_points_x']}")
    print(f"number of total grid points z is {cfgs['number_of_grid_points_z']}")
    
    print(f"input geometry file is \n {cfgs['geometry_input_file']}")
    print(f"input step file is \n {cfgs['step_input_file']}")
    print(f"export grid dir is \n {cfgs['grid_export_dir']}")
    print("-------------------------------------------------------")

    print_quality_checks(cfgs)

    print(f"coef is {cfgs['coef']}")
    if cfgs['t2b'] == 1:
        print("top(bdry_2) to bottom(bdry_1)")
    else:
        print("bottom(bdry_1) to top(bdry_2)")