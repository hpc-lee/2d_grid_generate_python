#!/usr/bin/env python3
"""
Python version of the 2D grid generation program with NumPy vectorization
This program generates 2D curvilinear grids using parabolic method
"""

import argparse
from argparse import Namespace
from pathlib import Path
import time
import sys
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from grid_data import cfgs_print, grid_sample
from grid_data import read_import_coord, coord_export_base_mpi
from grid_data import grid_quality_check_base_mpi
from common.grid_quality import cal_min_dist
from common.utils import remove_comment_keys


def _parse_args(args: list[str] | None) -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)

    return parser.parse_args(args)


def _swap_xz(gdcurv) -> None:
    """Swap x/z axes in place: transpose grid arrays and swap nx/nz, ni/nk."""
    gdcurv.x2d, gdcurv.z2d = gdcurv.z2d.T.copy(), gdcurv.x2d.T.copy()
    gdcurv.nx, gdcurv.nz = gdcurv.nz, gdcurv.nx
    gdcurv.ni, gdcurv.nk = gdcurv.nk, gdcurv.ni


def main(args_: list[str] | None = None) -> None:
    """Main function"""
    
    args = _parse_args(args_)
    
    # Read parameters
    try:
        with open(args.config_file, 'r') as f:
            cfgs = json.load(f)
            cfgs = remove_comment_keys(cfgs)
    except Exception as e:
        print(f"read config json file error: {e}")
    
    if (args.verbose > 0):
        cfgs_print(cfgs)

    t_start = time.time()
    print(type(cfgs))
    gdcurv = read_import_coord(cfgs)

    # Optional x/z axis swap: swap the x and z axes to convert from the
    # generation (transposed) coordinate system to the physical coordinate
    # system. This is the inverse of the transform applied in creat_bx_bdry.py
    # (bx1/bx2 -> bz1/bz2) when a parabolic grid is generated along x-direction
    # boundaries. Applied at the very beginning, so all subsequent operations
    # (sample/quality/export) run on the swapped (physical) grid. Swaps x<->z
    # for the grid arrays and every direction-dependent config (mpiprocs_out,
    # PML, sample_factor).
    if (cfgs['flag_swap_xz'] == 1):
        _swap_xz(gdcurv)
        cfgs['number_of_mpiprocs_out'] = list(cfgs['number_of_mpiprocs_out'][::-1])
        if 'number_of_pml_x1' in cfgs:
            cfgs['number_of_pml_x1'], cfgs['number_of_pml_z1'] = \
                cfgs['number_of_pml_z1'], cfgs['number_of_pml_x1']
            cfgs['number_of_pml_x2'], cfgs['number_of_pml_z2'] = \
                cfgs['number_of_pml_z2'], cfgs['number_of_pml_x2']
        if 'sample_factor' in cfgs:
            cfgs['sample_factor'] = list(cfgs['sample_factor'][::-1])
        print(f"applied x/z axis swap: grid now (nz={gdcurv.nz}, nx={gdcurv.nx})")

    if (cfgs['flag_sample'] == 1):
        print("-"*20 + " sample grid " + "-"*20)
        gdcurv_sample = grid_sample(gdcurv, cfgs['sample_factor'])
        print("export coord to file ...")
        coord_export_base_mpi(gdcurv_sample, cfgs)
        # Grid quality check and export quality data
        if cfgs['grid_check'] == 1:
            print("******************************************************")
            print("***** grid quality check and export quality data *****")
            print("******************************************************")
            grid_quality_check_base_mpi(gdcurv, cfgs)
        # Calculate min distance
        indx_i, indx_k, dL_min = cal_min_dist(gdcurv_sample)
        print(f"indx is ({indx_i},{indx_k}),dL_min_global is {dL_min}")
    else:
        print("-"*20 + " not sample grid " + "-"*20)
        print("export coord to file ...")
        coord_export_base_mpi(gdcurv, cfgs)
        # Grid quality check and export quality data
        if cfgs['grid_check'] == 1:
            print("******************************************************")
            print("***** grid quality check and export quality data *****")
            print("******************************************************")
            grid_quality_check_base_mpi(gdcurv, cfgs)
        # Calculate min distance
        indx_i, indx_k, dL_min = cal_min_dist(gdcurv)
        print(f"indx is ({indx_i},{indx_k}),dL_min_global is {dL_min}")

    # Write a draw-ready config (with swapped nx/nz if flag_swap_xz) so that
    # plotting scripts (draw_grid.py) read the correct output grid dimensions.
    out_gdcurv = gdcurv_sample if cfgs['flag_sample'] == 1 else gdcurv
    draw_cfgs = {
        "number_of_grid_points_x": int(out_gdcurv.nx),
        "number_of_grid_points_z": int(out_gdcurv.nz),
        "number_of_mpiprocs_x": int(cfgs['number_of_mpiprocs_out'][0]),
        "number_of_mpiprocs_z": int(cfgs['number_of_mpiprocs_out'][1]),
        "grid_export_dir": cfgs['grid_export_dir'],
    }
    with open(Path(cfgs['grid_export_dir']) / 'config.json', 'w') as f:
        json.dump(draw_cfgs, f, indent=4)

    t_end = time.time()
    print("\n************************************")
    print(f"grid generate running time is :{t_end - t_start} s")
    print("************************************\n")


if __name__ == "__main__":
    main()