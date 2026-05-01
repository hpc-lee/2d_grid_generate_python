#!/usr/bin/env python3
"""
Python version of the 2D grid generation program with NumPy vectorization
This program generates 2D curvilinear grids using hyperbolic method
"""

import argparse
from argparse import Namespace
from pathlib import Path
import time
import sys
import json
import ctypes
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from grid_data import grid_init_set, cfgs_print
from common.io_operetions import coord_export
from common.grid_quality import grid_quality_check
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

    # Load C library
    lib_path = str(project_root / "src_c" / "libgrid.so")
    lib = ctypes.CDLL(lib_path)
    lib.hyper_gene_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.hyper_gene_c.restype = None

    t_start = time.time()
    # Generate grid
    gdcurv = grid_init_set(cfgs)

    lib.hyper_gene_c(gdcurv.x2d, gdcurv.z2d, gdcurv.step,
                    gdcurv.nx, gdcurv.nz, cfgs['coef'],
                    cfgs['t2b'], cfgs['flag_stretch'])

    t_end = time.time()
    
    print("\n************************************")
    print(f"grid generate running time is :{t_end - t_start} s")
    print("************************************\n")
    
    print("export coord to file ...")
    coord_export(gdcurv, cfgs['grid_export_dir'])
    
    # Calculate min distance
    indx_i, indx_k, dL_min = cal_min_dist(gdcurv)
    print(f"indx is ({indx_i},{indx_k}),dL_min_global is {dL_min}")
    
    # Grid quality check and export quality data
    if cfgs['grid_check'] == 1:
        print("******************************************************")
        print("***** grid quality check and export quality data *****")
        print("******************************************************")
        grid_quality_check(gdcurv, cfgs)


if __name__ == "__main__":
    main()