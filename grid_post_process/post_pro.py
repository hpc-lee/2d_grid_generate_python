#!/usr/bin/env python3
"""
Post-processing for 2D grid: sample, stretch, quality check, export
"""

import argparse
from argparse import Namespace
from pathlib import Path
import time
import sys
import json
import ctypes
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from grid_data import GridData
from grid_io import read_import_coord, cfgs_print
from grid_sample import grid_sample
from common.io_operetions import coord_export_mpi
from common.grid_quality import grid_quality_check, cal_min_dist
from common.utils import remove_comment_keys


def _load_c_lib():
    """Load the C shared library for compute-heavy operations."""
    lib_path = str(project_root / "src_c" / "libgrid.so")
    lib = ctypes.CDLL(lib_path)

    # sample_interp_c — the only compute-heavy function worth C acceleration
    lib.sample_interp_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.sample_interp_c.restype = None

    return lib


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
    args = _parse_args(args_)

    try:
        with open(args.config_file, 'r') as f:
            cfgs = json.load(f)
            cfgs = remove_comment_keys(cfgs)
    except Exception as e:
        print(f"read config json file error: {e}")
        return

    if args.verbose > 0:
        cfgs_print(cfgs)

    lib = _load_c_lib()

    t_start = time.time()
    gdcurv = read_import_coord(cfgs)

    if cfgs['flag_sample'] == 1:
        print("-" * 20 + " sample grid " + "-" * 20)
        gdcurv_out = grid_sample(gdcurv, cfgs['sample_factor'],
                                 use_c=True, lib=lib)
    else:
        print("-" * 20 + " not sample grid " + "-" * 20)
        gdcurv_out = gdcurv

    print("export coord to file ...")
    coord_export_mpi(gdcurv_out, cfgs)

    if cfgs['grid_check'] == 1:
        print("******************************************************")
        print("***** grid quality check and export quality data *****")
        print("******************************************************")
        grid_quality_check(gdcurv_out, cfgs, export_mode='mpi')

    indx_i, indx_k, dL_min = cal_min_dist(gdcurv_out)
    print(f"indx is ({indx_i},{indx_k}),dL_min_global is {dL_min}")

    t_end = time.time()
    print(f"\ngrid post-process running time is :{t_end - t_start} s")


if __name__ == "__main__":
    main()
