#!/usr/bin/env python3
"""
Python version of the 2D grid generation program with NumPy vectorization
This program generates 2D curvilinear grids using elliptic method
"""

import argparse
from argparse import Namespace
from pathlib import Path
import time
import sys
import json
from mpi4py import MPI
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from grid_data import grid_info_set, grid_info_print
from grid_data import grid_init_set, init_bdry, cfgs_print
from grid_data import read_bdry, grid_info_reset
from grid_utils import linear_tfi
from dirichlet import diri_gene
from higenstock import higen_gene
from mympi import mympi_set, grid_coord_exchange
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

    # init mpi    
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    size = comm.Get_size()

    args = _parse_args(args_)
    if myid == 0:
        print(f"mpi size={size}")
    
    cfgs = None
    # Read parameters
    if myid == 0:
        try:
            with open(args.config_file, 'r') as f:
                cfgs = json.load(f)
                cfgs = remove_comment_keys(cfgs)
        except Exception as e:
            print(f"read config json file error: {e}")
            sys.exit(1)
    cfgs = comm.bcast(cfgs, root=0)

    if (args.verbose > 0 and myid == 0):
        cfgs_print(cfgs)

    lib = None
    if (cfgs['execute_C_code'] == 1):
        import ctypes
        lib_path = str(project_root / "src_c" / "libgrid.so")
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError as e:
            print(f"Rank {myid} 加载C库失败: {e}")
            comm.Abort(1)  # 所有rank退出，避免死锁

        lib.interp_inner_source_c.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_P 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_P_x1 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_P_x2 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_P_z1 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_P_z2 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_x1 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_x2 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_z1 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_z2 
            ctypes.c_int,  # nx
            ctypes.c_int,  # nz
            ctypes.c_int,  # gni1
            ctypes.c_int,  # gnk1
            ctypes.c_int,  # total_nx
            ctypes.c_int,  # total_nz
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_z1 
            np.ctypeslib.ndpointer(dtype=np.float32),  # src_Q_z2 
            ]
        lib.interp_inner_source_c.restype = None

        lib.compute_residual_c.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),  # x2d
            np.ctypeslib.ndpointer(dtype=np.float32),  # z2d
            np.ctypeslib.ndpointer(dtype=np.float32),  # x2d_tmp
            np.ctypeslib.ndpointer(dtype=np.float32),  # z2d_tmp
            np.ctypeslib.ndpointer(dtype=np.float32),    # local_max 
            ctypes.c_int,  # nx
            ctypes.c_int,  # nz
            ]
        lib.compute_residual_c.restype = None

    t_start = time.time()
    # set mpi info
    mympi = mympi_set(cfgs, myid, comm)
    gdcurv = grid_info_set(mympi, cfgs)
    for r in range(size):
        if myid == r:
            print(f"Rank {myid}: topoid = {mympi.topoid}")
            grid_info_print(gdcurv, myid)
        comm.Barrier()  

    # init grid
    grid_init_set(gdcurv)
    
    # init grid
    bdry = init_bdry(cfgs)

    read_bdry(myid, bdry, cfgs['geometry_input_file'])
    # Generate grid
    linear_tfi(gdcurv, bdry, mympi)
    grid_coord_exchange(gdcurv, mympi)

    if (cfgs['method'] == "dirichlet"):
        diri_gene(gdcurv, cfgs, mympi, lib)
    if (cfgs['method'] == "higenstock"):
        higen_gene(gdcurv, cfgs, mympi, lib)

    t_end = time.time()
    
    if (myid == 0):
        print("\n************************************")
        print(f"grid generate running time is :{t_end - t_start} s")
        print("************************************\n")
        print("export coord to file ...")

    grid_info_reset(gdcurv, mympi)
    coord_export(gdcurv, cfgs['grid_export_dir'])
    
    # Calculate min distance
    indx_i, indx_k, dL_min = cal_min_dist(gdcurv)
    for r in range(size):
        if myid == r:
            print(f"rank = {myid}, dL_min_indx is (ni = {indx_i}, nk = {indx_k}),dL_min_global is {dL_min}")
        comm.Barrier()  
    
    # Grid quality check and export quality data
    if cfgs['grid_check'] == 1:
        if (myid == 0):
            print("******************************************************")
            print("***** grid quality check and export quality data *****")
            print("******************************************************")
        grid_quality_check(gdcurv, cfgs)

if __name__ == "__main__":
    main()