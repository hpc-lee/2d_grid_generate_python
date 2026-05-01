import sys
import os
import gc
import numpy as np
import netCDF4 as nc
from grid_data import GridData
from common.algebra import xi_arc_stretch, zt_arc_stretch
from common.utils import print_quality_checks


def read_import_coord(cfgs: dict) -> GridData:
    if 'input_grid_number' not in cfgs:
        raise KeyError("cfgs missing required configuration: input_grid_number")
    grid_count = cfgs['input_grid_number']
    if grid_count < 1:
        raise ValueError(f"input_grid_number must be >= 1, current value: {grid_count}")

    if grid_count >= 2:
        if 'merge_direction' not in cfgs:
            raise KeyError(f"Missing configuration 'merge_direction' when merging {grid_count} grids")
        merge_dir = cfgs['merge_direction']
        if merge_dir not in ['x', 'z']:
            raise ValueError(f"merge_direction only supports 'x'/'z', current value: {merge_dir}")

    gdcurv_dict = {}
    for i in range(grid_count):
        grid_key = f"input_grid_info_{i}"
        nx = cfgs[grid_key]['number_of_grid_points'][0]
        nz = cfgs[grid_key]['number_of_grid_points'][1]
        nprocx_in = cfgs[grid_key]['number_of_mpiprocs_in'][0]
        nprocz_in = cfgs[grid_key]['number_of_mpiprocs_in'][1]
        import_dir = cfgs[grid_key]['grid_import_dir']
        gdcurv = GridData(nx, nz)
        gdcurv_dict[f"gdcurv_{i}"] = gdcurv

        att_global = "global_index_of_first_physical_points"
        att_count = "count_of_physical_points"
        start = [0, 0]
        for kk in range(nprocz_in):
            for ii in range(nprocx_in):
                fname_coords = f"px{ii}_pz{kk}"
                in_file = f"{import_dir}/coord_{fname_coords}.nc"
                try:
                    with nc.Dataset(in_file, 'r') as ds:
                        global_index = ds.getncattr(att_global)
                        count_points = ds.getncattr(att_count)

                        gni1 = global_index[0]
                        gnk1 = global_index[1]
                        ni = count_points[0]
                        nk = count_points[1]

                        coord_x = ds.variables['x'][start[0]:start[0]+nk,
                                                    start[1]:start[1]+ni].astype(np.float32)
                        coord_z = ds.variables['z'][start[0]:start[0]+nk,
                                                    start[1]:start[1]+ni].astype(np.float32)

                        global_z_slice = slice(gnk1, gnk1 + nk)
                        global_x_slice = slice(gni1, gni1 + ni)

                        gdcurv.x2d[global_z_slice, global_x_slice] = coord_x
                        gdcurv.z2d[global_z_slice, global_x_slice] = coord_z

                except FileNotFoundError:
                    raise FileNotFoundError(f"NC file not found: {in_file}")
                except KeyError as e:
                    raise KeyError(f"NC file {in_file} missing attribute/variable: {e}")
                except Exception as e:
                    raise RuntimeError(f"Failed to read NC file {in_file}: {str(e)}")

        if cfgs[grid_key]['flag_stretch'] == 1:
            stretch_file = cfgs[grid_key]['stretch_file']
            stretch_dire = cfgs[grid_key]['stretch_direction']

            try:
                with open(stretch_file, 'r') as fp:
                    lines = [line.strip() for line in fp
                             if line.strip() and not line.strip().startswith('#')]
            except Exception as e:
                print(f"read stretch file error: {e}")
                sys.exit(1)

            # Detect format: C format has npoints as first line, Python format is all floats
            first_val = int(float(lines[0]))
            if first_val == len(lines) - 1:
                npoints = first_val
                arc_len = np.array([float(v) for v in lines[1:npoints+1]], dtype=np.float32)
            else:
                arc_len = np.array([float(v) for v in lines], dtype=np.float32)

            expected_len = nz if stretch_dire == 'z' else nx
            if len(arc_len) != expected_len:
                print(f"ERROR: arc_len data length {len(arc_len)} does not match expected {expected_len}")
                sys.exit(1)

            if stretch_dire == 'z':
                zt_arc_stretch(gdcurv, arc_len)
            elif stretch_dire == 'x':
                xi_arc_stretch(gdcurv, arc_len)

    if grid_count == 1:
        return gdcurv_dict["gdcurv_0"]

    total_nx = 0
    total_nz = 0
    if cfgs['merge_direction'] == 'x':
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            total_nx += cfgs[grid_key]['number_of_grid_points'][0]
            total_nz = cfgs[grid_key]['number_of_grid_points'][1]
        total_nx = total_nx - grid_count + 1

    if cfgs['merge_direction'] == 'z':
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            total_nx = cfgs[grid_key]['number_of_grid_points'][0]
            total_nz += cfgs[grid_key]['number_of_grid_points'][1]
        total_nz = total_nz - grid_count + 1

    merge_gdcurv = GridData(total_nx, total_nz)
    gni1 = 0
    gnk1 = 0
    for i in range(grid_count):
        gdcurv_in_one = gdcurv_dict[f"gdcurv_{i}"]
        nx_in = gdcurv_in_one.nx
        nz_in = gdcurv_in_one.nz

        if cfgs['merge_direction'] == 'x' and i > 0:
            gni1 -= 1
        if cfgs['merge_direction'] == 'z' and i > 0:
            gnk1 -= 1

        global_z_slice = slice(gnk1, gnk1 + nz_in)
        global_x_slice = slice(gni1, gni1 + nx_in)

        merge_gdcurv.x2d[global_z_slice, global_x_slice] = gdcurv_in_one.x2d
        merge_gdcurv.z2d[global_z_slice, global_x_slice] = gdcurv_in_one.z2d

        if cfgs['merge_direction'] == 'x':
            gni1 += nx_in
        if cfgs['merge_direction'] == 'z':
            gnk1 += nz_in

        gdcurv_in_one.x2d = None
        gdcurv_in_one.z2d = None
        del gdcurv_dict[f"gdcurv_{i}"]
        gc.collect()

    return merge_gdcurv


def cfgs_print(cfgs: dict):
    print("-"*20 + " input grid info " + "-"*20)
    print(f"input_grid_number is {cfgs['input_grid_number']}")
    for i in range(cfgs['input_grid_number']):
        print(f"Information of the {i} grid:")
        grid_key = f"input_grid_info_{i}"
        print(f"number of grid points [nx, nz] is {cfgs[grid_key]['number_of_grid_points']}")
        print(f"number of mpi procs [proc_x proc_z] is {cfgs[grid_key]['number_of_mpiprocs_in']}")
        print(f"grid import dir is {cfgs[grid_key]['grid_import_dir']}")
        if cfgs[grid_key]['flag_stretch'] == 1:
            print(f"grid need stretch and stretch direction is {cfgs[grid_key]['stretch_direction']}")
            print(f"stretch file is {cfgs[grid_key]['stretch_file']}")

    print("-"*20 + " output grid info " + "-"*20)
    print(f"output grid number of mpi procs [proc_x proc_z] is {cfgs['number_of_mpiprocs_out']}")
    if cfgs["input_grid_number"] >= 2:
        print(f"input grid number >= 2, need merge and merge direction is {cfgs['merge_direction']}")
    if cfgs["flag_sample"] == 1:
        print(f"sample_factor along x and z is {cfgs['sample_factor']}")
        if cfgs['sample_factor'][0] < 1 or cfgs['sample_factor'][1] < 1:
            print("sample coef must >= 1")
            sys.exit(1)
    print(f"grid export dir is {cfgs['grid_export_dir']}")

    print_quality_checks(cfgs)

    if cfgs['pml_weight_2x'] == 1:
        print("double the grid computation weight of PML layers")
        print("number_of_pml_z2 maybe is free-surface, pml layers is 0")

        print(f"number_of_pml_x1 is {cfgs['number_of_pml_x1']}")
        print(f"number_of_pml_x2 is {cfgs['number_of_pml_x2']}")
        print(f"number_of_pml_z1 is {cfgs['number_of_pml_z1']}")
        print(f"number_of_pml_z2 is {cfgs['number_of_pml_z2']}")
