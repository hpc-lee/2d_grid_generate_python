import sys
import os
import gc
import numpy as np
import numba
import netCDF4 as nc
from typing import Tuple
from common.algebra import xi_arc_stretch, zt_arc_stretch
from common.grid_quality import cal_orth, cal_jacobi, cal_ratio
from common.grid_quality import cal_step_x, cal_step_z
from common.grid_quality import cal_smooth_x, cal_smooth_z


class GridData:
    def __init__(self, nx=0, nz=0):
        self.nx = nx
        self.nz = nz

        self.ni = nx
        self.nk = nz
        
        self.x2d = np.zeros((nz, nx), dtype=np.float32)
        self.z2d = np.zeros((nz, nx), dtype=np.float32)

    
def read_import_coord(cfgs: dict) -> GridData:
    # Core validation (key to resolving type hint issues)
    if 'input_grid_number' not in cfgs:
        raise KeyError("cfgs missing required configuration: input_grid_number")
    grid_count = cfgs['input_grid_number']
    if grid_count < 1:
        raise ValueError(f"input_grid_number must be ≥ 1, current value: {grid_count} (no grids to read)")

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
                                                # Define global slice ranges for tensor assignment
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
            if cfgs[grid_key]['stretch_direction'] == 'z':
                try:
                    with open(stretch_file, 'r') as fp:
                         arc_len = [line.strip() for line in fp
                            if line.strip() and not line.strip().startswith('#')]
                except Exception as e:
                    print(f"read stretch file error: {e}")
                    sys.exit(1)
                if (len(arc_len) != nz):
                    print(f"ERROR: arc_len data length {len(arc_len)} does not match nz={nz}")
                    sys.exit(1)
                arc_len = np.array(arc_len, dtype=np.float32)
                zt_arc_stretch(gdcurv, arc_len)

            if cfgs[grid_key]['stretch_direction'] == 'x':
                try:
                    with open(stretch_file, 'r') as fp:
                         arc_len = [line.strip() for line in fp
                            if line.strip() and not line.strip().startswith('#')]
                except Exception as e:
                    print(f"read stretch file error: {e}")
                    sys.exit(1)
                if (len(arc_len) != nx):
                    print(f"ERROR: arc_len data length {len(arc_len)} does not match nx={nx}")
                    sys.exit(1)
                arc_len = np.array(arc_len, dtype=np.float32)
                xi_arc_stretch(gdcurv, arc_len)

    total_nx = 0
    total_nz = 0
    if grid_count == 1:
        return gdcurv_dict["gdcurv_0"]
    else:
        if cfgs['merge_direction'] == 'x':
            for i in range(grid_count):
                grid_key = f"input_grid_info_{i}"
                total_nx += cfgs[grid_key]['number_of_grid_points'][0] 
                total_nz = cfgs[grid_key]['number_of_grid_points'][1] 
            # Subtract the number of shared boundary points 
            total_nx = total_nx - grid_count + 1

        if cfgs['merge_direction'] == 'z':
            for i in range(grid_count):
                grid_key = f"input_grid_info_{i}"
                total_nx = cfgs[grid_key]['number_of_grid_points'][0] 
                total_nz += cfgs[grid_key]['number_of_grid_points'][1] 
            # Subtract the number of shared boundary points 
            total_nz = total_nz - grid_count + 1

        merge_gdcurv = GridData(total_nx, total_nz)
        gni1 = 0
        gnk1 = 0
        for i in range(grid_count):
            grid_key = f"input_grid_info_{i}"
            # Get individual grid data
            gdcurv_in_one = gdcurv_dict[f"gdcurv_{i}"]
            nx_in = gdcurv_in_one.nx
            nz_in = gdcurv_in_one.nz

            # Skip shared boundary for next grid
            slice_offset = 0
            if cfgs['merge_direction'] == 'x' and i > 0:
                slice_offset = 1  
                gni1 -= slice_offset
            if cfgs['merge_direction'] == 'z' and i > 0:
                slice_offset = 1  
                gnk1 -= slice_offset

            # --------------------------
            # Vectorized grid merging 
            # --------------------------
            # Define global slice for current grid
            global_z_slice = slice(gnk1, gnk1 + nz_in)
            global_x_slice = slice(gni1, gni1 + nx_in)

            # Direct tensor assignment
            merge_gdcurv.x2d[global_z_slice, global_x_slice] = gdcurv_in_one.x2d
            merge_gdcurv.z2d[global_z_slice, global_x_slice] = gdcurv_in_one.z2d

            # Update start index for next grid
            if cfgs['merge_direction'] == 'x':
                gni1 += nx_in
            if cfgs['merge_direction'] == 'z':
                gnk1 += nz_in

            gdcurv_in_one.x2d = None       # clear numpy array
            gdcurv_in_one.z2d = None
            del gdcurv_dict[f"gdcurv_{i}"] 
            gc.collect()

        return merge_gdcurv


def grid_sample(gdcurv: GridData, coefs: list) -> GridData:
    nx = gdcurv.nx
    nz = gdcurv.nz
    coef_x = coefs[0]
    coef_z = coefs[1]
    if not isinstance(coef_x, int) or not isinstance(coef_z, int):
        print(f"ERROR：coef must be int type")
        sys.exit(1)
    if coef_x < 1 or coef_z < 1:
        print(f"ERROR: sample coef must be >= 1")
        sys.exit(1)
    nx_sample = (nx-1) * coef_x + 1
    nz_sample = (nz-1) * coef_z + 1

    gdcurv_sample = GridData(nx_sample, nz_sample)
    sample_interp(gdcurv, gdcurv_sample)

    return gdcurv_sample


def sample_interp(gdcurv: GridData, gdcurv_sample: GridData):
    nx = gdcurv.nx
    nz = gdcurv.nz
    nx_sample = gdcurv_sample.nx
    nz_sample = gdcurv_sample.nz

    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    x2d_sample = gdcurv_sample.x2d
    z2d_sample = gdcurv_sample.z2d

    u = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    r_z = np.linspace(0.0, 1.0, nz_sample, dtype=np.float32) 
    m_z = np.searchsorted(u, r_z, side='right') - 1
    m_z = np.clip(m_z, 0, nz-2)

    for i in range(nx):
        x_col = x2d[:, i]
        z_col = z2d[:, i]
        
        ratio = (r_z - u[m_z]) / (u[m_z+1] - u[m_z])
        x2d_sample[:, i] = x_col[m_z] + (x_col[m_z+1] - x_col[m_z]) * ratio
        z2d_sample[:, i] = z_col[m_z] + (z_col[m_z+1] - z_col[m_z]) * ratio

    v = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    r_x = np.linspace(0.0, 1.0, nx_sample, dtype=np.float32) 
    m_x = np.searchsorted(v, r_x, side='right') - 1
    m_x = np.clip(m_x, 0, nx-2)

    for k_sample in range(nz_sample):  
        x_temp = x2d_sample[k_sample, :nx].copy()
        z_temp = z2d_sample[k_sample, :nx].copy()
        
        ratio = (r_x - v[m_x]) / (v[m_x+1] - v[m_x])
        x2d_sample[k_sample, :] = x_temp[m_x] + (x_temp[m_x+1] - x_temp[m_x]) * ratio
        z2d_sample[k_sample, :] = z_temp[m_x] + (z_temp[m_x+1] - z_temp[m_x]) * ratio


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
    
    if cfgs['pml_weight_2x'] == 1:
        print("double the grid computation weight of PML layers")
        print("number_of_pml_z2 maybe is free-surface, pml layers is 0")

        print(f"number_of_pml_x1 is {cfgs['number_of_pml_x1']}")
        print(f"number_of_pml_x2 is {cfgs['number_of_pml_x2']}")
        print(f"number_of_pml_z1 is {cfgs['number_of_pml_z1']}")
        print(f"number_of_pml_z2 is {cfgs['number_of_pml_z2']}")


def gd_info_set(cfgs: dict, total_nx: int, total_nz: int, 
                iprocx: int, iprocz: int, global_index: list, 
                count: list) -> None:
    """
    Calculate global starting indices and data dimensions for a specific process.
    """
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocz_out = cfgs['number_of_mpiprocs_out'][1]
    if cfgs['pml_weight_2x'] == 1:
        num_of_pml_x1 = cfgs['number_of_pml_x1']  # PML layers on x1 (left) boundary
        num_of_pml_x2 = cfgs['number_of_pml_x2']  # PML layers on x2 (right) boundary
        num_of_pml_z1 = cfgs['number_of_pml_z1']  # PML layers on z1 (bottom) boundary
        num_of_pml_z2 = cfgs['number_of_pml_z2']  # PML layers on z2 (top) boundary
    else:
        num_of_pml_x1 = 0      
        num_of_pml_x2 = 0
        num_of_pml_z1 = 0
        num_of_pml_z2 = 0

    # -------------------------- Process X-direction sharding --------------------------
    # Total effective points including double-sided PML layers
    nx_et = total_nx + num_of_pml_x1 + num_of_pml_x2
    # Average points per process (base allocation)
    nx_avg = nx_et // nprocx_out
    # Remainder points (distributed to first nx_left processes, +1 point each)
    nx_left = nx_et % nprocx_out

    # Validate: Average points per process must exceed PML layer count
    if nx_avg <= num_of_pml_x1 or nx_avg <= num_of_pml_x2:
        print("Error: Average x-direction points per process (nx_avg) must be larger than PML layer counts")
        print(f"nx_avg={nx_avg}, PML x1={num_of_pml_x1}, PML x2={num_of_pml_x2}")
        sys.stdout.flush()
        exit(1)

    # Calculate shard length (ni) in x-direction
    ni = nx_avg  # Start with base average allocation
    # Subtract PML layers for boundary processes (exclude PML from physical data)
    if iprocx == 0:
        ni -= num_of_pml_x1  # First x-process removes left PML layers
    if iprocx == nprocx_out - 1:
        ni -= num_of_pml_x2  # Last x-process removes right PML layers
    # Distribute remainder points (load balancing)
    if iprocx < nx_left:
        ni += 1  # First nx_left processes get 1 extra point

    # Calculate global starting index (gni1) for x-direction
    if iprocx == 0:
        gni1 = 0  # First process starts at global index 0
    else:
        # Base offset minus left PML layers (already excluded from boundary process)
        gni1 = iprocx * nx_avg - num_of_pml_x1
    # Adjust for remainder points (load balancing correction)
    if nx_left != 0:
        gni1 += iprocx if iprocx < nx_left else nx_left

    # -------------------------- Process Z-direction sharding --------------------------
    # Same logic as x-direction (mirror implementation)
    nz_et = total_nz + num_of_pml_z1 + num_of_pml_z2  # Total effective points (with PML)
    nz_avg = nz_et // nprocz_out                      # Average points per z-process
    nz_left = nz_et % nprocz_out                      # Remainder points for load balancing

    # Validate z-direction PML vs average points
    if nz_avg <= num_of_pml_z1 or nz_avg <= num_of_pml_z2:
        print("Error: Average z-direction points per process (nz_avg) must be larger than PML layer counts")
        print(f"nz_avg={nz_avg}, PML z1={num_of_pml_z1}, PML z2={num_of_pml_z2}")
        sys.stdout.flush() 
        exit(1)

    # Calculate shard length (nk) in z-direction
    nk = nz_avg  # Base average allocation
    # Subtract PML layers for boundary processes
    if iprocz == 0:
        nk -= num_of_pml_z1  # First z-process removes bottom PML layers
    if iprocz == nprocz_out - 1:
        nk -= num_of_pml_z2  # Last z-process removes top PML layers
    # Distribute remainder points
    if iprocz < nz_left:
        nk += 1  # First nz_left processes get 1 extra point

    # Calculate global starting index (gnk1) for z-direction
    if iprocz == 0:
        gnk1 = 0  # First process starts at global index 0
    else:
        # Base offset minus bottom PML layers
        gnk1 = iprocz * nz_avg - num_of_pml_z1
    # Adjust for remainder points
    if nz_left != 0:
        gnk1 += iprocz if iprocz < nz_left else nz_left

    # Package return values (matches original C function's output)
    global_index[0] = gni1
    global_index[1] = gnk1
    count[0] = ni
    count[1] = nk


def coord_export_base_mpi(gdcurv: GridData, cfgs: dict) -> None:
    """
    Export curvature coordinate data into sharded netCDF files (per process partition).
    """
    # Number of processes in x-direction
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    # Number of processes in z-direction
    nprocz_out = cfgs['number_of_mpiprocs_out'][1] 

    total_nx = gdcurv.nx  # Total grid points in global x-direction
    total_nz = gdcurv.nz  # Total grid points in global z-direction

    os.makedirs(cfgs['grid_export_dir'], exist_ok=True)

    global_index = [0, 0]  
    count = [0, 0]   
    for kk in range(nprocz_out):  
        for ii in range(nprocx_out):  
            fname_coords = f"px{ii}_pz{kk}"  
            ou_file = os.path.join(cfgs['grid_export_dir'], f"coord_{fname_coords}.nc")  

            gd_info_set(cfgs, total_nx, total_nz, ii, kk, global_index, count)
            gni1, gnk1 = global_index  
            ni, nk = count             

            coord_x = np.zeros((nk, ni), dtype=np.float32)  
            coord_z = np.zeros((nk, ni), dtype=np.float32)  

            coord_x[:] = gdcurv.x2d[gnk1:gnk1+nk, gni1:gni1+ni]
            coord_z[:] = gdcurv.z2d[gnk1:gnk1+nk, gni1:gni1+ni]

            with nc.Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                ncid.createDimension('k', nk)
                ncid.createDimension('i', ni)

                x_var = ncid.createVariable('x', np.float32, ('k', 'i'))  
                z_var = ncid.createVariable('z', np.float32, ('k', 'i'))  

                ncid.setncattr(
                    "global_index_of_first_physical_points",
                    np.array(global_index, dtype=np.int32)  # [gni1, gnk1] as int32
                )
                ncid.setncattr(
                    "count_of_physical_points",
                    np.array(count, dtype=np.int32)  # [ni, nk] as int32
                )

                x_var[:] = coord_x
                z_var[:] = coord_z


def grid_quality_check_base_mpi(gdcurv: 'GridData', cfgs: dict) -> None:
    """Perform grid quality checks"""
    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    total_nx = gdcurv.nx
    total_nz = gdcurv.nz

    var = np.zeros((total_nz, total_nx), dtype=np.float32)
    if cfgs['check_orth'] == 1:
        quality_name = "orth"
        cal_orth(var, x2d, z2d, total_nx, total_nz)
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_jac'] == 1:
        quality_name = "jacobi"
        cal_jacobi(var, x2d, z2d, total_nx, total_nz )
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_ratio'] == 1:
        quality_name = "ratio"
        cal_ratio(var, x2d, z2d, total_nx, total_nz )
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_step_xi'] == 1:
        quality_name = "step_xi"
        cal_step_x(var, x2d, z2d, total_nx)
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_step_zt'] == 1:
        quality_name = "step_zt"
        cal_step_z(var, x2d, z2d, total_nz)
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_smooth_xi'] == 1:
        quality_name = "smooth_xi"
        cal_smooth_x(var, x2d, z2d, total_nx)
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)
    
    if cfgs['check_smooth_zt'] == 1:
        quality_name = "smooth_zt"
        cal_smooth_z(var, x2d, z2d, total_nz)
        quality_export_base_mpi(var, cfgs, total_nx, total_nz, quality_name)


def quality_export_base_mpi(var_in: np.ndarray, cfgs: dict, total_nx: int,
                            total_nz: int, var_name: str) -> None:
    """
    Export curvature coordinate data into sharded netCDF files (per process partition).
    """
    # Number of processes in x-direction
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    # Number of processes in z-direction
    nprocz_out = cfgs['number_of_mpiprocs_out'][1] 

    os.makedirs(cfgs['grid_export_dir'], exist_ok=True)

    global_index = [0, 0]  
    count = [0, 0]   
    for kk in range(nprocz_out):  
        for ii in range(nprocx_out):  
            fname_coords = f"px{ii}_pz{kk}"  
            ou_file = os.path.join(cfgs['grid_export_dir'], f"{var_name}_{fname_coords}.nc")  

            gd_info_set(cfgs, total_nx, total_nz, ii, kk, global_index, count)
            gni1, gnk1 = global_index  
            ni, nk = count             

            var = np.zeros((nk, ni), dtype=np.float32)  

            var[:] = var_in[gnk1:gnk1+nk, gni1:gni1+ni]

            with nc.Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                ncid.createDimension('k', nk)
                ncid.createDimension('i', ni)

                var_out = ncid.createVariable('x', np.float32, ('k', 'i'))  

                ncid.setncattr(
                    "global_index_of_first_physical_points",
                    np.array(global_index, dtype=np.int32)  # [gni1, gnk1] as int32
                )
                ncid.setncattr(
                    "count_of_physical_points",
                    np.array(count, dtype=np.int32)  # [ni, nk] as int32
                )

                var_out[:] = var