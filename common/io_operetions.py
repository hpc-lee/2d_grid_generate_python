from netCDF4 import Dataset
import os
import sys
import numpy as np


def coord_export(gdcurv: 'GridData', output_dir: str) -> None:
    """Export grid coordinates to single NetCDF file (single-partition mode)."""

    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    ni = gdcurv.ni
    nk = gdcurv.nk
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    fname_part = gdcurv.fname_part

    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")

    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('i', ni)

        x_var = nc.createVariable('x', 'f4', ('k', 'i'))
        z_var = nc.createVariable('z', 'f4', ('k', 'i'))

        nc.global_index_of_first_physical_points = [gni1, gnk1]
        nc.count_of_physical_points = [ni, nk]

        x_var[:, :] = x2d
        z_var[:, :] = z2d


def quality_export(gdcurv: 'GridData', var: np.ndarray,
                   output_dir: str, var_name: str) -> None:
    """Export quality data to single NetCDF file (single-partition mode)."""

    ni = gdcurv.ni
    nk = gdcurv.nk
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    fname_part = gdcurv.fname_part

    ou_file = os.path.join(output_dir, f"{var_name}_{fname_part}.nc")

    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('k', nk)
        nc.createDimension('i', ni)

        var_out = nc.createVariable(var_name, 'f4', ('k', 'i'))

        nc.global_index_of_first_physical_points = [gni1, gnk1]
        nc.count_of_physical_points = [ni, nk]

        var_out[:, :] = var


# ---------------------------------------------------------------------------
# MPI-partitioned export functions (for grid_post_process with PML support)
# ---------------------------------------------------------------------------

def gd_info_set(cfgs: dict, total_nx: int, total_nz: int,
                iprocx: int, iprocz: int, global_index: list,
                count: list) -> None:
    """Calculate global starting indices and data dimensions for a specific process."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocz_out = cfgs['number_of_mpiprocs_out'][1]
    if cfgs.get('pml_weight_2x', 0) == 1:
        num_of_pml_x1 = cfgs['number_of_pml_x1']
        num_of_pml_x2 = cfgs['number_of_pml_x2']
        num_of_pml_z1 = cfgs['number_of_pml_z1']
        num_of_pml_z2 = cfgs['number_of_pml_z2']
    else:
        num_of_pml_x1 = num_of_pml_x2 = num_of_pml_z1 = num_of_pml_z2 = 0

    # X-direction partitioning
    nx_et = total_nx + num_of_pml_x1 + num_of_pml_x2
    nx_avg = nx_et // nprocx_out
    nx_left = nx_et % nprocx_out

    if nx_avg <= num_of_pml_x1 or nx_avg <= num_of_pml_x2:
        print("Error: nx_avg must be larger than PML layer counts")
        print(f"nx_avg={nx_avg}, PML x1={num_of_pml_x1}, PML x2={num_of_pml_x2}")
        sys.exit(1)

    ni = nx_avg
    if iprocx == 0:
        ni -= num_of_pml_x1
    if iprocx == nprocx_out - 1:
        ni -= num_of_pml_x2
    if iprocx < nx_left:
        ni += 1

    if iprocx == 0:
        gni1 = 0
    else:
        gni1 = iprocx * nx_avg - num_of_pml_x1
    if nx_left != 0:
        gni1 += iprocx if iprocx < nx_left else nx_left

    # Z-direction partitioning
    nz_et = total_nz + num_of_pml_z1 + num_of_pml_z2
    nz_avg = nz_et // nprocz_out
    nz_left = nz_et % nprocz_out

    if nz_avg <= num_of_pml_z1 or nz_avg <= num_of_pml_z2:
        print("Error: nz_avg must be larger than PML layer counts")
        print(f"nz_avg={nz_avg}, PML z1={num_of_pml_z1}, PML z2={num_of_pml_z2}")
        sys.exit(1)

    nk = nz_avg
    if iprocz == 0:
        nk -= num_of_pml_z1
    if iprocz == nprocz_out - 1:
        nk -= num_of_pml_z2
    if iprocz < nz_left:
        nk += 1

    if iprocz == 0:
        gnk1 = 0
    else:
        gnk1 = iprocz * nz_avg - num_of_pml_z1
    if nz_left != 0:
        gnk1 += iprocz if iprocz < nz_left else nz_left

    global_index[0] = gni1
    global_index[1] = gnk1
    count[0] = ni
    count[1] = nk


def coord_export_mpi(gdcurv: 'GridData', cfgs: dict) -> None:
    """Export grid coordinates to MPI-partitioned NetCDF files."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocz_out = cfgs['number_of_mpiprocs_out'][1]
    total_nx = gdcurv.nx
    total_nz = gdcurv.nz

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

            coord_x = gdcurv.x2d[gnk1:gnk1+nk, gni1:gni1+ni].astype(np.float32)
            coord_z = gdcurv.z2d[gnk1:gnk1+nk, gni1:gni1+ni].astype(np.float32)

            with Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                ncid.createDimension('k', nk)
                ncid.createDimension('i', ni)

                x_var = ncid.createVariable('x', np.float32, ('k', 'i'))
                z_var = ncid.createVariable('z', np.float32, ('k', 'i'))

                ncid.setncattr("global_index_of_first_physical_points",
                               np.array(global_index, dtype=np.int32))
                ncid.setncattr("count_of_physical_points",
                               np.array(count, dtype=np.int32))

                x_var[:] = coord_x
                z_var[:] = coord_z


def quality_export_mpi(var_in: np.ndarray, gdcurv: 'GridData',
                       cfgs: dict, var_name: str) -> None:
    """Export quality data to MPI-partitioned NetCDF files."""
    nprocx_out = cfgs['number_of_mpiprocs_out'][0]
    nprocz_out = cfgs['number_of_mpiprocs_out'][1]
    total_nx = gdcurv.nx
    total_nz = gdcurv.nz

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

            var = var_in[gnk1:gnk1+nk, gni1:gni1+ni].astype(np.float32)

            with Dataset(ou_file, 'w', format='NETCDF4') as ncid:
                ncid.createDimension('k', nk)
                ncid.createDimension('i', ni)

                var_out = ncid.createVariable(var_name, np.float32, ('k', 'i'))

                ncid.setncattr("global_index_of_first_physical_points",
                               np.array(global_index, dtype=np.int32))
                ncid.setncattr("count_of_physical_points",
                               np.array(count, dtype=np.int32))

                var_out[:] = var
