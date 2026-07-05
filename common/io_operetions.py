from netCDF4 import Dataset
import os
import numpy as np


def coord_export(gdcurv: 'GridData', output_dir: str) -> None:
    """
    Export grid coordinates to NetCDF file
    """

    x2d = gdcurv.x2d
    z2d = gdcurv.z2d
    ni = gdcurv.ni
    nk = gdcurv.nk
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    fname_part = gdcurv.fname_part
    
    # Create output filename
    ou_file = os.path.join(output_dir, f"coord_{fname_part}.nc")
    
    # Create NetCDF file
    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        # Define dimensions
        nc.createDimension('k', nk)
        nc.createDimension('i', ni)
        
        # Create variables
        x_var = nc.createVariable('x', 'f4', ('k', 'i'))
        z_var = nc.createVariable('z', 'f4', ('k', 'i'))
        
        # Add global attributes
        nc.global_index_of_first_physical_points = [gni1, gnk1]
        nc.count_of_physical_points = [ni, nk]
        
        # Write data
        x_var[:, :] = x2d
        z_var[:, :] = z2d


def quality_export(gdcurv: 'GridData', var: np.ndarray, 
                   output_dir: str, var_name: str) -> None:
    """
    Export quality data to NetCDF file
    """
    ni = gdcurv.ni
    nk = gdcurv.nk
    gni1 = gdcurv.gni1
    gnk1 = gdcurv.gnk1
    fname_part = gdcurv.fname_part
    
    # Create output filename
    ou_file = os.path.join(output_dir, f"{var_name}_{fname_part}.nc")
    
    # Create NetCDF file
    with Dataset(ou_file, 'w', format='NETCDF4') as nc:
        # Define dimensions
        nc.createDimension('k', nk)
        nc.createDimension('i', ni)
        
        # Create variable
        var_out = nc.createVariable(var_name, 'f4', ('k', 'i'))
        
        # Add global attributes
        nc.global_index_of_first_physical_points = [gni1, gnk1]
        nc.count_of_physical_points = [ni, nk]
        
        # Write data
        var_out[:, :] = var