import os
import numpy as np
from netCDF4 import Dataset
import glob


def locate_coord(cfgs: dict, subs: list[int], subc: list[int],
                 subt: list[int]) -> list:
    """
    Locates coordinate NetCDF files based on parameters and spatial
    selection (0-based indexing).

    Args:
        cfgs (dict): JSON parameter dict.
        subs (list): Starting indices for x and z dimensions (0-based).
        subc (list): Count of points to select for x and z dimensions (-1 for
                     all).
        subt (list): Stride for x and z dimensions.

    Returns:
        list: A list of dictionaries containing information about each
        relevant NetCDF file.
    """

    ngik = [cfgs['number_of_grid_points_x'], cfgs['number_of_grid_points_z']]

    gsubs = subs.copy()
    gsubt = subt.copy()
    gsubc = subc.copy()

    # Reset count=-1 to total number (interpret total_points as 1-based max,
    # so count is max - start)
    # Note:ceil, due to count = number of intervals + 1
    for idx, val in enumerate(gsubc):
        if val == -1:
            gsubc[idx] = int(np.ceil((ngik[idx] - gsubs[idx]) / gsubt[idx]))
    gsube = [gsubs[0] + (gsubc[0] - 1) * gsubt[0],
             gsubs[1] + (gsubc[1] - 1) * gsubt[1]]

    # Search the nc file headers to locate the threads/processors
    coordprefix = 'coord'
    # Use glob to find files matching the pattern
    coord_pattern = os.path.join(cfgs['grid_export_dir'], f"{coordprefix}*.nc")
    coord_files = glob.glob(coord_pattern)
    
    px = []
    pz = []
    
    for coordnm in coord_files:
        with Dataset(coordnm, 'r') as nc_file:
            xzs = nc_file.getncattr('global_index_of_first_physical_points')
            xs, zs = int(xzs[0]), int(xzs[1])
            xzc = nc_file.getncattr('count_of_physical_points')
            xc, zc = int(xzc[0]), int(xzc[1])
        
        xarray = np.arange(xs, xs + xc - 1)
        zarray = np.arange(zs, zs + zc - 1)

        x_in_range = np.any((xarray >= gsubs[0]) & (xarray <= gsube[0]))
        z_in_range = np.any((zarray >= gsubs[1]) & (zarray <= gsube[1]))
        if (x_in_range and z_in_range):
            filename = os.path.basename(coordnm)
            px_match = filename.find('px') + 2
            pz_match = filename.find('_pz')
            px_val = int(filename[px_match:pz_match])
            
            pz_match_start = filename.find('pz') + 2
            nc_match = filename.find('.nc')
            pz_val = int(filename[pz_match_start:nc_match])
            
            px.append(px_val)
            pz.append(pz_val)

    coordinfo = []
    for ip in range(len(px)):
        coordnm = os.path.join(cfgs['grid_export_dir'],
                               f"{coordprefix}_px{px[ip]}_pz{pz[ip]}.nc")
        
        with Dataset(coordnm, 'r') as nc_file:
            xzs = nc_file.getncattr('global_index_of_first_physical_points')
            xs, zs = int(xzs[0]), int(xzs[1])
            xzc = nc_file.getncattr('count_of_physical_points')
            xc, zc = int(xzc[0]), int(xzc[1])
        
        xe = xs + xc - 1
        ze = zs + zc - 1
        
        # Note: end in gsube, need + 1
        gxarray = np.arange(gsubs[0], gsube[0] + 1, gsubt[0])
        gzarray = np.arange(gsubs[1], gsube[1] + 1, gsubt[1])
        
        i_mask = (gxarray >= xs) & (gxarray <= xe)
        k_mask = (gzarray >= zs) & (gzarray <= ze)
        i_indices = np.where(i_mask)[0]  # i_indices is the index into gxarray
        k_indices = np.where(k_mask)[0]  # k_indices is the index into gzarray
        
        if len(i_indices) > 0 and len(k_indices) > 0:
            coord_dict = {}
            coord_dict['thisid'] = [px[ip], pz[ip]]
            coord_dict['indxs'] = [int(i_indices[0]), int(k_indices[0])]
            coord_dict['indxe'] = [int(i_indices[-1]), int(k_indices[-1])]
            coord_dict['indxc'] = [coord_dict['indxe'][0] - coord_dict['indxs'][0] + 1,
                                   coord_dict['indxe'][1] - coord_dict['indxs'][1] + 1]
            
            coord_dict['subs'] = [int(gxarray[i_indices[0]] - xs),
                                  int(gzarray[k_indices[0]] - zs)]
            coord_dict['subc'] = coord_dict['indxc'].copy() 
            coord_dict['subt'] = gsubt.copy()
            
            coord_dict['fnmprefix'] = coordprefix
            
            coordinfo.append(coord_dict)

    return coordinfo


def gather_coord(coordinfo: list, output_dir: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gathers coordinate data from multiple NetCDF files specified in coordinfo

    Args:
        coordinfo (list): List of dictionaries from locate_coord (0-based).
        output_dir (str): Directory containing the NetCDF files.

    Returns:
        tuple: Two numpy arrays (x, z) containing the gathered coordinates.
    """
    # Determine the total size of the final coordinate arrays
    # The output array size is determined by the highest index in coordinfo + 1
    max_i = 0
    max_k = 0
    for info in coordinfo:
        max_i = max(max_i, info['indxe'][0])
        max_k = max(max_k, info['indxe'][1])
    x = np.zeros((max_k + 1, max_i + 1))
    z = np.zeros((max_k + 1, max_i + 1))

    # Load coordinates from each relevant file
    for info in coordinfo:
        n_i, n_k = info['thisid']
        i1, k1 = info['indxs']
        i2, k2 = info['indxe']
        subs = info['subs']
        subc = info['subc']
        subt = info['subt']
        
        fnm_coord = os.path.join(output_dir,
                                 f"{info['fnmprefix']}_px{n_i}_pz{n_k}.nc")
        
        if not os.path.exists(fnm_coord):
            raise FileNotFoundError(f"gather_coord: file {fnm_coord} does not exist")

        with Dataset(fnm_coord, 'r') as nc_file:
            start_x = subs[0]
            start_z = subs[1]
            # Note: end need add 1, due to python feature
            stop_x_py = start_x + (subc[0] - 1) * subt[0] + 1
            stop_z_py = start_z + (subc[1] - 1) * subt[1] + 1
            step_x_py = subt[0]
            step_z_py = subt[1]

            # Apply the slice to read the data
            x_data = nc_file['x'][start_z:stop_z_py:step_z_py,
                                  start_x:stop_x_py:step_x_py]
            z_data = nc_file['z'][start_z:stop_z_py:step_z_py,
                                  start_x:stop_x_py:step_x_py]

            x[k1:k2+1, i1:i2+1] = x_data
            z[k1:k2+1, i1:i2+1] = z_data

    return x, z

def gather_coord(coordinfo: list, output_dir: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gathers coordinate data from multiple NetCDF files specified in coordinfo

    Args:
        coordinfo (list): List of dictionaries from locate_coord (0-based).
        output_dir (str): Directory containing the NetCDF files.

    Returns:
        tuple: Two numpy arrays (x, z) containing the gathered coordinates.
    """
    # Determine the total size of the final coordinate arrays
    # The output array size is determined by the highest index in coordinfo + 1
    max_i = 0
    max_k = 0
    for info in coordinfo:
        max_i = max(max_i, info['indxe'][0])
        max_k = max(max_k, info['indxe'][1])
    x = np.zeros((max_k + 1, max_i + 1))
    z = np.zeros((max_k + 1, max_i + 1))

    # Load coordinates from each relevant file
    for info in coordinfo:
        n_i, n_k = info['thisid']
        i1, k1 = info['indxs']
        i2, k2 = info['indxe']
        subs = info['subs']
        subc = info['subc']
        subt = info['subt']
        
        fnm_coord = os.path.join(output_dir,
                                 f"{info['fnmprefix']}_px{n_i}_pz{n_k}.nc")
        
        if not os.path.exists(fnm_coord):
            raise FileNotFoundError(f"gather_coord: file {fnm_coord} does not exist")

        with Dataset(fnm_coord, 'r') as nc_file:
            start_x = subs[0]
            start_z = subs[1]
            # Note: end need add 1, due to python feature
            stop_x_py = start_x + (subc[0] - 1) * subt[0] + 1
            stop_z_py = start_z + (subc[1] - 1) * subt[1] + 1
            step_x_py = subt[0]
            step_z_py = subt[1]

            # Apply the slice to read the data
            x_data = nc_file['x'][start_z:stop_z_py:step_z_py,
                                  start_x:stop_x_py:step_x_py]
            z_data = nc_file['z'][start_z:stop_z_py:step_z_py,
                                  start_x:stop_x_py:step_x_py]

            x[k1:k2+1, i1:i2+1] = x_data
            z[k1:k2+1, i1:i2+1] = z_data

    return x, z

def gather_quality(coordinfo: list, output_dir: str, varnm: str
                 ) -> np.ndarray:
    """
    Gathers quality of data from multiple NetCDF files specified in coordinfo

    Args:
        coordinfo (list): List of dictionaries from locate_coord (0-based).
        output_dir (str): Directory containing the NetCDF files.
        varnm (str): quality name

    Returns:
        np.ndarray
    """
    # Determine the total size of the final coordinate arrays
    # The output array size is determined by the highest index in coordinfo + 1
    max_i = 0
    max_k = 0
    for info in coordinfo:
        max_i = max(max_i, info['indxe'][0])
        max_k = max(max_k, info['indxe'][1])
    var_data = np.zeros((max_k + 1, max_i + 1))

    # Load coordinates from each relevant file
    for info in coordinfo:
        n_i, n_k = info['thisid']
        i1, k1 = info['indxs']
        i2, k2 = info['indxe']
        subs = info['subs']
        subc = info['subc']
        subt = info['subt']
        
        fnm_var = os.path.join(output_dir,
                                 f"{varnm}_px{n_i}_pz{n_k}.nc")
        
        if not os.path.exists(fnm_var):
            raise FileNotFoundError(f"gather_coord: file {fnm_var} does not exist")

        with Dataset(fnm_var, 'r') as nc_file:
            start_x = subs[0]
            start_z = subs[1]
            # Note: end need add 1, due to python feature
            stop_x_py = start_x + (subc[0] - 1) * subt[0] + 1
            stop_z_py = start_z + (subc[1] - 1) * subt[1] + 1
            step_x_py = subt[0]
            step_z_py = subt[1]

            # Apply the slice to read the data
            var = nc_file[varnm][start_z:stop_z_py:step_z_py,
                                  start_x:stop_x_py:step_x_py]

            var_data[k1:k2+1, i1:i2+1] = var

    return var_data
