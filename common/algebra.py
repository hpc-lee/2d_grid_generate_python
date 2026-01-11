import numpy as np
from typing import Optional

def zt_arc_stretch(gdcurv: 'GridData', 
                   arc_len: Optional[np.ndarray] = None) -> None:
    """
    Arc length stretching in z-direction
    """
    nx = gdcurv.nx
    nz = gdcurv.nz
    x2d = gdcurv.x2d  # shape: (nz, nx)
    z2d = gdcurv.z2d  # shape: (nz, nx)

    if arc_len is None:
        arc_len = np.linspace(0, 1, nz, dtype=np.float32)
    
    # Pre-allocate temporary arrays
    x2d_temp = np.zeros(nz, dtype=np.float32)
    z2d_temp = np.zeros(nz, dtype=np.float32)
    s = np.zeros(nz, dtype=np.float32)
    u = np.zeros(nz, dtype=np.float32)
    
    # Process each column (i=0 to nx-1) - vectorized approach
    for i in range(nx):
        # Copy old coordinates to temp space
        x2d_temp[:] = x2d[:, i]
        z2d_temp[:] = z2d[:, i]
        
        # Calculate arc length (vectorized)
        x_diff = np.diff(x2d_temp)  # x2d_temp[1:] - x2d_temp[:-1]
        z_diff = np.diff(z2d_temp)  # z2d_temp[1:] - z2d_temp[:-1]
        dh_len = np.sqrt(x_diff**2 + z_diff**2)
        s[1:] = np.cumsum(dh_len)  # cumulative sum instead of loop
        
        # Normalize arc length
        u[:] = s / s[nz-1]
        
        # Vectorized interpolation for k in range(1, nz-1)
        # Use searchsorted to find intervals efficiently
        arc_targets = arc_len[1:nz-1]  # targets to interpolate to
        
        # Find which intervals contain each target
        indices = np.searchsorted(u, arc_targets, side='right') - 1
        indices = np.clip(indices, 0, nz-2)  # ensure valid bounds
        
        # Get the corresponding u values for interpolation
        u_left = u[indices]
        u_right = u[indices + 1]
        
        # Calculate interpolation ratios
        ratios = (arc_targets - u_left) / (u_right - u_left)
        # Handle division by zero
        mask = (u_right - u_left) != 0
        ratios = np.where(mask, ratios, 0.0)
        
        # Get corresponding coordinate values
        x_left = x2d_temp[indices]
        x_right = x2d_temp[indices + 1]
        z_left = z2d_temp[indices]
        z_right = z2d_temp[indices + 1]
        
        # Perform linear interpolation
        x_interp = x_left + ratios * (x_right - x_left)
        z_interp = z_left + ratios * (z_right - z_left)
        
        # Update the result arrays
        x2d[1:nz-1, i] = x_interp
        z2d[1:nz-1, i] = z_interp


def xi_arc_stretch(gdcurv: 'GridData', 
                   arc_len: Optional[np.ndarray] = None) -> None:
    """
    Arc length stretching in x-direction
    """
    nx = gdcurv.nx
    nz = gdcurv.nz
    x2d = gdcurv.x2d  # shape: (nz, nx)
    z2d = gdcurv.z2d  # shape: (nz, nx)

    if arc_len is None:
        arc_len = np.linspace(0, 1, nx, dtype=np.float32)
    
    # Pre-allocate temporary arrays
    x2d_temp = np.zeros(nx, dtype=np.float32)
    z2d_temp = np.zeros(nx, dtype=np.float32)
    s = np.zeros(nx, dtype=np.float32)
    u = np.zeros(nx, dtype=np.float32)
    
    # Process each row (k=0 to nz-1) - vectorized approach
    for k in range(nz):
        # Copy old coordinates to temp space
        x2d_temp[:] = x2d[k, :]  # x2d_temp[i] = x2d[k, i]
        z2d_temp[:] = z2d[k, :]  # z2d_temp[i] = z2d[k, i]
        
        # Calculate arc length (vectorized)
        x_diff = np.diff(x2d_temp)  # x2d_temp[1:] - x2d_temp[:-1]
        z_diff = np.diff(z2d_temp)  # z2d_temp[1:] - z2d_temp[:-1]
        dh_len = np.sqrt(x_diff**2 + z_diff**2)
        s[1:] = np.cumsum(dh_len)  # cumulative sum instead of loop
        
        # Normalize arc length
        u[:] = s / s[nx-1]
        
        # Vectorized interpolation for i in range(1, nx-1)
        # Use searchsorted to find intervals efficiently
        arc_targets = arc_len[1:nx-1]  # targets to interpolate to
        
        # Find which intervals contain each target
        indices = np.searchsorted(u, arc_targets, side='right') - 1
        indices = np.clip(indices, 0, nx-2)  # ensure valid bounds
        
        # Get the corresponding u values for interpolation
        u_left = u[indices]
        u_right = u[indices + 1]
        
        # Calculate interpolation ratios
        ratios = (arc_targets - u_left) / (u_right - u_left)
        # Handle division by zero
        mask = (u_right - u_left) != 0
        ratios = np.where(mask, ratios, 0.0)
        
        # Get corresponding coordinate values
        x_left = x2d_temp[indices]
        x_right = x2d_temp[indices + 1]
        z_left = z2d_temp[indices]
        z_right = z2d_temp[indices + 1]
        
        # Perform linear interpolation
        x_interp = x_left + ratios * (x_right - x_left)
        z_interp = z_left + ratios * (z_right - z_left)
        
        # Update the result arrays
        x2d[k, 1:nx-1] = x_interp
        z2d[k, 1:nx-1] = z_interp