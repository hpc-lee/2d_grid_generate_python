import json
import numpy as np
import matplotlib.pyplot as plt
import os
from draw_grid_func import locate_coord, gather_coord, gather_quality


# -------------------------- parameters input -------------------------- %
# file and path name
cfs_file = '../elliptic/output/config.json'
#cfs_file = '../hyperbolic/output/config.json'
#cfs_file = '../parabolic/output/config.json'
# varable to plot 
# 'orth', 'jacobi', 'ratio', 'smooth_xi', 
# 'smooth_zt', 'step_xi', 'step_zt'
#varnm = 'jacobi'
varnm = 'step_zt'
# which grid profile to plot
subs = [0, 0]  # Start from index 0 in both x and z dimensions
subc = [-1, -1]   # '-1' to plot all points in this dimension (0-based interpretation)
subt = [1, 1]
# figure control parameters
flag_km = 1
flag_print = 1
flag_title = 0


# Check parameter file exists
if not os.path.exists(cfs_file):
    raise FileNotFoundError(f"locate_coord: file {cfs_file} does not exist")

# Read parameters file
with open(cfs_file, 'r') as f:
    cfgs = json.load(f)

# -----------------------------------------------------------
# -- load coord
# -----------------------------------------------------------
coordinfo = locate_coord(cfgs, subs, subc, subt)
x, z = gather_coord(coordinfo, cfgs['grid_export_dir'])
var_data = gather_quality(coordinfo, cfgs['grid_export_dir'], varnm)

print(f"{varnm} max value is {np.max(var_data)}")
print(f"{varnm} min value is {np.min(var_data)}")

# - set coord unit
if flag_km:
    x = x / 1e3
    z = z / 1e3
    str_unit = 'km'
else:
    str_unit = 'm'

# -----------------------------------------------------------
# -- set figure
# -----------------------------------------------------------
fig, ax = plt.subplots()
fig.set_facecolor('white')
plt.set_cmap('jet')

im = ax.pcolormesh(x, z, var_data, shading='gouraud')

ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)

ax.set_axis_on()
ax.tick_params(labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')

ax.set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, ax=ax)


if flag_title:
    gridtitle = 'XOZ-Grid'
    ax.set_title(gridtitle)

plt.tight_layout()

plt.show()

if flag_print:
    plt.savefig('quality.png', dpi=300, bbox_inches='tight', facecolor='white')
    #plt.savefig('quality.svg', format='svg', dpi=600, bbox_inches='tight', facecolor='white')
