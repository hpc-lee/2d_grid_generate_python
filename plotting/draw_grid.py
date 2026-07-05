import json
import matplotlib.pyplot as plt
import os
from draw_grid_func import locate_coord, gather_coord


# -------------------------- parameters input -------------------------- %
# file and path name
#cfs_file = '../elliptic/output/config.json'
#cfs_file = '../hyperbolic/output/config.json'
#cfs_file = '../parabolic/output/config.json'
cfs_file = '../grid_post_process/output/config.json'
# which grid profile to plot - NOW 1-BASED
subs = [0, 0]  # Start from index 0 in both x and z dimensions
subc = [-1, -1]   # '-1' to plot all points in this dimension (0-based interpretation)
subt = [4, 4]

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

ax.plot(x, z, 'k-', linewidth=0.1)
ax.plot(x.T, z.T, 'k-', linewidth=0.1)

ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)

ax.set_axis_on()
ax.tick_params(labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')

ax.set_aspect('equal', adjustable='box')

if flag_title:
    gridtitle = 'XOZ-Grid'
    ax.set_title(gridtitle)

plt.tight_layout()

plt.show()

if flag_print:
    #plt.savefig('grid.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('grid.svg', format='svg', dpi=600, bbox_inches='tight', facecolor='white')
