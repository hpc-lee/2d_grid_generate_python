"""
    To keep the code simple, the parabolic grid generation
    method typically only supports advancing along the
    z-direction to generate grids. If grids need to be 
    generated along the x-direction based on given boundaries
    bx1 and bx2, they can first be converted to boundaries bz1
    and bz2 along the z-direction for grid generation, and then
    converted back after post-processing. This situation is rare
    because the free surface is typically defined as boundary bz2.
    This coordinate transformation corresponds to a reflection
    about the line y = x.

    ^ z(x)
    |           bz2
    |       -----------------
    |       |                |
    |   bx1 |                | bx2
    |  (bz1)|                |(bz2) 
    |       |                |
    |       |                |
    |       -----------------
    |           bz1
    +---------------------> x(z)
    (0,0)      
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.bdry_operations import extend_abs_layer, arc_strech
from export import export_bdry

# The first step is to provide the boundary definitions bx1 and bx2.
# Then, this script will transform them into bz1 and bz2
flag_printf = 1
flag_topo = 1
flag_extend_abs = 1
flag_arc_strech = 0

file_dir = "./data_file_2d.txt"

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0

nz1_ori = 801
nz_ori = nz1_ori + 2 * num_pml
nx_ori = 601

if flag_arc_strech:
    A = -0.0001

dx_ori = 50.0
dz_ori = 50.0
origin_x = 0.0
origin_z = 0.0

bx1 = np.zeros((nz_ori, 2))
bx2 = np.zeros((nz_ori, 2))

for k in range(nz1_ori):
    idx = k + num_pml
    bx1[idx, 0] = origin_x 
    bx1[idx, 1] = origin_z - (nz1_ori-1-k) *  dz_ori

    bx2[idx, 0] = origin_x + (nx_ori-1) * dx_ori
    bx2[idx, 1] = origin_z - (nz1_ori-1-k) *  dz_ori

if flag_topo:
    z0 = 10.0 * 1e3
    z1 = 30.0 * 1e3
    a = 3.0 * 1e3
    H = 6.0 * 1e3

    for k in range(nz1_ori):
        idx = k + num_pml
        z = idx * dz_ori
        topo = (H * np.exp(-((z - z0)**2) / (a**2)) -
                H * np.exp(-((z - z1)**2) / (a**2)))
        bx2[idx, 0] = bx2[idx, 0] + topo

# Step 2: Transform the coordinates of boundaries bx1 and bx2 
# (defined in Step 1) into bz1 and bz2.
# nx and nz in config.json are the post-transformation values
nx = nz_ori
nz = nx_ori
dx = dz_ori
dz = dx_ori

bz1 = np.zeros((nx, 2))
bz2 = np.zeros((nx, 2))

# coords transform x->z z->x
bz1[num_pml:nx-num_pml,0] = bx1[num_pml:nx-num_pml,1]
bz1[num_pml:nx-num_pml,1] = bx1[num_pml:nx-num_pml,0]
bz2[num_pml:nx-num_pml,0] = bx2[num_pml:nx-num_pml,1]
bz2[num_pml:nx-num_pml,1] = bx2[num_pml:nx-num_pml,0]

bz1 = extend_abs_layer(bz1, dx, nx, num_pml)
bz2 = extend_abs_layer(bz2, dx, nx, num_pml)

if flag_arc_strech:
    bz1 = arc_strech(A, bz1)
    bz2 = arc_strech(A, bz2)

bx1[:num_pml,0] = bz1[:num_pml,1]
bx1[:num_pml,1] = bz1[:num_pml,0]
bx1[nx-num_pml:,0] = bz1[nx-num_pml:,1]
bx1[nx-num_pml:,1] = bz1[nx-num_pml:,0]

bx2[:num_pml,0] = bz2[:num_pml,1]
bx2[:num_pml,1] = bz2[:num_pml,0]
bx2[nx-num_pml:,0] = bz2[nx-num_pml:,1]
bx2[nx-num_pml:,1] = bz2[nx-num_pml:,0]


# plot figure to show the trans
x = np.zeros(2*nx)
y = np.zeros(2*nx)
for i in range(2*nx):
    y[i] = x[i] = (i-nx) * dx

plt.figure(1)
plt.plot(bz1[:, 0], bz1[:, 1], 'b', label='Bottom Boundary')
plt.plot(bz2[:, 0], bz2[:, 1], 'r', label='Top Boundary')
plt.plot(bx1[:, 0], bx1[:, 1], 'b', label='Left Boundary')
plt.plot(bx2[:, 0], bx2[:, 1], 'r', label='Right Boundary')
plt.plot(x, y, 'm')
plt.axis('equal')
plt.tight_layout()
plt.xlabel('X axis (m)')
plt.ylabel('Z axis (m)')
plt.gca().set_facecolor('white')
plt.legend(fontsize=7, loc='best')
plt.show()

# export_bdry;
export_bdry(bz1, bz2, nx, file_dir)

if flag_printf:
    plt.savefig('model1.png', dpi=300, bbox_inches='tight', facecolor='white')

print(f"nx is {nx}")
print(f"nz is {nz}")
print("Script execution completed.")
