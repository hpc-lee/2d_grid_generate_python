"""
    ^ z
    |           bz2
    |       -----------------
    |       |                |
    |   bx1 |                | bx2
    |       |                | 
    |       |                |
    |       |                |
    |       -----------------
    |           bz1
    +---------------------> x
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

flag_printf = 1
flag_topo = 1
flag_extend_abs = 1
flag_arc_strech = 0

file_dir = "./data_file_2d.txt"

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0

nx1 = 801
nx = nx1 + 2 * num_pml
nz = 601

if flag_arc_strech:
    A = -0.0001

dx = 50.0
dz = 50.0
origin_x = 0.0
origin_z = 0.0

bz1 = np.zeros((nx, 2))
bz2 = np.zeros((nx, 2))

for i in range(nx1):
    idx = i + num_pml
    bz1[idx, 0] = origin_x + i * dx
    bz1[idx, 1] = origin_z - (nz - 1) * dz

    bz2[idx, 0] = origin_x + i * dx
    bz2[idx, 1] = origin_z

if flag_topo:
    x0 = 10.0 * 1e3
    x1 = 30.0 * 1e3
    a = 3.0 * 1e3
    H = 6.0 * 1e3

    for i in range(nx1):
        idx = i + num_pml
        x = idx * dx
        topo = (H * np.exp(-((x - x0)**2) / (a**2)) -
                H * np.exp(-((x - x1)**2) / (a**2)))
        bz2[idx, 1] = bz2[idx, 1] + topo

bz1 = extend_abs_layer(bz1, dx, nx, num_pml)
bz2 = extend_abs_layer(bz2, dx, nx, num_pml)


if flag_arc_strech:
    bz1 = arc_strech(A, bz1)
    bz2 = arc_strech(A, bz2)

plt.figure(1)
plt.plot(bz1[:, 0], bz1[:, 1], 'b', label='Bottom Boundary')
plt.plot(bz2[:, 0], bz2[:, 1], 'r', label='Top Boundary')
plt.axis('equal')
plt.tight_layout()
plt.xlabel('X axis (m)')
plt.ylabel('Z axis (m)')
plt.gca().set_facecolor('white')
plt.legend()
plt.show()

# export_bdry;
export_bdry(bz1, bz2, nx, file_dir)

if flag_printf:
    plt.savefig('model1.png', dpi=300, bbox_inches='tight', facecolor='white')

print(f"nx is {nx}")
print(f"nz is {nz}")
print("Script execution completed.")
