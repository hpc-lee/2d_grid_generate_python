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
flag_topo_z = 1
flag_extend_abs = 0
flag_arc_strech = 0
file_dir = "./data_file_2d.txt"

if flag_arc_strech:
    A = -0.0001

if flag_extend_abs:
    num_pml = 20
else:
    num_pml = 0


nx1 = 801
nx = nx1 + 2 * num_pml
nz = 601

dx = 50
dz = 50
origin_x = 0
origin_z = 0

# Initialize arrays with zeros
bz1 = np.zeros((nx, 2))
bz2 = np.zeros((nx, 2))
bx1 = np.zeros((nz, 2))
bx2 = np.zeros((nz, 2))
i_indices = np.arange(nx1)
bz1[num_pml:num_pml+nx1, 0] = origin_x + i_indices * dx
bz1[num_pml:num_pml+nx1, 1] = origin_z - (nz - 1) * dz
bz2[num_pml:num_pml+nx1, 0] = origin_x + i_indices * dx
bz2[num_pml:num_pml+nx1, 1] = origin_z

if flag_topo_z:
    x0 = 10.0 * 1e3
    x1 = 30.0 * 1e3
    a = 3.0 * 1e3
    H = 3.0 * 1e3
    
    for i in range(nx1):
        idx = i + num_pml
        x = idx * dx
        topo = (H * np.exp(-((x - x0)**2) / (a**2)) -
                H * np.exp(-((x - x1)**2) / (a**2)))
        bz2[idx, 1] = bz2[idx, 1] + topo

bz1 = extend_abs_layer(bz1, dx, nx, num_pml)
bz2 = extend_abs_layer(bz2, dx, nx, num_pml)

if flag_arc_strech:
    bz2 = arc_strech(A, bz2)

dz1 = (bz2[0, 1] - bz1[0, 1]) / (nz - 1)
dz2 = (bz2[nx-1, 1] - bz1[nx-1, 1]) / (nz - 1)

# Fill bx1 and bx2 arrays
for k in range(nz):
    bx1[k, 0] = bz1[0, 0]
    bx1[k, 1] = bz1[0, 1] + k * dz1
    
    bx2[k, 0] = bz1[nx-1, 0]
    bx2[k, 1] = bz1[nx-1, 1] + k * dz2

plt.figure(1)
plt.plot(bx1[:, 0], bx1[:, 1])
plt.plot(bx2[:, 0], bx2[:, 1])
plt.plot(bz1[:, 0], bz1[:, 1])
plt.plot(bz2[:, 0], bz2[:, 1])
plt.axis('equal')
plt.tight_layout()
plt.xlabel('X axis (m)')
plt.ylabel('Z axis (m)')
plt.gca().set_facecolor('white')
plt.show()

# export_bdry;
export_bdry(bx1, bx2, bz1, bz2, nx, nz, file_dir)

if flag_printf:
    plt.savefig('model1.png', dpi=300, bbox_inches='tight', facecolor='white')

print(f"nx is {nx}")
print(f"nz is {nz}")
print("Script execution completed.")
