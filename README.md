# 2D Curvilinear Grid Generation (Python)

Python implementation of 2D curvilinear grid generation for seismic wave numerical simulation. Provides three grid generation methods — parabolic, hyperbolic, and elliptic — with grid quality evaluation, post-processing, and optional C-backend acceleration.

## Project Structure

```
2d_grid_generate_python/
├── common/                  # Shared modules
│   ├── algebra.py           # Arc-length stretching functions
│   ├── bdry_operations.py   # Boundary operations
│   ├── grid_data.py         # SimpleGridData class (non-MPI)
│   ├── grid_math.py         # Matrix operations
│   ├── grid_quality.py      # Grid quality metrics & cal_min_dist
│   ├── io_operetions.py     # NetCDF export (single & MPI-partitioned)
│   └── utils.py             # Utility functions (comment filtering, quality print)
├── parabolic/               # Parabolic grid generation
│   ├── generate_grid/       #   para.py, grid_data.py, grid_generation.py
│   ├── prep_grid_bdry/      #   Boundary data files
│   └── output/              #   Output directory
├── hyperbolic/              # Hyperbolic grid generation
│   ├── generate_grid/       #   hyper.py, grid_data.py, grid_generation.py
│   ├── prep_grid_bdry/      #   Boundary data files
│   └── output/              #   Output directory
├── elliptic/                # Elliptic grid generation (MPI parallel)
│   ├── generate_grid/       #   ellip.py, grid_data.py, grid_utils.py, dirichlet.py, higenstock.py, mympi.py
│   ├── prep_grid_bdry/      #   Boundary data files
│   └── output/              #   Output directory
├── grid_post_process/       # Post-processing (merging, sampling, stretching, PML)
│   ├── post_pro.py          #   Main script
│   ├── grid_io.py           #   Grid import & merge
│   ├── grid_sample.py       #   Bilinear upsampling
│   └── grid_data.py         #   GridData import
├── plotting/                # Grid visualization utilities
├── src_c/                   # C extension source code
│   ├── parabolic.c/h        #   Parabolic C kernel
│   ├── hyperbolic.c/h       #   Hyperbolic C kernel
│   ├── elliptic.c/h         #   Elliptic C kernels
│   ├── post_process.c/h     #   Post-process C kernels
│   ├── lib_math.c/h         #   Math library
│   └── build.sh             #   Build script for libgrid.so
└── docs/                    # Documentation
```

## Features

- **Three grid generation methods** with different algorithmic trade-offs
- **Grid quality evaluation**: orthogonality, Jacobian, aspect ratio, step size, smoothness
- **Post-processing**: multi-grid merging, bilinear sampling, arc-length stretching, PML layer support
- **C backend acceleration** via ctypes for compute-intensive kernels
- **MPI parallel execution** for the elliptic solver (2D Cartesian decomposition)
- **NetCDF I/O** for interoperability with seismic simulation codes

## Prerequisites

- Python >= 3.10
- NumPy
- Numba
- netCDF4
- mpi4py (for elliptic method only)
- Matplotlib (for plotting)
- GCC (for building C backend)

## Installation

```bash
git clone <repository_url>
cd 2d_grid_generate_python

# Build C backend (optional but recommended for performance)
cd src_c && bash build.sh && cd ..
```

## Quick Start

### Parabolic Method

```bash
cd parabolic/generate_grid
bash start.sh
```

### Hyperbolic Method

```bash
cd hyperbolic/generate_grid
bash start.sh
```

### Elliptic Method (MPI)

```bash
cd elliptic/generate_grid
bash start.sh
```

### Post-Processing

```bash
cd grid_post_process
bash start.sh
```

Each `start.sh` generates a `config.json` and runs the corresponding Python script. You can also run directly:

```bash
# Parabolic / Hyperbolic (single process)
python para.py --config-file config.json --verbose 10

# Elliptic (MPI)
mpiexec -np 4 python ellip.py --config-file config.json --verbose 10

# Post-processing
python post_pro.py --config-file config.json --verbose 10
```

## Configuration

All parameters are specified in a JSON file. Keys prefixed with `#` are treated as comments and ignored.

### Common Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `number_of_grid_points_x` | int | — | Number of grid points in x (xi) direction |
| `number_of_grid_points_z` | int | — | Number of grid points in z (zt) direction |
| `grid_check` | int (0/1) | 0 | Enable grid quality checks |
| `check_orth` | int (0/1) | 0 | Check orthogonality |
| `check_jac` | int (0/1) | 0 | Check Jacobian determinant |
| `check_ratio` | int (0/1) | 0 | Check aspect ratio |
| `check_step_xi` | int (0/1) | 0 | Check step size in xi direction |
| `check_step_zt` | int (0/1) | 0 | Check step size in zt direction |
| `check_smooth_xi` | int (0/1) | 0 | Check smoothness in xi direction |
| `check_smooth_zt` | int (0/1) | 0 | Check smoothness in zt direction |
| `geometry_input_file` | string | — | Path to boundary geometry file |
| `grid_export_dir` | string | — | Output directory for grid files |

### Parabolic Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step_input_file` | string | — | Path to step length file (nz-1 lines) |
| `coef` | int | 40 | Clustering coefficient (larger = finer near surface) |
| `t2b` | int (0/1) | 0 | Marching direction: 1 = top to bottom, 0 = bottom to top |

### Hyperbolic Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step_input_file` | string | — | Path to step length file (nz-1 lines) |
| `flag_stretch` | int (0/1) | 0 | Enable arc-length stretching after generation |
| `coef` | int | 70 | Smoothing/dissipation coefficient |
| `t2b` | int (0/1) | 0 | Marching direction: 1 = top to bottom, 0 = bottom to top |

### Elliptic Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `number_of_mpiprocs_x` | int | 1 | MPI processes in x direction |
| `number_of_mpiprocs_z` | int | 1 | MPI processes in z direction |
| `method` | object | — | Method configuration (see below) |

**Method sub-configuration** (choose `dirichlet` or `higenstock`):

```json
"method": {
    "dirichlet": {
        "coef": [20, 20, 20, 20],
        "weight": [0.0, 1.0],
        "iter_err": 1e-2,
        "max_iter": 5000
    }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `coef` | [int x4] | Source term coefficients for 4 boundaries [x1, x2, z1, z2] |
| `weight` | [float x2] | Exponential decay weights for source interpolation [min, max] |
| `iter_err` | float | Convergence tolerance (maximum residual) |
| `max_iter` | int | Maximum number of SOR iterations |

### Post-Processing Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_grid_number` | int | 1 | Number of input grids to merge |
| `input_grid_info_*` | object | — | Per-grid config (see below) |
| `number_of_mpiprocs_out` | [int, int] | [1,1] | Output MPI partitioning |
| `merge_direction` | string | — | Merge direction: `"x"` or `"z"` (required if input_grid_number >= 2) |
| `flag_sample` | int (0/1) | 0 | Enable bilinear sampling |
| `sample_factor` | [int, int] | [1,1] | Upsampling factors [x, z] |
| `pml_weight_2x` | int (0/1) | 0 | Double computation weight for PML layers |
| `number_of_pml_x1` | int | 0 | PML layer count at x1 boundary |
| `number_of_pml_x2` | int | 0 | PML layer count at x2 boundary |
| `number_of_pml_z1` | int | 0 | PML layer count at z1 boundary |
| `number_of_pml_z2` | int | 0 | PML layer count at z2 boundary |

**Per-grid input config** (`input_grid_info_0`, `input_grid_info_1`, ...):

| Key | Type | Description |
|-----|------|-------------|
| `number_of_grid_points` | [int, int] | Grid dimensions [nx, nz] |
| `number_of_mpiprocs_in` | [int, int] | MPI partitioning of input files |
| `grid_import_dir` | string | Directory containing input NetCDF files |
| `flag_stretch` | int (0/1) | Apply arc-length stretching |
| `stretch_direction` | string | Stretching direction: `"x"` or `"z"` |
| `stretch_file` | string | Path to arc-length file |

## Grid Quality Metrics

| Metric | Variable | Description | Ideal Value |
|--------|----------|-------------|-------------|
| Orthogonality | `orth` | Deviation from 90° between grid lines | 90° (perfect) |
| Jacobian | `jacobi` | Determinant of coordinate transformation | > 0 everywhere |
| Aspect Ratio | `ratio` | Ratio of adjacent cell dimensions | Close to 1 |
| Step xi | `step_xi` | Physical step length in xi direction | — |
| Step zt | `step_zt` | Physical step length in zt direction | — |
| Smooth xi | `smooth_xi` | Ratio of adjacent step lengths in xi | Close to 1 |
| Smooth zt | `smooth_zt` | Ratio of adjacent step lengths in zt | Close to 1 |

## C Backend

The C backend provides accelerated computation for core algorithms. To build:

```bash
cd src_c
bash build.sh
```

This compiles `libgrid.so` which is loaded via Python ctypes at runtime.

## Data Flow

```
prep_grid_bdry/          generate_grid/           grid_post_process/
(data_file_2d.txt)  -->  (para/hyper/ellip.py)  -->  (post_pro.py)
(step_file_2d.txt)       |                          |
                         v                          v
                    output/ (NetCDF)            output/ (NetCDF)
                    coord_*.nc                  coord_*.nc
                    orth_*.nc, jacobi_*.nc, ... quality_*.nc, ...
```

## Input File Formats

### Geometry File (`data_file_2d.txt`)

For parabolic/hyperbolic methods:
- 2 * nx lines (or nx lines for hyperbolic single boundary)
- Each line: `x z` (space-separated float values)
- Lines starting with `#` are comments

For elliptic method:
- 4 boundary blocks, each preceded by a `# boundary_x` comment
- Each block contains the (x, z) coordinates of one boundary

### Step File (`step_file_2d.txt`)

- nz - 1 lines, each containing a float step length
- Lines starting with `#` are comments

## License

See [LICENSE](LICENSE) file.
