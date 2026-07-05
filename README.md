# 2d_grid_generate_python

2D curvilinear grid generation (Python + C++ pybind11 acceleration). Supports three methods — elliptic, hyperbolic, and parabolic — plus post-processing. Hot kernels run in the C++ layer; Python orchestrates.

## Directory Structure

| Directory | Description |
|---|---|
| `elliptic/` | Elliptic method (SOR/Gauss-Seidel iteration, MPI Cartesian topology) |
| `hyperbolic/` | Hyperbolic method (block Thomas + 2×2 inversion) |
| `parabolic/` | Parabolic method (TFI initialization + layer-by-layer Thomas marching) |
| `grid_post_process/` | Post-processing (multi-grid merge / sample / x-z axis swap / quality checks) |
| `src_cpp/` | C++ acceleration kernel `gridcpp` (pybind11, zero-copy) |
| `common/` | Common utilities (boundary ops, grid quality, algebra) |
| `plotting/` | Plotting (grid + quality metrics) |

## Dependencies

- Python ≥ 3.10
- NumPy
- netCDF4
- mpi4py (elliptic method)
- Matplotlib (plotting)
- g++ ≥ 4.9 (C++14) + pybind11 (to build the C++ kernel)

## Building the C++ Kernel

```bash
cd src_cpp && bash build.sh
```

This produces `gridcpp.so`. Hot kernels (parabolic `predict_point`/`update_point`, hyperbolic `cal_matrix`/`thomas_block`, elliptic `update_SOR`/TFI) live in the C++ layer; Python handles orchestration.

## Usage

Run `start.sh` in each method's directory:

```bash
cd elliptic/generate_grid && bash start.sh    # MPI 2x2, dirichlet/higenstock
cd hyperbolic/generate_grid && bash start.sh  # single process, t2b + step control marching direction
cd parabolic/generate_grid && bash start.sh   # single process, t2b + step
```

### Marching Direction (hyperbolic/parabolic)

Marching is along z; the direction is set by the sign of `step` together with `t2b`:
- Upward in z: positive `step` + `t2b=0`
- Downward in z: negative `step` + `t2b=1`

## Post-Processing (grid_post_process)

`start.sh` configuration keys:

| Key | Description |
|---|---|
| `input_grid_number` | Number of input grids (1=single, 2+=merge) |
| `merge_direction` | Merge direction (`x`/`z`) |
| `flag_sample` | Resampling (1=enabled) |
| `flag_swap_xz` | x/z axis swap (1=reflect back to physical coordinates) |
| `flag_stretch` | Arc-length stretching (1=enabled) |

After `post_pro.py` finishes, it auto-writes a `config.json` (with post-swap `nx/nz`) to `grid_export_dir` for `draw_grid.py` to read directly — no manual size maintenance.

### x/z Axis Swap (`flag_swap_xz`)

parabolic/hyperbolic march along z. When generation is based on `bx1/bx2` (x-direction boundaries, free surface on x), the workflow is:
1. `creat_bx_bdry.py` reflects physical bx1/bx2 into bz1/bz2 via y=x
2. Grid generation (in the reflected coordinate system)
3. Post-processing with `flag_swap_xz=1` swaps the grid back to physical coordinates

The swap runs immediately after `read_import_coord`; all subsequent operations (sample/quality/export) act on the swapped, physical-coordinate grid.

## Plotting

```bash
cd plotting
python draw_grid.py       # draw grid lines
python draw_quality.py    # draw quality metrics (orth/jacobi/ratio/step_xi/step_zt/smooth_xi/smooth_zt)
```

Edit `cfs_file` at the top of the script to point at the target `config.json`, and `varnm` to select the quality metric.

## Documentation

Full algorithm derivation and usage manual: [docs/user_manual.pdf](docs/user_manual.pdf) (LaTeX source `docs/user_manual.tex`).

## License

BSD 2-Clause License — see [LICENSE](LICENSE).
