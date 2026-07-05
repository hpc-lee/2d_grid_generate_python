# 2d_grid_generate_python

二维曲线网格生成（Python + C++ pybind11 加速）。支持椭圆/双曲/抛物三种方法 + 后处理，热点算子在 C++ 内核、Python 负责调度。

## 目录结构

| 目录 | 说明 |
|---|---|
| `elliptic/` | 椭圆方法（SOR/Gauss-Seidel 迭代，MPI 笛卡尔拓扑并行） |
| `hyperbolic/` | 双曲方法（块三对角 Thomas + 2×2 求逆） |
| `parabolic/` | 抛物方法（TFI 初始化 + Thomas 逐层推进） |
| `grid_post_process/` | 后处理（多网格 merge / sample / x/z 轴 swap / 质量检查） |
| `src_cpp/` | C++ 加速内核 `gridcpp`（pybind11，零拷贝） |
| `common/` | 通用工具（边界操作、网格质量、代数） |
| `plotting/` | 画图（网格 + 质量指标） |

## 依赖

- Python ≥ 3.10
- NumPy
- netCDF4
- mpi4py（elliptic 方法）
- Matplotlib（画图）
- g++ ≥ 4.9（C++14）+ pybind11（编译 C++ 内核）

## 编译 C++ 内核

```bash
cd src_cpp && bash build.sh
```

生成 `gridcpp.so`，热点算子（parabolic 的 predict_point/update_point、hyperbolic 的 cal_matrix/thomas_block、elliptic 的 update_SOR/TFI）在 C++ 内核，Python 负责调度。

## 三方法使用

各方法目录下 `start.sh` 一键运行：

```bash
cd elliptic/generate_grid && bash start.sh    # MPI 2x2，dirichlet/higenstock
cd hyperbolic/generate_grid && bash start.sh  # 单进程，t2b + step 控制推进方向
cd parabolic/generate_grid && bash start.sh   # 单进程，t2b + step
```

### 推进方向控制（hyperbolic/parabolic）

沿 z 方向推进，方向由 `step` 正负 + `t2b` 共同决定：
- 向 z 上：step 正值 + `t2b=0`
- 向 z 下：step 负值 + `t2b=1`

## 后处理（grid_post_process）

`start.sh` 配置项：

| 字段 | 说明 |
|---|---|
| `input_grid_number` | 输入网格数（1=单网格，2+=merge） |
| `merge_direction` | merge 方向（`x`/`z`） |
| `flag_sample` | 重采样（1=启用） |
| `flag_swap_xz` | x/z 轴交换（1=反射坐标系回物理坐标系） |
| `flag_stretch` | 弧长拉伸（1=启用） |

`post_pro.py` 执行完成后自动写一份 `config.json`（含输出网格 swap 后的 `nx/nz`）到 `grid_export_dir`，供 `draw_grid.py` 直接读取，无需手动维护尺寸。

### x/z 轴 swap（`flag_swap_xz`）

parabolic/hyperbolic 沿 z 推进。若基于 `bx1/bx2`（x 方向边界，自由面在 x）生成，流程：
1. `creat_bx_bdry.py` 把物理 bx1/bx2 经 y=x 反射成 bz1/bz2
2. 网格生成（反射坐标系）
3. 后处理 `flag_swap_xz=1` 把网格 swap 回物理坐标系

swap 在 `read_import_coord` 之后立即执行，所有后续操作（sample/quality/export）基于 swap 后的物理坐标系网格。

## 画图

```bash
cd plotting
python draw_grid.py       # 画网格线
python draw_quality.py    # 画质量指标（orth/jacobi/ratio/step_xi/step_zt/smooth_xi/smooth_zt）
```

通过修改脚本顶部 `cfs_file` 指向目标 `config.json`，`varnm` 选择质量指标。

## 文档

完整算法推导与使用手册见 [docs/user_manual.pdf](docs/user_manual.pdf)（LaTeX 源码 `docs/user_manual.tex`）。

## 许可证

BSD 2-Clause License，详见 [LICENSE](LICENSE)。
