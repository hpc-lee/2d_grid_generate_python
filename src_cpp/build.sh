#!/usr/bin/env bash
# Build C++ grid generation kernels as pybind11 module gridcpp.so
# Minimal build, no CMake dependency
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PY=python3
PY_INC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PB_INC=$($PY -c "import pybind11; print(pybind11.get_include())")

g++ -shared -fPIC -O3 -std=c++14 -Wall \
    -I"$PY_INC" -I"$PB_INC" \
    gridcpp.cpp -o gridcpp.so

echo "built $SCRIPT_DIR/gridcpp.so"
