#!/usr/bin/env bash

gcc -shared -fPIC -O3 -o libgrid.so \
    grid_interface.c \
    parabolic.c hyperbolic.c utils.c \
    lib_mem.c lib_math.c \
    -lm