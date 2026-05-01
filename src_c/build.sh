#!/usr/bin/env bash

gcc -shared -fPIC -O3 -o libgrid.so \
    elliptic.c parabolic.c hyperbolic.c post_process.c \
    utils.c lib_mem.c lib_math.c \
    -lm
