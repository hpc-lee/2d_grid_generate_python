#!/usr/bin/env bash

INPUTDIR1=`pwd`/../hyperbolic/output
STRETCH_FILE1=`pwd`/arc_len_file1.txt

INPUTDIR2=`pwd`/../hyperbolic/output_up
STRETCH_FILE2=`pwd`/arc_len_file2.txt

OUTPUTDIR=`pwd`/output
CONFIGS=${OUTPUTDIR}/config.json

rm -rf "${OUTPUTDIR}"
mkdir -p "${OUTPUTDIR}"

cat << ieof > ${CONFIGS}
{
    "input_grid_number" : 2,

    "input_grid_info_0" : {
        "number_of_grid_points" : [841, 601],
        "number_of_mpiprocs_in" : [1, 1],
        "grid_import_dir" : "${INPUTDIR1}",
        "flag_stretch" : 0,
        "stretch_direction" : "z",
        "stretch_file" : "${STRETCH_FILE1}"
    },

    "input_grid_info_1" : {
        "number_of_grid_points" : [841, 601],
        "number_of_mpiprocs_in" : [1, 1],
        "grid_import_dir" : "${INPUTDIR2}",
        "flag_stretch" : 0,
        "stretch_direction" : "z",
        "stretch_file" : "${STRETCH_FILE2}"
    },

    "number_of_mpiprocs_out" : [1, 1],
    "merge_direction" : "z",
    "flag_sample" : 0,
    "sample_factor" : [1, 2],
    "flag_swap_xz" : 0,
    "grid_export_dir" : "${OUTPUTDIR}",

    "grid_check" : 1,
    "check_orth" : 1,
    "check_jac" : 1,
    "check_ratio" : 1,
    "check_step_xi" : 1,
    "check_step_zt" : 1,
    "check_smooth_xi" : 1,
    "check_smooth_zt" : 1,

    "pml_weight_2x" : 1,
    "number_of_pml_x1" : 20,
    "number_of_pml_x2" : 20,
    "number_of_pml_z1" : 20,
    "number_of_pml_z2" : 0
}
ieof

python post_pro.py \
    --config-file ${CONFIGS} \
    --verbose 10  2>&1 | tee output.log

# vim:ts=4:sw=4:nu:et:ai:
