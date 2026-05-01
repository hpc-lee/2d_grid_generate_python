#!/usr/bin/env bash

INPUTDIR1=`pwd`/../elliptic/output
STRETCH_FILE1=`pwd`/arc_len_file1.txt

OUTPUTDIR=`pwd`/output
CONFIGS=${OUTPUTDIR}/config.json

rm -rf "${OUTPUTDIR}"
mkdir -p "${OUTPUTDIR}"

cat << ieof > ${CONFIGS}
{
    "input_grid_number" : 1,

    "input_grid_info_0" : {
        "number_of_grid_points" : [801, 601],
        "number_of_mpiprocs_in" : [2, 2],
        "grid_import_dir" : "${INPUTDIR1}",
        "flag_stretch" : 1,
        "stretch_direction" : "z",
        "stretch_file" : "${STRETCH_FILE1}"
    },

    "number_of_mpiprocs_out" : [1, 1],
    "flag_sample" : 1,
    "sample_factor" : [1, 1],
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
