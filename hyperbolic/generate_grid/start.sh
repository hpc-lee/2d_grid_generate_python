#!/usr/bin/env bash

INPUTDIR=`pwd`/../prep_grid_bdry
OUTPUTDIR=`pwd`/../output_up
CONFIGS=${OUTPUTDIR}/config.json

rm -rf "${OUTPUTDIR}"
mkdir -p "${OUTPUTDIR}"

cat << ieof > ${CONFIGS} 
{
    "number_of_grid_points_x" : 841,
    "number_of_grid_points_z" : 601,


    "grid_check" : 1,
    "check_orth" : 1,
    "check_jac" : 1,
    "check_ratio" : 1,
    "check_step_xi" : 1,
    "check_step_zt" : 1,
    "check_smooth_xi" : 1,
    "check_smooth_zt" : 1,

    "geometry_input_file" : "${INPUTDIR}/data_file_2d.txt",
    "step_input_file" : "${INPUTDIR}/step_file_up.txt",
    "grid_export_dir" : "${OUTPUTDIR}",

    "flag_stretch" : 0,
    "coef" : 70,
    "t2b" : 0
}
ieof

python hyper.py \
    --config-file ${CONFIGS} \
    --verbose 10  2>&1 | tee log

# vim:ts=4:sw=4:nu:et:ai:
