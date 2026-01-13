#!/usr/bin/env bash

INPUTDIR=`pwd`/../prep_grid_bdry
OUTPUTDIR=`pwd`/../output
CONFIGS=${OUTPUTDIR}/config.json

rm -rf "${OUTPUTDIR}"
mkdir -p "${OUTPUTDIR}"

#-- total x mpi procs
NPROCS_X=2
#-- total z mpi procs
NPROCS_Z=2

cat << ieof > ${CONFIGS} 
{
    "number_of_grid_points_x" : 841,
    "number_of_grid_points_z" : 601,

    "number_of_mpiprocs_x" : $NPROCS_X,
    "number_of_mpiprocs_z" : $NPROCS_Z,

    "grid_check" : 1,
    "check_orth" : 1,
    "check_jac" : 1,
    "check_ratio" : 1,
    "check_step_xi" : 1,
    "check_step_zt" : 1,
    "check_smooth_xi" : 1,
    "check_smooth_zt" : 1,

    "geometry_input_file" : "${INPUTDIR}/data_file_2d.txt",
    "grid_export_dir" : "${OUTPUTDIR}",

    "method" : {
        "#tfi":"",
        "dirichlet" : {
            "coef" : [20,20,20,20],
            "weight" : [0.0,1.0],
            "iter_err" : 1E-2,
            "max_iter" : 5E3
        },
        "#higenstock" : {
            "coef" : [2000,2000,50,20],
            "weight" : [0.0,1.0],
            "iter_err" : 1E-2,
            "max_iter" : 5E3
        }
    }
}
ieof

python ellip.py \
    --config-file ${CONFIGS} \
    --verbose 10  2>&1 | tee log

# vim:ts=4:sw=4:nu:et:ai:
