#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-flat
#PJM -L node=1
#PJM --mpi proc=4
#PJM --omp thread=32
#PJM -L elapse=2:00:00
#PJM -g jh180024
#PJM -j

GEOMETRY=G6_nonP_0
PARAMETERS=parameters_sr900_ryr55

RUNNR=$(printf "%04d" $PJM_BULKNUM)
CASENAME="result_"$GEOMETRY"_sr900_ryr55_"$RUNNR

TSTOP=50
REACTION_DT="-T 5e-4"
STOCHASTIC_DT="-d 1e-4"
SAVE_DT=0.1
SIM_RES=12

MPI_ARGS="--mpi_dist 2 2 1"
ADDITIONAL_ARGS="-f -X 1 -s Ca -O ryr "

VAR=$(shuf -i 0-49 -n 1)
APP="calcium_sparks"

module load phdf5
#------ Pinning setting --------#
export I_MPI_PROCESS_MANAGER=hydra
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=/bin/pjrsh
export I_MPI_HYDRA_BOOTSTRAP_AUTOFORK=0
export I_MPI_HYDRA_HOST_FILE=$PJM_O_NODEINF
export I_MPI_HYDRA_PMI_CONNECT=alltoall
export I_MPI_FABRICS_LIST=tmi
export I_MPI_TMI_PROVIDER=psm2
export I_MPI_FALLBACK=0
export I_MPI_JOB_FAST_STARTUP=1
export I_MPI_HARD_FINALIZE=1
export I_MPI_PERHOST=$PJM_PROC_BY_NODE
export I_MPI_PIN=1
export OMP_PLACES=threads

unset I_MPI_PIN_PROCESSOR_LIST
unset KMP_AFFINITY

function set_HFI_NO_CPUAFFINITY () {
    if [ $PJM_PROC_BY_NODE -ge 2 ]; then
        export HFI_NO_CPUAFFINITY=1
    else
        unset HFI_NO_CPUAFFINITY
    fi
}
export I_MPI_PIN_DOMAIN=$OMP_NUM_THREADS
unset I_MPI_PIN_PROCESSOR_EXCLUDE_LIST
set_HFI_NO_CPUAFFINITY

#------- Program execution -------#
mpiexec.hydra -n ${PJM_MPI_PROC} numactl --preferred=1 ./$APP --geometry_file=$GEOMETRY".h5" --species_file=$PARAMETERS".h5"        -t $TSTOP -o $SAVE_DT -h $SIM_RES --casename=$CASENAME        $REACTION_DT $STOCHASTIC_DT $MPI_ARGS $ADDITIONAL_ARGS $VAR
