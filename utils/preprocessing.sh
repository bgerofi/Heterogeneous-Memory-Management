#!/usr/bin/env bash

WINDOW_LEN=1000
PAGE_SIZE=12
COMPARE_UNIT="ms"
INTERVAL_DISTANCE=23
OBSERVATION_ROWS=128
OBSERVATION_COLUMNS=128
INPUTS=(\
				"data/lulesh2.0.2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather:data/lulesh2.0.2/MCDRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather" \
	    "data/lulesh2.0.2/sparse_phased/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather:data/lulesh2.0.2/sparse_phased/MCDRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather" \
	    "data/lulesh2.0.2/sparse_phased_2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather:data/lulesh2.0.2/sparse_phased_2/MCDRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather" \
	    "data/nekbone-2.3.4/test/example1/DRAM-nekbone-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather:data/nekbone-2.3.4/test/example1/MCDRAM-nekbone-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather" \
	    "data/lammps-7-12-2016/src/USER-INTEL/TEST/DRAM-lammps-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather:data/lammps-7-12-2016/src/USER-INTEL/TEST/MCDRAM-lammps-PEBS-countdown-8-PEBS-buffer-4096-CPU-0.feather" \
	    "data/miniFE/miniFE-2.0.1_openmp_opt/miniFE_openmp_opt/DRAM-miniFE.x-PEBS-countdown-8-PEBS-buffer-4096.feather:data/miniFE/miniFE-2.0.1_openmp_opt/miniFE_openmp_opt/MCDRAM-miniFE.x-PEBS-countdown-8-PEBS-buffer-4096.feather"
       )


function preprocess_input() {
    echo "Processing $(dirname $1)"
    python memai/preprocessing.py \
	   --ddr-input $1 \
	   --hbm-input $2 \
	   --window-len $WINDOW_LEN \
	   --page-size $PAGE_SIZE \
	   --interval-distance $INTERVAL_DISTANCE \
	   --compare-unit $COMPARE_UNIT \
	   --observation-rows $OBSERVATION_ROWS \
	   --observation-columns $OBSERVATION_COLUMNS
}


function usage() {
    printf "Usage: %s " ${0##*/}
    printf "[-c <observation-columns>] "
    printf "[-i <interval-distance>] "
    printf "[-p <page-size>] "
    printf "[-r <observation-rows>] "
    printf "[-u <compare-unit>] "
    printf "[-w <window-len]"
    printf "\n"
    printf "See memai/preprocessing.py --help\n"
}


while getopts c:hi:p:r:u:w: OPTION
do
    case $OPTION in
	c) OBSERVATION_ROWS=$OPTARG
	   ;;
	h) echo "Preprocess all data traces one by one."
	   usage
	   exit 0
	   ;;
	i) INTERVAL_DISTANCE=$OPTARG
	   ;;
	p) PAGE_SIZE=$OPTARG
	   ;;
	r) OBSERVATION_ROWS=$OPTARG
	   ;;
	u) COMPARE_UNIT=$OPTARG
	   ;;
	w) WINDOW_LEN=$OPTARG
	   ;;
	\:) exit 1
	    ;;
	\?) exit 1
	    ;;
    esac
done


for input in ${INPUTS[@]}; do
    ddr=$(cut -d ':' -f1 <<< $input)
    hbm=$(cut -d ':' -f2 <<< $input)
    preprocess_input $ddr $hbm
done
