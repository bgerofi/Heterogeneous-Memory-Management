#!/usr/bin/env bash

ACTION="train"
NUM_ACTIONS=128
MOVE_PAGES_COST=0.01
HBM_SIZE=35
MODEL_DIR="data/models"


function train_ai() {
    echo "Training."
    python memai/ai.py train\
	   --gpu 0 \
	   --num-actions $NUM_ACTIONS \
	   --hbm-size $HBM_SIZE \
	   --model-dir $MODEL_DIR \
	   --input "data/miniFE/miniFE-2.0.1_openmp_opt/miniFE_openmp_opt/MCDRAM-miniFE.x-PEBS-countdown-8-PEBS-buffer-4096-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
	   --input "data/lammps-7-12-2016/src/USER-INTEL/TEST/DRAM-lammps-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
	   --input "data/nekbone-2.3.4/test/example1/DRAM-nekbone-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
# --input "data/lulesh2.0.2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather"
# --input "data/lulesh2.0.2/sparse_phased/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather"
# --input  "data/lulesh2.0.2/sparse_phased_2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather"
}


function eval_ai() {
    echo "Evaluation."
    python memai/ai.py eval\
	   --gpu 0 \
	   --num-actions $NUM_ACTIONS \
	   --hbm-size $HBM_SIZE \
	   --model-dir $MODEL_DIR \
	   --input "data/lulesh2.0.2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather"
}


function usage() {
    printf "Usage: %s " ${0##*/}
    printf "[-a <action: train | eval>] "
    printf "[-n <num-actions>] "
    printf "[-d <model-dir>] "
    printf "[-p <move-page-cost>] "
    printf "[-s <hbm-size>] "
    printf "\n"
    printf "See memai/ai.py --help\n"
}


while getopts a:hp:s: OPTION
do
    case $OPTION in
	a) ACTION=$(grep -Ee "train|eval" <<< $OPTARG)
	   if [ -z "$ACTION" ]; then
	       echo "Invalid action value: $OPTARG"
	       usage
	       exit 1
	   fi
	   ;;
	d) MODEL_DIR=$OPTARG
	   ;;
	h) echo "Train an AI."
	   usage
	   exit 0
	   ;;
	n) NUM_ACTIONS=$OPTARG
	   ;;
	p) MOVE_PAGES_COST=$OPTARG
	   ;;
	s) HBM_SIZE=$OPTARG
	   ;;
	\:) exit 1
	    ;;
	\?) exit 1
	    ;;
    esac
done


if [ $ACTION == "train" ]; then train_ai; fi
if [ $ACTION == "eval" ]; then eval_ai; fi
