#!/usr/bin/env bash

ACTION="train"
NUM_ACTIONS=128
MOVE_PAGES_COST=0.01
HBM_SIZE=35
MODEL_DIR="data/models"
TRAINING_SET=("data/miniFE/miniFE-2.0.1_openmp_opt/miniFE_openmp_opt/MCDRAM-miniFE.x-PEBS-countdown-8-PEBS-buffer-4096-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
		  "data/lammps-7-12-2016/src/USER-INTEL/TEST/DRAM-lammps-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
		  "data/nekbone-2.3.4/test/example1/DRAM-nekbone-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
		  # "data/lulesh2.0.2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
		  # "data/lulesh2.0.2/sparse_phased/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
		  # "data/lulesh2.0.2/sparse_phased_2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather" \
	     )
TEST_SET="data/lulesh2.0.2/DRAM-lulesh2.0-PEBS-countdown-8-PEBS-buffer-4096-CPU-0-window_len=1000-compare_unit=ms-interval_distance=23-page_size=12-observation_rows=128-observation_columns=128.feather"


function train_ai() {
    echo "Training from $(dirname $1)"
    python memai/ai.py train\
	   --gpu 0 \
	   --num-actions $NUM_ACTIONS \
	   --hbm-size $HBM_SIZE \
	   --model-dir $MODEL_DIR \
	   --input $1
}


function eval_ai() {
    echo "Evaluation for $(dirname $TEST_SET)"
    python memai/ai.py eval\
	   --gpu 0 \
	   --num-actions $NUM_ACTIONS \
	   --hbm-size $HBM_SIZE \
	   --model-dir $MODEL_DIR \
	   --input $TEST_SET
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


if [ $ACTION == "train" ]; then
    for input in ${TRAINING_SET[@]}; do
	train_ai $input
    done
fi
if [ $ACTION == "eval" ]; then
    eval_ai
fi
