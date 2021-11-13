#!/bin/bash

SCRIPT="`readlink -f ${BASH_SOURCE[0]:-}`"
SCRIPT_DIR=$(dirname ${SCRIPT})

if [ $# -eq 0 ]; then
	echo "error: specify input PEBS .dat file"
	exit 1
fi

while [ $# -gt 0 ]; do
	f=$1
	shift
	python3 -u ${SCRIPT_DIR}/pebsview.py -f ${f}
	f="${f%.*}.feather";
	f_new=`echo ${f} | sed 's/-pid-[0-9]*-tid-[0-9]*//'`
	mv ${f} ${f_new}
	echo "${f_new}"
done
