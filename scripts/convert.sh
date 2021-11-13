#!/bin/bash

SCRIPT="`readlink -f ${BASH_SOURCE[0]:-}`"
SCRIPT_DIR=$(dirname ${SCRIPT})

for f in *.dat; do
	python3 -u ${SCRIPT_DIR}/pebsview.py -f ${f}
done

for f in *.feather; do 
	f_new=`echo ${f} | sed 's/-pid-[0-9]*-tid-[0-9]*//'`
	mv ${f} ${f_new}
done

ls *.feather
