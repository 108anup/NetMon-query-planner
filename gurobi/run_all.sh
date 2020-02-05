#!/bin/bash

if [[ -z $1 ]]; then
    $1 = "0"
fi

if [[ -n $2 ]]; then
    args="-p"
fi

python mip.py -c "$1" -s netmon -v $args | tee "sens_placements/$1_$2netmon.out"
python mip.py -c "$1" -s univmon -v $args | tee "sens_placements/$1_$2univmon_ns.out"
python mip.py -c "$1" -s univmon_greedy -v $args | tee "sens_placements/$1_$2univmon_greedy_ns.out"
python mip.py -c "$1" -s univmon_greedy_rows -v $args | tee "sens_placements/$1_$2univmon_greedy_rows_ns.out"
