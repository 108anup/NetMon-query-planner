#!/bin/bash

dir="outputs"
if [[ -z $1 ]]; then
    $1 = "0"
fi

if [[ -n $2 ]]; then
    dir="$dir/$2"
    if [[ ! -d $dir ]]; then
        mkdir -p $dir
    fi
fi

args=""
if [[ -n $3 ]]; then
    args="-p"
fi

python mip.py -c "$1" -s univmon -vv $args | tee "$dir/$1_$3univmon_ns.out"
python mip.py -c "$1" -s univmon_greedy -vv $args | tee "$dir/$1_$3univmon_greedy_ns.out"
python mip.py -c "$1" -s univmon_greedy_rows -vv $args | tee "$dir/$1_$3univmon_greedy_rows_ns.out"
python mip.py -c "$1" -s netmon -vv $args | tee "$dir/$1_$3netmon.out"
