#!/bin/sh

python mip.py -c "$1" -s netmon | tee "placements/$1_netmon.out"
python mip.py -c "$1" -s univmon | tee "placements/$1_univmon_ns.out"
python mip.py -c "$1" -s univmon_greedy | tee "placements/$1_univmon_greedy_ns.out"
