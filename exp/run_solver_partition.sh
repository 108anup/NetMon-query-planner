#!/bin/bash
set -m

num_options=11

for ((i=0; i<$num_options; i++)); do
    python solver_partition.py $i &
done

while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done
