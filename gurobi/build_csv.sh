#!/bin/bash

set -e
path_prefix="outputs/clustering/testing"
output_file="${path_prefix}/testing.csv"
params=('h' 'v' 'hv' 'n')
param_args=('--hp' '--vp' '--hp --vp' '')

inps=($(echo "18 19"))
# inps=({0..13})

for param in {0..3}; do
    for inp in ${inps[@]}; do
        echo -n "${params[$param]}, ${inp}, " >> $output_file
        args="-i ${inp} -o ${output_file} -v ${param_args[$param]} --mipout"
        filepath="${path_prefix}/${params[$param]}_${inp}"
        python main.py -s Univmon $args 2>&1 | tee "${filepath}_univmon.out"
        python main.py -s UnivmonGreedy $args 2>&1 | tee "${filepath}_univmon_greedy.out"
        python main.py -s UnivmonGreedyRows $args 2>&1 | tee "${filepath}_univmon_greedy_rows.out"
        python main.py -s Netmon $args 2>&1 | tee "${filepath}_netmon.out"
        echo "" >> $output_file
        # rm pickle_objs/inp-*
    done
done
set +e
