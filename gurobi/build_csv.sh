#!/bin/bash

set -e
path_prefix="outputs/clustering/testing"
results_file="${path_prefix}/testing.csv"
params=('h' 'v' 'hv' 'n')
param_args=('--hp' '--vp' '--hp --vp' '')

inps=($(echo "23"))
# inps=({0..13})

init=false

for param in {0..3}; do
    for inp in ${inps[@]}; do
        if $init ; then
            echo -n "init, ${params[$param]}, ${inp}, " >> $results_file
            args="-i ${inp} -r ${results_file} --init -vvv ${param_args[$param]} --mipout"
            filepath="${path_prefix}/${params[$param]}_init_${inp}"
        else
            echo -n "noinit, ${params[$param]}, ${inp}, " >> $results_file
            args="-i ${inp} -r ${results_file} -vvv ${param_args[$param]} --mipout"
            filepath="${path_prefix}/${params[$param]}_noinit_${inp}"
        fi
        python main.py -s Univmon $args 2>&1 | tee "${filepath}_univmon.out"
        python main.py -s UnivmonGreedy $args 2>&1 | tee "${filepath}_univmon_greedy.out"
        python main.py -s UnivmonGreedyRows $args 2>&1 | tee "${filepath}_univmon_greedy_rows.out"
        python main.py -s Netmon $args 2>&1 | tee "${filepath}_netmon.out"
        echo "" >> $results_file
        # rm pickle_objs/inp-*
    done
done
set +e
