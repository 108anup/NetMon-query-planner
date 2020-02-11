#!/bin/bash

output_file="outputs/sensitivity3.csv"
params=('h' 'v' 'hv' 'n')
param_args=('--hp' '--vp' '--hp --vp' '')

cfgs=($(echo "9"))

for param in {0..3}; do
    for cfg in ${cfgs[@]}; do
        echo -n "${params[$param]}, ${cfg}, " >> $output_file
        args="-c ${cfg} -o ${output_file} -v ${param_args[$param]}"
        python mip.py -s univmon $args \
            | tee "outputs/build_csv/${params[$param]}_${cfg}_univmon.out"
        python mip.py -s univmon_greedy $args \
            | tee "outputs/build_csv/${params[$param]}_${cfg}_univmon_greedy.out"
        python mip.py -s univmon_greedy_rows $args \
            | tee "outputs/build_csv/${params[$param]}_${cfg}_univmon_greedy_rows.out"
        python mip.py -s netmon $args \
            | tee "outputs/build_csv3/${params[$param]}_${cfg}_netmon.out"
        echo "" >> $output_file
        rm pickle_objs/cfg-*
    done

done
