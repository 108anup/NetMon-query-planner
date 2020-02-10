#!/bin/bash

output_file="outputs/sensitivity.csv"

cfgs=($(echo 12))

for cfg in ${cfgs[@]}; do
    echo -n "v, ${cfg}, " >> $output_file
    args="-c ${cfg} -o ${output_file} -v --mipout"
    python mip.py -s univmon $args \
        | tee "outputs/build_csv/${cfg}_univmon.out"
    python mip.py -s univmon_greedy $args \
        | tee "outputs/build_csv/${cfg}_univmon_greedy.out"
    python mip.py -s univmon_greedy_rows $args \
        | tee "outputs/build_csv/${cfg}_univmon_greedy_rows.out"
    python mip.py -s netmon $args \
        | tee "outputs/build_csv/${cfg}_netmon.out"
    echo "" >> $output_file
done
