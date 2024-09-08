#!/bin/bash

declare -a seeds=(10 278 1872)

for seed in "${seeds[@]}"
do
	sbatch submit_resnet_baseline.sh ${seed} resnet_baseline
done
