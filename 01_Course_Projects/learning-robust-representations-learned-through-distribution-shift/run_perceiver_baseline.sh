#!/bin/bash

declare -a seeds=(10 43 25 31 100)

for seed in "${seeds[@]}"
do
	sbatch submit_perceiver_baseline.sh ${seed} perceiver_baseline
done
