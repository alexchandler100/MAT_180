#!/bin/bash

declare -a seeds=(10 43 25 31 100)

for seed in "${seeds[@]}"
do
	sbatch submit_perceiver_cifar10.sh ${seed} perceiver_knn_ret 100 knn
done
