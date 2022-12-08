declare -a seeds=(10 31 100)

for seed in "${seeds[@]}"
do
    sbatch submit_resnet_cifar10.sh ${seed} knn
done
