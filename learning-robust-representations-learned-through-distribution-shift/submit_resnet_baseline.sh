#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/network/scratch/a/ayush.chakravarthy/retrieval/logs/cifar10-resnet-baseline-%j.txt
#SBATCH --error=/network/scratch/a/ayush.chakravarthy/retrieval/errlogs/cifar10-resnet-baseline-%j.txt

module load anaconda/3
module load cuda/10.1
conda activate mem

seed=$1
name=$2

python3 train_resnet_on_cifar10.py \
	--buffer_model_path=./pretrained_models/byol_cifar10.ckpt --buffer_model_dict=./pretrained_models/byol_cifar10.json \
	--batch_size=256 --buffer_path=none  \
	--eval_batches=3 --eval_every_steps=5  \
	--name=${name} --num-workers=1 --root_dir='~/data/' --seed=${seed} \
	 --epochs=200 --optimizer=sgd --learning_rate 0.01
