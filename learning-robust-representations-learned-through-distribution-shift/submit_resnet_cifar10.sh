#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02-00:00:00
#SBATCH --output=/network/scratch/a/ayush.chakravarthy/retrieval/logs/cifar10-retrieval-%j.txt
#SBATCH --error=/network/scratch/a/ayush.chakravarthy/retrieval/errlogs/cifar10-retrieval-%j.txt

module load anaconda/3
module load cuda/10.1
conda activate mem

seed=$1
name=$2

python3 train_resnet_cifar10_oracle_db.py \
	--buffer_model_path=./pretrained_models/byol_cifar10.ckpt --buffer_model_dict=./pretrained_models/byol_cifar10.json --attn_dropout=0.1 \
	--attn_retrieval=reps_for_both --batch_size=256 --buffer_path=./data_buffer/byol_cifar10.pt --cross_dim_head=32 \
	--cross_heads=1 --depth=2 --eval_batches=3 --eval_every_steps=5 --ff_dropout=0.1 --latent_dim=512 \
	--latent_heads=2 --name=${name} --num-workers=1 --num_latents=10 --root_dir='~/data/' --seed=${seed} \
	--retrieval_access=oracle_sampled --epochs 300 --no_labels_from_reps
