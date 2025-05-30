#!/bin/bash
#SBATCH --job-name=diff_train
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="80gb"
#SBATCH --output=logs/print_%j.out
#SBATCH --error=logs/train_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  accelerate launch \
    train_diffusion.py \
    --optimization_steps 101 \
    --data_root /home/daviddrexlin/TCGA/TCGA\
    --batch_size 32 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --holdout_mask tissue_source_site gender race \
    --resolution 128 \
    --use_wandb \
    --FID_tracker 100 \
    --checkpointing_steps 10 \
    --output_dir Models/deep_CLS_res:128__additive_embed_comb

#--num_processes 1 --multi_gpu \