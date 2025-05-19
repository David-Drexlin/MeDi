#!/bin/bash
#SBATCH --job-name=diff_train
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint="80gb"
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  accelerate launch \
    --num_processes 2 --multi_gpu \
    train_diffusion.py \
    --optimization_steps 3000000 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --model UNet \
    --cond_type additive \
    --mixed_precision fp16 \
    --holdout_mask tissue_source_site gender race \
    --resolution 128 \
    --use_wandb \
    --FID_tracker 1000000 \
    --checkpointing_steps 100000 \
    --domains_to_condition tissue_source_site \
    --output_dir deep_TSS_res:128__additive_embed_comb

