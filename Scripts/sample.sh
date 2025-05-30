#!/bin/bash
#SBATCH --job-name=sample_OOD
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/print_%j.out
#SBATCH --error=logs/sample_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  python sample.py \
    --path /home/daviddrexlin/MeDi/Models/deep_TSS_res:128__additive_embed_comb/checkpoint-100/model.safetensors \
    --n 16 \
    --mode full \
    --domains_to_condition tissue_source_site \
    --number_of_different_conditional 8 \
    --cancer_types Cholangiocarcinoma \
    --real_data_root /home/daviddrexlin/TCGA/TCGA

