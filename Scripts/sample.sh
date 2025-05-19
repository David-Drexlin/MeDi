#!/bin/bash
#SBATCH --job-name=sample_OOD
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  accelerate launch sample.py \
    --path /home/space/datasets/tcga_uniform/deep_TSS_only_concatembed/checkpoint-100000/model.safetensors \
    --n 512 \
    --mode OOD \
    --domains_to_condition tissue_source_site \
    --number_of_different_conditional 8 \
    --cancer_types Colon_adenocarcinoma

