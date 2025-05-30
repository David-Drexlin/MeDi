#!/bin/bash
#SBATCH --job-name=sample_union_tss
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sample_union_tss_%j.out
#SBATCH --error=logs/sample_union_tss_%j.err

export CUDA_VISIBLE_DEVICES=0

# run inside container via accelerate
apptainer run --nv python_container.sif \
  accelerate launch /home/daviddrexlin/MeDi/train_linear.py \
    --ratio 0.0 \
    --max_real 20 \
    --seed 5 \
    --cancer_types Lung_squamous_cell_carcinoma Lung_adenocarcinoma \
    --iters 1000 \
    --sweep_number_tss 5 \

