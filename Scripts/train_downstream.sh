#!/bin/bash
#SBATCH --job-name=embed_pipeline
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  accelerate launch downstream.py \
    --split train                \
    --base-path splits/all_classes \
    # add `--augment` here if you want data augmentation
#!/bin/bash
#SBATCH --job-name=embed_pipeline
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

apptainer run --nv python_container.sif \
  accelerate launch downstream.py \
    --split train                \
    --base-path splits/all_classes \
    # add `--augment` here if you want data augmentation

