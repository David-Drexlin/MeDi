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
  accelerate launch /home/daviddrexlin/MeDi/embed.py \
    --path /home/daviddrexlin/MeDi/Models/deep_TSS_res:128__additive_embed_comb/checkpoint-100/model.safetensors \
    --cancer_types Lung_squamous_cell_carcinoma Lung_adenocarcinoma \
    --n 20 \
    --mode_union \
    --batch_size 8 \
    --infer_steps 100 \
    --output_dir /home/daviddrexlin/MeDi/embeddings \
    --device cuda \
    --domains_to_condition tissue_source_site \
    --holdout_meta /home/daviddrexlin/MeDi/holdout_metadata_df_complex.csv \
    --real_data_root /home/daviddrexlin/TCGA/TCGA\


