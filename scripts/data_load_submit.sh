#!/bin/bash
#SBATCH --job-name=splicevo_data_load_mouse
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

SPLICEVO_DIR=/home/elek/projects/splicevo

GENOME="mouse_GRCm38"

python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/genomes.json \
    --genome_id ${GENOME} \
    --output_dir ${SPLICEVO_DIR}/data/processed \
    --window_size 1000 \
    --context_size 450 \
    --n_cpus 8