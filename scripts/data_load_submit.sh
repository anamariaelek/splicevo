#!/bin/bash
#SBATCH --job-name=mouse
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=2
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Genome to process
GENOME="mouse_GRCm38"
#GENOME="rat_Rnor_5.0"
#GENOME="human_GRCh37"

# Where to save
OUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/processed_full_50kb/

# Create output directory if it doesn't exist
mkdir -p ${OUT_DIR}

# Load the genome
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/genomes.json \
    --genome_id ${GENOME} \
    --output_dir ${OUT_DIR} \
    --window_size 50000 \
    --context_size 5000 \
    --n_cpus 4 \
    --quiet