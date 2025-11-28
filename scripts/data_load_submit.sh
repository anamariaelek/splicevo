#!/bin/bash
#SBATCH --job-name=splicevo_data_human
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate splicevo

# Splicevo directory
SPLICEVO_DIR=/home/hd/hd_hd/hd_mf354/projects/splicevo/

# Genome to process
GENOME="human_GRCh37"

# Load the genome
OUT_DIR=/home/hd/hd_hd/hd_mf354/sds/sd17d003/Anamaria/splicevo/data/processed
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/genomes_helix.json \
    --genome_id ${GENOME} \
    --output_dir ${OUT_DIR} \
    --window_size 1000 \
    --context_size 450 \
    --n_cpus 1