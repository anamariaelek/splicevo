#!/bin/bash
#SBATCH --job-name=human
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=2
#SBATCH --mem=512G
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
#GENOME="rat_Rnor_5.0"
#GENOME="mouse_GRCm38"
GENOME="human_GRCh37"

# Load the genome
OUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/processed_helix/
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/genomes_helix.json \
    --genome_id ${GENOME} \
    --output_dir ${OUT_DIR} \
    --window_size 1000 \
    --context_size 450 \
    --n_cpus 2 \
    --quiet
