#!/bin/bash
#SBATCH --job-name=split_mouse_human
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=10gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

# Memory rules of thumb (with optimizations):
# - 32G for small datasets (e.g., a few chromosomes in a single genome, a few usage conditions)
# - 64G for medium datasets (e.g., multiple chromosomes from multiple genomes, a few usage conditions)
# - 128G for large datasets (e.g., whole genomes of multiple species, many usage conditions)

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
FOLD="fold0"
FOLD_FILE=${HOME}/sds/sd17d003/Anamaria/borzoi_folds/split_genes_${FOLD}.tsv

# Split the data
SUBSET="adult"
KB="10"
SPECIES="mouse_human"
INPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data_new/processed_${SUBSET}_${KB}kb/
OUTPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data_new/splits_${SUBSET}_${KB}kb/borzoi/${FOLD}

python ${SPLICEVO_DIR}/scripts/data_split_borzoi.py \
    --input_dir ${INPUT_DIR} --genome_ids mouse_GRCm38 human_GRCh37 \
    --output_dir ${OUTPUT_DIR} \
    --folds_file ${FOLD_FILE} \
    --quiet &
