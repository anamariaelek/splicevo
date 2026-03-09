#!/bin/bash
#SBATCH --job-name=split-small-50
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=10gb
#SBATCH --time=6:00:00
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
ORTHOLOGY_FILE=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/ortholog_groups.tsv

# Split the data
SUBSET="small"
SPECIES="mouse_human"
KB=""
if [ "$KB" != "" ]; then
    SUBSET="${SUBSET}_${KB}kb"
fi
INPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/processed_${SUBSET}/
OUTPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}/${SPECIES}/

python ${SPLICEVO_DIR}/scripts/data_split.py \
    --input_dir ${INPUT_DIR} --genome_ids mouse_GRCm38 human_GRCh37 \
    --output_dir ${OUTPUT_DIR} \
    --orthology_file ${ORTHOLOGY_FILE} \
    --pov_genome mouse_GRCm38 \
    --test_chromosomes 15 \
    --quiet
