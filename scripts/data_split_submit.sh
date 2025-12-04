#!/bin/bash
#SBATCH --job-name=splicevo_data_split
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
ORTHOLOGY_FILE=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/ortholog_groups.tsv
INPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/processed_smallest/

# Split the data
OUTPUT_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_smallest/mouse_rat_human/
python ${SPLICEVO_DIR}/scripts/data_split.py \
    --input_dir ${INPUT_DIR} --genome_ids mouse_GRCm38 rat_Rnor_5.0 human_GRCh37 \
    --output_dir ${OUTPUT_DIR} \
    --orthology_file ${ORTHOLOGY_FILE} \
    --pov_genome mouse_GRCm38 \
    --test_chromosomes 18 \
    --quiet &
