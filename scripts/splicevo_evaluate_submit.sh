#!/bin/bash
#
# This script evaluates predictions on test data from pre-trained Splicevo models.
#
#SBATCH --job-name=evaluate_small
#SBATCH --partition=cpu-single
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Set number of threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
SUBSET="full"
SPECIES="mouse_rat_human"
KB="5"
MODEL=${SUBSET}_${SPECIES}_${KB}kb

DATA_TEST_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}_${KB}kb/${SPECIES}/test/
PREDICTIONS_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/predictions/transformer/${MODEL}/

echo "Starting evaluation job at "$(date)
echo "Test data: ${DATA_TEST_DIR}"
echo "Predictions directory: ${PREDICTIONS_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_evaluate.py \
    --test-data ${DATA_TEST_DIR} \
    --predictions ${PREDICTIONS_DIR} \
    --output ${PREDICTIONS_DIR}/evaluation/

echo "Evaluation completed at "$(date)
