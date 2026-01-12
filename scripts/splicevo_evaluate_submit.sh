#!/bin/bash
#
# This script evaluates predictions on test data from pre-trained Splicevo models.
#
#SBATCH --job-name=evaluate_mouse_rat_human
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Cuda
module load devel/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
SUBSET="full"
SPECIES="mouse_rat_human"
MODEL=${SUBSET}_${SPECIES}

DATA_TEST_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}/${SPECIES}/test/
PREDICTIONS_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/predictions/transformer/${MODEL}/

echo "Starting evaluation job at "$(date)
echo "Test data: ${DATA_TEST_DIR}"
echo "Predictions directory: ${PREDICTIONS_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_evaluate.py \
    --test-data ${DATA_TEST_DIR} \
    --predictions ${PREDICTIONS_DIR} \
    --output ${PREDICTIONS_DIR}/evaluation/

echo "Evaluation completed at "$(date)
