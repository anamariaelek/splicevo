#!/bin/bash
# 
# This script generates predictions on test data using a pre-trained Splicevo model.
#
#SBATCH --job-name=predict
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A80:1
#SBATCH --mem=80gb
#SBATCH --time=6:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Activate conda environment
conda activate splicevo

# Cuda
module load devel/cuda
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
SUBSET="full"
SPECIES="mouse_rat_human"
KB="1"
MODEL=${SUBSET}_${SPECIES}_${KB}kb_dynloss_focal

DATA_TEST_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}_${KB}kb/${SPECIES}/test/
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/models/transformer/${MODEL}/
PREDICTIONS_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/predictions/transformer/${MODEL}/
echo "Starting prediction job at "$(date)
echo "Test data: ${DATA_TEST_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Predictions directory: ${PREDICTIONS_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
    --checkpoint ${MODEL_DIR}/best_model.pt \
    --test-data ${DATA_TEST_DIR} \
    --output ${PREDICTIONS_DIR} \
    --batch-size 32 \
    --quiet

echo "Predict completed at "$(date)
