#!/bin/bash
#SBATCH --job-name=predict_mouse_rat_human
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
SUBSET="small"
SPECIES="mouse_rat_human"
LOSS="weighted_mse"
MODEL=${SUBSET}_${SPECIES}_${LOSS}

DATA_TEST_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}/${SPECIES}/test/
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/models/${MODEL}
PREDICTIONS_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/predictions/${MODEL}/

echo "Starting training job at "$(date)
echo "Test data: ${DATA_TEST_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Predictions directory: ${PREDICTIONS_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
    --checkpoint ${MODEL_DIR}/best_model.pt \
    --test-data ${DATA_TEST_DIR} \
    --output ${PREDICTIONS_DIR} \
    --use-memmap \
    --save-memmap \
    --batch-size 128

echo "Predict completed at "$(date)