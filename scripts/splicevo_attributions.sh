#!/bin/bash
# 
# This script computes attributions for splice sites using a pre-trained model.
# It utilizes the compute_attributions.py script to perform the computations.
# 
#SBATCH --job-name=contributions
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/contributions_%j.log
#SBATCH --error=logs/contributions_%j.err

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

# Configuration
SUBSET="full"
SPECIES="mouse_rat_human"
N_SEQS="all"
TRIM_FLANK=15
N_CORES=4

# Paths
BASE_DIR="/home/elek/sds/sd17d003/Anamaria/splicevo"
MODEL_PATH="${BASE_DIR}/models/${SUBSET}_${SPECIES}_weighted_mse/best_model.pt"
DATA_PATH="${BASE_DIR}/data/splits_${SUBSET}/${SPECIES}/test"
PREDICTIONS_PATH="${BASE_DIR}/predictions/${SUBSET}_${SPECIES}_weighted_mse"
OUTPUT_DIR="${BASE_DIR}/attributions/${SUBSET}_${SPECIES}_weighted_mse"

# Create logs directory
mkdir -p logs

# Print configuration
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname: $(hostname)"
echo ""
echo "Configuration:"
echo "  Subset: $SUBSET"
echo "  Species: $SPECIES"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Predictions: $PREDICTIONS_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Parameters:"
echo "  n_seqs: $N_SEQS"
echo "  n_cores: $N_CORES"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data not found at $DATA_PATH"
    exit 1
fi

# Check if predictions exist
if [ ! -d "$PREDICTIONS_PATH" ]; then
    echo "Error: Predictions not found at $PREDICTIONS_PATH"
    exit 1
fi

# Run analysis
python ${SPLICEVO_DIR}/scripts/splicevo_attributions.py \
    --model $MODEL_PATH \
    --data $DATA_PATH \
    --predictions $PREDICTIONS_PATH \
    --windows '0:1000' \
    --output $OUTPUT_DIR