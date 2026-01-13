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
SPECIES="mouse_rat_human"
KB="1"
MODEL="${SPECIES}_${KB}kb"
WINDOW=400
N_CORES=4
BATCH_SIZE=1000

# Paths
BASE_DIR="/home/elek/sds/sd17d003/Anamaria/splicevo"
MODEL_PATH="${BASE_DIR}/models/transformer/full_${MODEL}/best_model.pt"
DATA_PATH="${BASE_DIR}/data/splits_full_${KB}kb/${SPECIES}/test"
PREDICTIONS_PATH="${BASE_DIR}/predictions/transformer/full_${MODEL}/"
OUTPUT_DIR="${BASE_DIR}/attributions/transformer/${MODEL}_window_${WINDOW}"

# Optionally calculate attributions for a subset of sequences
# SUBSET should look like "0,1,2" or "0:10"
SUBSET=""
if [ "$SUBSET" != "" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_subset_${SUBSET//:/-}"
fi
echo "Attribution output directory: $OUTPUT_DIR"

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
if [ "$SUBSET" != "" ]; then
    python ${SPLICEVO_DIR}/scripts/splicevo_attributions.py \
        --model $MODEL_PATH \
        --data $DATA_PATH \
        --predictions $PREDICTIONS_PATH \
        --sequences $SUBSET \
        --window $WINDOW \
        --output $OUTPUT_DIR \
        --share-attributions-across-conditions
else
    # Process in batches to avoid OOM with large datasets
    # Count sequences from metadata.csv (subtract 1 for header)
    TOTAL_SEQUENCES=$(awk 'END {print NR-1}' "${DATA_PATH}/metadata.csv")
    echo "Total sequences: $TOTAL_SEQUENCES"
    echo "Batch size: $BATCH_SIZE"
    
    for ((START=0; START<TOTAL_SEQUENCES; START+=BATCH_SIZE)); do
        END=$((START + BATCH_SIZE))
        if [ $END -gt $TOTAL_SEQUENCES ]; then
            END=$TOTAL_SEQUENCES
        fi
        
        echo ""
        echo "Processing batch: sequences $START to $END"
        
        python ${SPLICEVO_DIR}/scripts/splicevo_attributions.py \
            --model $MODEL_PATH \
            --data $DATA_PATH \
            --predictions $PREDICTIONS_PATH \
            --sequences "${START}:${END}" \
            --window $WINDOW \
            --output "${OUTPUT_DIR}/batch_${START}_${END}" \
            --share-attributions-across-conditions \
            --skip-splice-attributions
        
        if [ $? -ne 0 ]; then
            echo "Error processing batch $START to $END"
            exit 1
        fi
    done
    
    echo ""
    echo "All batches completed successfully."
    echo ""
    echo "Merging batch results..."
    python ${SPLICEVO_DIR}/scripts/splicevo_attributions_merge_batches.py --input ${OUTPUT_DIR}
    
    if [ $? -eq 0 ]; then
        echo "Merge completed successfully."
        echo "Removing individual batch directories to save space..."
        rm -rf ${OUTPUT_DIR}/batch_*/
        echo "Cleanup complete. Final results are in: ${OUTPUT_DIR}"
    else
        echo "Warning: Merge failed. Batch directories preserved for debugging."
        exit 1
    fi
fi
