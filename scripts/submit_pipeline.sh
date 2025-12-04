#!/bin/bash
#
# SplicEvo SLURM Pipeline Submission Script
# Submits a series of dependent jobs: load -> split -> train -> predict
#
# Usage: bash submit_pipeline.sh [OPTIONS]
#
# Options:
#   --start-from STEP    Start pipeline from: load|split|train|predict (default: load)
#   --dry-run           Print jobs without submitting
#

set -e  # Exit on error

# Configuration - MODIFY THESE
SPLICEVO_DIR="/home/hd/hd_hd/hd_mf354/projects/splicevo"
OUT_DIR="/home/hd/hd_hd/hd_mf354/projects/splicing/results"

# Config files
DATA_CONFIG="${SPLICEVO_DIR}/configs/data_human_mouse.json"
TRAINING_CONFIG="${SPLICEVO_DIR}/configs/training_resnet.yaml"

# Data directories
LOAD_DIR="${OUT_DIR}/data/load/hsap_mmus"
SPLIT_DIR="${OUT_DIR}/data/split/hsap_mmus"
DATA_TRAIN_DIR="${SPLIT_DIR}/memmap_train"
DATA_TEST_DIR="${SPLIT_DIR}/memmap_test"
MODEL_DIR="${OUT_DIR}/models/resnet_hybridloss"
PREDICTIONS_DIR="${OUT_DIR}/predictions/resnet_hybridloss"

# SLURM job parameters
PARTITION_CPU="cpu-single"  # Change to your partition name
PARTITION_GPU="gpu-single"  # Change to your GPU partition name
ACCOUNT=""  # Change if needed
TIME_LOAD="24:00:00"
TIME_SPLIT="24:00:00"
TIME_TRAIN="72:00:00"
TIME_PREDICT="12:00:00"
MEM_LOAD="64G"
MEM_SPLIT="128G"
MEM_TRAIN="128G"
MEM_PREDICT="32G"
CPUS_LOAD="16"
CPUS_SPLIT="4"
CPUS_TRAIN="8"
CPUS_PREDICT="4"
GPUS_TRAIN="1"
NTASKS_PER_NODE_TRAIN="10"
GPUS_PREDICT="1"
NTASKS_PER_NODE_PREDICT="10"

# Split parameters
POV_GENOME="human_GRCh37"
TEST_CHROMOSOMES="1 3 5"
WINDOW_SIZE="1000"
CONTEXT_SIZE="450"
ALPHA_THRESHOLD="5"

# Prediction parameters
BATCH_SIZE="64"

# Parse arguments
DRY_RUN=false
START_FROM="load"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--start-from load|split|train|predict]"
            exit 1
            ;;
    esac
done

# Validate start-from option
if [[ ! "$START_FROM" =~ ^(load|split|train|predict)$ ]]; then
    echo "Error: --start-from must be one of: load, split, train, predict"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Jobs will not be submitted"
fi

echo "Starting pipeline from: ${START_FROM}"

# Create log directory
LOG_DIR="${OUT_DIR}/logs/pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Pipeline submission starting..."
echo "Logs will be saved to: ${LOG_DIR}"

# Function to submit job
submit_job() {
    local job_name=$1
    local dependency=$2
    local script_content=$3
    
    local script_file="${LOG_DIR}/${job_name}.sh"
    echo "$script_content" > "$script_file"
    chmod +x "$script_file"
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would submit: ${job_name}"
        echo "  Dependency: ${dependency}"
        echo "  Script: ${script_file}"
        echo ""
        return 0
    fi
    
    if [ -z "$dependency" ]; then
        sbatch "$script_file"
    else
        sbatch --dependency=afterok:${dependency} "$script_file"
    fi
}

# Track job IDs
LOAD_JOB=""
SPLIT_JOB=""
TRAIN_JOB=""
PREDICT_JOB=""

# ============================================================================
# JOB 1: Load data
# ============================================================================
if [[ "$START_FROM" == "load" ]]; then
    LOAD_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=splicevo_load
#SBATCH --output=${LOG_DIR}/load_%j.out
#SBATCH --error=${LOG_DIR}/load_%j.err
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --time=${TIME_LOAD}
#SBATCH --mem=${MEM_LOAD}
#SBATCH --cpus-per-task=${CPUS_LOAD}
$([ -n "$ACCOUNT" ] && echo "#SBATCH --account=${ACCOUNT}")

set -e

# Initialize conda for bash shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate splicevo

echo "Starting data load job at \$(date)"
echo "Loading data from: ${DATA_CONFIG}"
echo "Output directory: ${LOAD_DIR}"

python ${SPLICEVO_DIR}/scripts/data_load.py \\
    --config ${DATA_CONFIG} \\
    --output_dir ${LOAD_DIR} \\
    --n_cpus ${CPUS_LOAD}

echo "Data load completed at \$(date)"
EOF
)

    echo "Submitting job 1: Load data..."
    LOAD_JOB=$(submit_job "load" "" "$LOAD_SCRIPT" | tail -n1 | awk '{print $NF}')
    echo "  Job ID: ${LOAD_JOB}"
else
    echo "Skipping job 1: Load data (starting from ${START_FROM})"
fi

# ============================================================================
# JOB 2: Split data
# ============================================================================
if [[ "$START_FROM" =~ ^(load|split)$ ]]; then
    SPLIT_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=splicevo_split
#SBATCH --output=${LOG_DIR}/split_%j.out
#SBATCH --error=${LOG_DIR}/split_%j.err
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --time=${TIME_SPLIT}
#SBATCH --mem=${MEM_SPLIT}
#SBATCH --cpus-per-task=${CPUS_SPLIT}
$([ -n "$ACCOUNT" ] && echo "#SBATCH --account=${ACCOUNT}")

set -e

# Initialize conda for bash shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate splicevo

echo "Starting data split job at \$(date)"
echo "Input directory: ${LOAD_DIR}"
echo "Output directory: ${SPLIT_DIR}"

python ${SPLICEVO_DIR}/scripts/data_split.py \\
    --input_dir ${LOAD_DIR} \\
    --output_dir ${SPLIT_DIR} \\
    --n_cpus ${CPUS_SPLIT} \\
    --pov_genome ${POV_GENOME} \\
    --test_chromosomes ${TEST_CHROMOSOMES} \\
    --window_size ${WINDOW_SIZE} \\
    --context_size ${CONTEXT_SIZE} \\
    --alpha_threshold ${ALPHA_THRESHOLD}

echo "Data split completed at \$(date)"
EOF
)

    echo "Submitting job 2: Split data..."
    if [[ "$START_FROM" == "load" && -n "$LOAD_JOB" ]]; then
        SPLIT_JOB=$(submit_job "split" "$LOAD_JOB" "$SPLIT_SCRIPT" | tail -n1 | awk '{print $NF}')
        echo "  Job ID: ${SPLIT_JOB}"
        echo "  Depends on: ${LOAD_JOB}"
    else
        SPLIT_JOB=$(submit_job "split" "" "$SPLIT_SCRIPT" | tail -n1 | awk '{print $NF}')
        echo "  Job ID: ${SPLIT_JOB}"
    fi
else
    echo "Skipping job 2: Split data (starting from ${START_FROM})"
fi

# ============================================================================
# JOB 3: Train model
# ============================================================================
if [[ "$START_FROM" =~ ^(load|split|train)$ ]]; then
    TRAIN_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=splicevo_train
#SBATCH --output=${LOG_DIR}/train_%j.out
#SBATCH --error=${LOG_DIR}/train_%j.err
#SBATCH --partition=${PARTITION_GPU}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE_TRAIN}
#SBATCH --gres=gpu:A40:1
#SBATCH --time=${TIME_TRAIN}
#SBATCH --mem=${MEM_TRAIN}
#SBATCH --cpus-per-task=${CPUS_TRAIN}
#SBATCH --gres=gpu:${GPUS_TRAIN}
$([ -n "$ACCOUNT" ] && echo "#SBATCH --account=${ACCOUNT}")

set -e

# Initialize conda for bash shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate splicevo

module load devel/cuda
export OMP_NUM_THREADS=${SLURM_NTASKS}
echo "Starting training job at \$(date)"
echo "Training data: ${DATA_TRAIN_DIR}"
echo "Model directory: ${MODEL_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_train.py \\
    --config ${TRAINING_CONFIG} \\
    --data ${DATA_TRAIN_DIR} \\
    --checkpoint-dir ${MODEL_DIR}

echo "Training completed at \$(date)"
EOF
)

    echo "Submitting job 3: Train model..."
    if [[ "$START_FROM" =~ ^(load|split)$ && -n "$SPLIT_JOB" ]]; then
        TRAIN_JOB=$(submit_job "train" "$SPLIT_JOB" "$TRAIN_SCRIPT" | tail -n1 | awk '{print $NF}')
        echo "  Job ID: ${TRAIN_JOB}"
        echo "  Depends on: ${SPLIT_JOB}"
    else
        TRAIN_JOB=$(submit_job "train" "" "$TRAIN_SCRIPT" | tail -n1 | awk '{print $NF}')
        echo "  Job ID: ${TRAIN_JOB}"
    fi
else
    echo "Skipping job 3: Train model (starting from ${START_FROM})"
fi

# ============================================================================
# JOB 4: Predict
# ============================================================================
PREDICT_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=splicevo_predict
#SBATCH --output=${LOG_DIR}/predict_%j.out
#SBATCH --error=${LOG_DIR}/predict_%j.err
#SBATCH --partition=${PARTITION_GPU}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE_PREDICT}
#SBATCH --gres=gpu:A40:1
#SBATCH --time=${TIME_PREDICT}
#SBATCH --mem=${MEM_PREDICT}
#SBATCH --cpus-per-task=${CPUS_PREDICT}
#SBATCH --gres=gpu:${GPUS_PREDICT}
$([ -n "$ACCOUNT" ] && echo "#SBATCH --account=${ACCOUNT}")

set -e

# Initialize conda for bash shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate conda environment
conda activate splicevo

echo "Starting prediction job at \$(date)"
echo "Test data: ${DATA_TEST_DIR}"
echo "Model checkpoint: ${MODEL_DIR}/best_model.pt"
echo "Predictions output: ${PREDICTIONS_DIR}"

python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \\
    --checkpoint ${MODEL_DIR}/best_model.pt \\
    --test-data ${DATA_TEST_DIR} \\
    --output ${PREDICTIONS_DIR} \\
    --use-memmap \\
    --save-memmap \\
    --batch-size ${BATCH_SIZE}

echo "Prediction completed at \$(date)"
EOF
)

echo "Submitting job 4: Predict..."
if [[ "$START_FROM" =~ ^(load|split|train)$ && -n "$TRAIN_JOB" ]]; then
    PREDICT_JOB=$(submit_job "predict" "$TRAIN_JOB" "$PREDICT_SCRIPT" | tail -n1 | awk '{print $NF}')
    echo "  Job ID: ${PREDICT_JOB}"
    echo "  Depends on: ${TRAIN_JOB}"
else
    PREDICT_JOB=$(submit_job "predict" "" "$PREDICT_SCRIPT" | tail -n1 | awk '{print $NF}')
    echo "  Job ID: ${PREDICT_JOB}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Pipeline submission complete!"
echo "============================================"
echo "Job chain (started from: ${START_FROM}):"
[[ -n "$LOAD_JOB" ]] && echo "  1. Load:    ${LOAD_JOB}"
[[ -n "$SPLIT_JOB" ]] && echo "  2. Split:   ${SPLIT_JOB}$([ -n "$LOAD_JOB" ] && echo " (after ${LOAD_JOB})")"
[[ -n "$TRAIN_JOB" ]] && echo "  3. Train:   ${TRAIN_JOB}$([ -n "$SPLIT_JOB" ] && echo " (after ${SPLIT_JOB})")"
[[ -n "$PREDICT_JOB" ]] && echo "  4. Predict: ${PREDICT_JOB}$([ -n "$TRAIN_JOB" ] && echo " (after ${TRAIN_JOB})")"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${LOG_DIR}"
echo ""
if [[ -n "$LOAD_JOB" || -n "$SPLIT_JOB" || -n "$TRAIN_JOB" || -n "$PREDICT_JOB" ]]; then
    echo "To cancel all jobs:"
    echo "  scancel$([ -n "$LOAD_JOB" ] && echo " ${LOAD_JOB}")$([ -n "$SPLIT_JOB" ] && echo " ${SPLIT_JOB}")$([ -n "$TRAIN_JOB" ] && echo " ${TRAIN_JOB}")$([ -n "$PREDICT_JOB" ] && echo " ${PREDICT_JOB}")"
fi
echo "============================================"
