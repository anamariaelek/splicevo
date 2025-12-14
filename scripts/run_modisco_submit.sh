#!/bin/bash
# 
# This script runs the full modisco analysis on saved attributions
#
#SBATCH --job-name=modisco_analysis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/modisco_%j.log
#SBATCH --error=logs/modisco_%j.err

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
ATTR_WINDOW=400
SLIDING_WINDOW_SIZE=12
FLANK_SIZE=5
TARGET_SEQLET_FDR=0.01
MAX_SEQLETS_PER_METACLUSTER=20000
MIN_METACLUSTER_SIZE=100
TRIM_TO_WINDOW_SIZE=12
INITIAL_FLANK_TO_ADD=5
MIN_PASSING_WINDOWS_FRAC=0.03
MAX_PASSING_WINDOWS_FRAC=0.2
N_CORES=4

# Paths
BASE_DIR="/home/elek/sds/sd17d003/Anamaria/splicevo"
ATTRIBUTION_DIR="${BASE_DIR}/attributions/${SPECIES}_weighted_mse_window_${ATTR_WINDOW}"
OUTPUT_DIR="${BASE_DIR}/tfmodisco/${SPECIES}_weighted_mse_window_${TRIM_TO_WINDOW_SIZE}_flank_${INITIAL_FLANK_TO_ADD}_fdr_${TARGET_SEQLET_FDR}"

# Create logs directory
mkdir -p logs

# Print configuration
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname: $(hostname)"
echo ""
echo "Configuration:"
echo "  Species: $SPECIES"
echo "  Attribution base: $ATTRIBUTION_DIR"
echo "  Modisco output: $OUTPUT_DIR"
echo ""
echo "Parameters:"
echo "  sliding_window_size: $SLIDING_WINDOW_SIZE"
echo "  flank_size: $FLANK_SIZE"
echo "  target_seqlet_fdr: $TARGET_SEQLET_FDR"
echo "  max_seqlets_per_metacluster: $MAX_SEQLETS_PER_METACLUSTER"
echo "  min_metacluster_size: $MIN_METACLUSTER_SIZE"
echo "  trim_to_window_size: $TRIM_TO_WINDOW_SIZE"
echo "  initial_flank_to_add: $INITIAL_FLANK_TO_ADD"
echo "  min_passing_windows_frac: $MIN_PASSING_WINDOWS_FRAC"
echo "  max_passing_windows_frac: $MAX_PASSING_WINDOWS_FRAC"
echo "  n_cores: $N_CORES"
echo ""

# Check if attribution directory exists
if [ ! -d "$ATTRIBUTION_DIR" ]; then
    echo "Error: Attribution directory not found at $ATTRIBUTION_DIR"
    exit 1
fi

# Run TF-MoDISco analysis
echo ""
echo "Running TF-MoDISco analysis on both splice and usage attributions..."

python ${SPLICEVO_DIR}/scripts/run_modisco_analysis.py \
    --attributions-base "$ATTRIBUTION_DIR" \
    --output "$OUTPUT_DIR" \
    --sliding-window-size "$SLIDING_WINDOW_SIZE" \
    --flank-size "$FLANK_SIZE" \
    --target-seqlet-fdr "$TARGET_SEQLET_FDR" \
    --max-seqlets-per-metacluster "$MAX_SEQLETS_PER_METACLUSTER" \
    --min-metacluster-size "$MIN_METACLUSTER_SIZE" \
    --trim-to-window-size "$TRIM_TO_WINDOW_SIZE" \
    --initial-flank-to-add "$INITIAL_FLANK_TO_ADD" \
    --min-passing-windows-frac "$MIN_PASSING_WINDOWS_FRAC" \
    --max-passing-windows-frac "$MAX_PASSING_WINDOWS_FRAC" \
    --n-cores "$N_CORES"

MODISCO_EXIT_CODE=$?

# Summary
echo ""
if [ $MODISCO_EXIT_CODE -eq 0 ]; then
    echo "TF-MoDISco analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "TF-MoDISco analysis failed with exit code $MODISCO_EXIT_CODE"
fi
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

exit $MODISCO_EXIT_CODE
