#!/bin/bash
#SBATCH --job-name=train_mouse_rat_human
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#
# Memory requirements for Option A (Transformer on full sequence):
# - Attention memory for batch=64, seq_len=5900: ~8.9 GB
# - With gradient accumulation (steps=4), effective batch=16: ~2.2 GB attention
# - Plus model params, gradients, optimizer states: ~10-15 GB total
# - A100 80GB recommended for safety, A40 40GB may work with gradient accumulation
# - For A40 40GB, use: --gres=gpu:a40:1 (remove --constraint=gpu80)
# - For H200 141GB, use: --gres=gpu:h200:1 --constraint=gpu141

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module BEFORE activating conda environment
module load devel/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Activate conda environment
conda activate splicevo

# Verify CUDA setup
echo "CUDA setup verification:"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
nvidia-smi
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Exit if CUDA is not available
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    echo "Please reinstall PyTorch with CUDA support:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

# Splicevo directory
SPLICEVO_DIR=${HOME}/projects/splicevo/

# Inputs
SUBSET="full"
SPECIES="mouse_rat_human"
KB="1"
TRAINING_CONFIG=${HOME}/projects/splicevo/configs/training_transformer.yaml
DATA_TRAIN_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/data/splits_${SUBSET}_${KB}kb/${SPECIES}/train/
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/splicevo/models/transformer/${SUBSET}_${SPECIES}_${KB}kb/
echo "Starting training job at "$(date)
echo "Training config: ${TRAINING_CONFIG}"
echo "Training data: ${DATA_TRAIN_DIR}"
echo "Model directory: ${MODEL_DIR}"

# Train the model
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \
    --config ${TRAINING_CONFIG} \
    --data ${DATA_TRAIN_DIR} \
    --checkpoint-dir ${MODEL_DIR} \
    --quiet

echo "Training completed at "$(date)
