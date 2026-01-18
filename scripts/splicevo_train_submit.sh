#!/bin/bash
#SBATCH --job-name=train_mouse_rat_human
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=40gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#
# Memory requirements for Option A (Transformer on full sequence):
# - Attention memory for batch=64, seq_len=5900: ~8.9 GB
# - With gradient accumulation (steps=4), effective batch=16: ~2.2 GB attention
# - Plus model params, gradients, optimizer states: ~10-15 GB total
# 
# Helix GPU options (see https://wiki.bwhpc.de/e/Helix/Hardware):
# - A40 (48 GB):   --gres=gpu:A40:1 (no fp64 constraint)
# - A100 (40 GB):  --gres=gpu:A100:1 --constraint=fp64
# - A100 (80 GB):  --gres=gpu:A100:1 --constraint=fp64 (will get 80GB if available)
# - H200 (141 GB): --gres=gpu:H200:1 --constraint=fp64
# 
# Alternative: Request by memory instead of type:
# --gres=gpu:1,gpumem_per_gpu:80GB --constraint=fp64

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
    echo ""
    echo "System has CUDA 13.0, but PyTorch was installed without CUDA support."
    echo "Reinstall PyTorch with CUDA 12.4 support (compatible with CUDA 13.0):"
    echo ""
    echo "  conda activate splicevo"
    echo "  pip uninstall torch torchvision torchaudio -y"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    echo ""
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
