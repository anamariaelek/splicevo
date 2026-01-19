#!/bin/bash
#SBATCH --job-name=train_mouse_rat_human
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#
# Optimized for A100 80GB with full dataset (257k samples, 99 conditions)
# Memory requirements:
# - GPU (batch=8, seq=5900, 99 conditions, fp16): ~20-25 GB
# - RAM (6 workers * 2 prefetch * batch_8): ~15-20 GB
# - Model + optimizer states: ~10-15 GB GPU
# Total: ~45-55 GB GPU, ~30-40 GB RAM
# 
# Helix GPU options:
# - A40 (48 GB):   --gres=gpu:A40:1
# - A100 (40 GB):  --gres=gpu:A100:1
# - A100 (80 GB):  --gres=gpu:A100:1 (current)
# - H200 (141 GB): --gres=gpu:H200:1

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module before activating conda environment
module load devel/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Fix PyTorch memory fragmentation (reduces reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
KB="5"
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
