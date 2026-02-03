#!/bin/bash
#SBATCH --job-name=msx_hpc
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=def-agodbout

# HPC Job: Phases 1-3 (Tiling, YOLO Detection, Biomarkers)
# Generates a bridge package for Globus transfer to local reasoning tier.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-agodbout/jezul/msx_sentinel"
SCRATCH_DIR="/home/jezul/scratch/oyster_data"
SIF_PATH="${PROJECT_DIR}/msx_sentinel.sif"

# Input paths (override via environment or defaults)
WSI_PATH="${WSI_PATH:-${SCRATCH_DIR}/oyster_data/raw/wsis/U18068-24 A_01.ome.tif}"
MASK_PATH="${MASK_PATH:-${SCRATCH_DIR}/oyster_data/interim/oyster_masks/U18068-24 A_01/oyster_1_mask.png}"
CLIENT_ID="${CLIENT_ID:-default}"

# ─── Environment Setup ────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[INFO] WSI: ${WSI_PATH}"
echo "[INFO] Mask: ${MASK_PATH}"
echo "[INFO] Mode: HPC (Phases 1-3)"

# Load modules
module load apptainer/1.1
module load cuda/11.8

# W&B API key from .env if present
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/.env"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Python path for src imports
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Create output directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/data/bridge_packages"

# ─── Pipeline Execution (Phases 1-3) ─────────────────────────────────────────
echo "[INFO] Launching MSX-Sentinel HPC pipeline (Tiling, YOLO, Biomarkers)..."

apptainer exec --nv \
    --bind /project:/project \
    --bind /scratch:/scratch \
    --bind "${SLURM_TMPDIR}:/tmp" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    "${SIF_PATH}" \
    python scripts/run_pipeline.py \
        "${WSI_PATH}" \
        "${MASK_PATH}" \
        --mode hpc \
        --client-id "${CLIENT_ID}" \
        --verbose

PIPELINE_EXIT=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job complete (exit code: ${PIPELINE_EXIT})"
echo "[INFO] Bridge package ready for Globus transfer at: ${PROJECT_DIR}/data/bridge_packages/"
exit ${PIPELINE_EXIT}
