#!/bin/bash
#SBATCH --job-name=msx_sentinel
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=def-agodbout

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-agodbout/jezul/msx_sentinel"
SCRATCH_DIR="/home/jezul/scratch/oyster_data"
SIF_PATH="${PROJECT_DIR}/msx_sentinel.sif"

# Input paths (override via environment or defaults)
WSI_PATH="${WSI_PATH:-${SCRATCH_DIR}/slides/sample.ome.tiff}"
MASK_PATH="${MASK_PATH:-${SCRATCH_DIR}/masks/sample_mask.png}"
CLIENT_ID="${CLIENT_ID:-default}"

# Derive WSI name for output
WSI_NAME=$(basename "${WSI_PATH}" .ome.tiff)
OUTPUT_DIR="${PROJECT_DIR}/data/reports"

# ─── Environment Setup ────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[INFO] WSI: ${WSI_PATH}"
echo "[INFO] Mask: ${MASK_PATH}"

# Load modules
module load apptainer/1.1
module load cuda/11.8

# Persistent Ollama model cache (survives job termination)
export OLLAMA_MODELS="${PROJECT_DIR}/ollama_models"
mkdir -p "${OLLAMA_MODELS}"

# Runtime state directory (job-local temp)
export OLLAMA_HOME="${SLURM_TMPDIR}/ollama_runtime"
mkdir -p "${OLLAMA_HOME}"

# W&B API key from .env if present
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/.env"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Python path for src imports
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ─── Ollama Sidecar ───────────────────────────────────────────────────────────
echo "[INFO] Starting Ollama server..."

# Start Ollama in background
ollama serve > "${SLURM_TMPDIR}/ollama.log" 2>&1 &
OLLAMA_PID=$!

# Wait for server to initialize
sleep 10

# Verify server is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[ERROR] Ollama failed to start. Check ${SLURM_TMPDIR}/ollama.log"
    cat "${SLURM_TMPDIR}/ollama.log"
    exit 1
fi
echo "[INFO] Ollama server ready (PID: ${OLLAMA_PID})"

# Pre-pull models (uses cached weights from OLLAMA_MODELS if available)
echo "[INFO] Pulling VLM model (llava:13b)..."
ollama pull llava:13b

echo "[INFO] Pulling LLM model (llama3:8b)..."
ollama pull llama3:8b

echo "[INFO] Models ready"

# ─── Pipeline Execution ───────────────────────────────────────────────────────
echo "[INFO] Launching MSX-Sentinel pipeline..."

apptainer exec --nv \
    --bind /project:/project \
    --bind /scratch:/scratch \
    --bind "${SLURM_TMPDIR}:/tmp" \
    --env OLLAMA_HOST=http://localhost:11434 \
    --env OLLAMA_MODELS="${OLLAMA_MODELS}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    "${SIF_PATH}" \
    python scripts/run_pipeline.py \
        "${WSI_PATH}" \
        "${MASK_PATH}" \
        --client-id "${CLIENT_ID}" \
        --output "${OUTPUT_DIR}/${WSI_NAME}_report.json" \
        --verbose

PIPELINE_EXIT=$?

# ─── Cleanup ──────────────────────────────────────────────────────────────────
echo "[INFO] Shutting down Ollama..."
kill "${OLLAMA_PID}" 2>/dev/null || true
wait "${OLLAMA_PID}" 2>/dev/null || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job complete (exit code: ${PIPELINE_EXIT})"
exit ${PIPELINE_EXIT}
