# MSX-Sentinel

Anatomically-aware diagnostic framework for MSX (*Haplosporidium nelsoni*) detection in oyster whole slide images (WSIs).

## Overview

MSX-Sentinel employs a multi-scale cascade architecture (1.25x → 10x → 40x) combining YOLO-based candidate detection with VLM morphological confirmation. The system correlates plasmodium detections with anatomical context and hemocyte density biomarkers to produce explainable diagnostic reports.

## Quick Start

```bash
# Setup
git clone https://github.com/JonathanZul/msx-sentinel.git
cd msx-sentinel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py /path/to/slide.ome.tiff /path/to/mask.png
```

## Pipeline Usage

The `run_pipeline.py` script orchestrates all 5 diagnostic phases:

```bash
python scripts/run_pipeline.py <wsi_path> <mask_path> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `wsi_path` | Path to OME-TIFF whole slide image (required for standalone/hpc modes) |
| `mask_path` | Path to tissue mask PNG (required for standalone/hpc modes) |
| `--mode` | Execution mode: `standalone` (default), `hpc`, or `local` |
| `--package` | Path to bridge package directory (required for `--mode local`) |
| `--client-id` | Multi-tenant identifier (default: "default") |
| `--output` | JSON report output path |
| `--skip-tiling` | Skip tile extraction (use existing) |
| `--skip-detection` | Skip YOLO inference |
| `--skip-biomarkers` | Skip hemocyte analysis |
| `--skip-vlm` | Skip VLM verification |
| `--force-vlm` | Re-verify all candidates, ignoring existing VLM results |
| `--vlm-min-confidence` | Min YOLO confidence for VLM (0.0-1.0, default: 0.0) |
| `--vlm-concurrency` | Max concurrent VLM requests (default: 1, recommend 4-8) |
| `-v, --verbose` | Enable debug logging |

### Examples

```bash
# Standalone: Full pipeline locally (default)
python scripts/run_pipeline.py slide.ome.tiff mask.png

# HPC mode: Run Phases 1-3, generate bridge package for Globus transfer
python scripts/run_pipeline.py slide.ome.tiff mask.png --mode hpc

# Local mode: Run Phases 4-5 from transferred bridge package
python scripts/run_pipeline.py --mode local --package data/bridge_packages/slide_20240115_120000

# Resume from existing tiles
python scripts/run_pipeline.py slide.ome.tiff mask.png --skip-tiling

# Fast VLM with concurrent requests and confidence filtering (5-10x speedup)
python scripts/run_pipeline.py slide.ome.tiff mask.png --vlm-concurrency 8 --vlm-min-confidence 0.5

# Re-verify all candidates with a different VLM model
python scripts/run_pipeline.py slide.ome.tiff mask.png --force-vlm --vlm-concurrency 8 --skip-tiling --skip-detection --skip-biomarkers

# Only run diagnosis on processed data
python scripts/run_pipeline.py slide.ome.tiff mask.png \
    --skip-tiling --skip-detection --skip-biomarkers --skip-vlm

# Custom output path
python scripts/run_pipeline.py slide.ome.tiff mask.png --output results/report.json
```

## Pipeline Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `TilingEngine` | Extract multi-scale tiles (1.25x, 10x, 40x) with tissue filtering |
| 2 | `YOLODetector` | Detect MSX candidates on 10x tiles |
| 3 | `BiomarkerAnalyzer` | Measure hemocyte density via blob detection |
| 4 | `VLMEye` | Morphological verification of candidates at 40x |
| 5 | `DiagnosticBrain` | LLM-based severity synthesis with RAG |

## Architecture

```
src/
├── core/           # Config, paths, multi-provider abstraction
│   ├── config.py   # YAML configuration loader
│   ├── paths.py    # Path resolution singleton
│   └── providers.py# VLM/LLM provider factory (Ollama, Anthropic, etc.)
├── hpc/            # Tier 1: Heavy compute (HPC-ready)
│   ├── tiling.py   # Multi-scale tile extraction
│   ├── detection.py# YOLO inference with Level 0 coord mapping
│   ├── biomarkers.py# Hemocyte density via OpenCV blob detection
│   ├── train.py    # YOLO training with W&B integration
│   └── dataset.py  # QuPath GeoJSON → YOLO format conversion
├── bridge/         # Tier 2: Data synchronization
│   └── manifest.py # SQLite manifest for tile metadata
├── local/          # Tier 3: Reasoning (local/API)
│   ├── reasoning.py# VLMEye: 40x patch morphology verification
│   └── brain.py    # DiagnosticBrain: LLM severity synthesis
└── scripts/
    └── run_pipeline.py # End-to-end orchestrator
```

### Execution Tiers

| Tier | Environment | Modules |
|------|-------------|---------|
| 1 (Heavy) | HPC/Siku | Tiling, YOLO, Hemocyte Density |
| 2 (Bridge) | Globus | Manifest + Tiles Packaging, Checksum Verification |
| 3 (Reasoning) | Local/macOS | VLM Eye, LLM Brain |

## Multi-Scale Coordinate System

All coordinates normalize to Level 0 (40x) resolution:

| Scale | Magnification | Downsample | Purpose |
|-------|---------------|------------|---------|
| Macro | 1.25x | 32x | Anatomical segmentation |
| Mid | 10x | 4x | YOLO detection, hemocyte density |
| High | 40x | 1x | VLM morphological confirmation |

## Severity Scale

| Score | Label | Criteria |
|-------|-------|----------|
| 0 | Negative | No plasmodia observed |
| 1 | Light | 1-5 plasmodia, localized |
| 2 | Moderate | Multiple organs affected |
| 3 | Heavy | Systemic infiltration |

## Requirements

- Python 3.11+
- For HPC: Apptainer, CUDA 11.8+
- For local VLM/LLM: Ollama running locally, or API keys

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `ultralytics`, `opencv-python`, `tifffile`, `httpx`, `rich`

## Configuration

Edit `config.yaml`:

```yaml
active_provider: ollama  # ollama | anthropic | openai | gemini

ollama:
  host: "http://localhost:11434"
  vlm_model: "llava:13b"
  llm_model: "llama3:8b"

yolo:
  dataset_path: "/path/to/yolo_dataset"
  use_wandb: true

debug:
  enabled: true
  dpi: 300
```

## Ollama Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Pull required models
ollama pull llava:13b
ollama pull llama3:8b

# Start server (runs on localhost:11434)
ollama serve
```

## Testing

```bash
pytest tests/ -v
```

## HPC Deployment

### Apptainer

```bash
apptainer build msx_sentinel.sif apptainer.def

apptainer exec --bind /scratch:/scratch msx_sentinel.sif \
    python scripts/run_pipeline.py /scratch/data/slide.ome.tiff /scratch/data/mask.png
```

### Docker

```bash
docker build -t msx-sentinel .
docker run -v $(pwd)/data:/app/data msx-sentinel \
    python scripts/run_pipeline.py /app/data/slide.ome.tiff /app/data/mask.png
```

## Data Transfer via Globus

For hybrid workflows where heavy compute runs on HPC and reasoning runs locally, use Globus for reliable, resumable transfers.

### Step 1: Run HPC Job

Submit the Slurm job to run Phases 1-3 (Tiling, YOLO, Biomarkers):

```bash
# On Siku HPC
sbatch scripts/siku_job.sh

# Or run directly with --mode hpc
python scripts/run_pipeline.py slide.ome.tiff mask.png --mode hpc
```

This generates a bridge package at `data/bridge_packages/{wsi_name}_{timestamp}/` containing:
- `tiles.db` — SQLite manifest with all tile metadata
- `processed/{wsi_name}/` — Extracted PNG tiles
- `metadata.json` — HPC results and pipeline state
- `checksum.txt` — SHA256 checksums for transfer verification

### Step 2: Transfer via Globus

1. Open [Globus Web App](https://app.globus.org/)
2. Set **Source Endpoint**: `alliancecan#siku`
3. Navigate to: `/project/def-agodbout/jezul/msx_sentinel/data/bridge_packages/`
4. Set **Destination Endpoint**: Your Globus Connect Personal endpoint
5. Select the package folder and click **Start**

**Verify transfer integrity** (optional):
```bash
cd /path/to/bridge_package
sha256sum -c checksum.txt
```

### Step 3: Run Local Reasoning

Run Phases 4-5 (VLM Eye, Diagnostic Brain) on the transferred package:

```bash
# On local machine
python scripts/run_pipeline.py --mode local --package data/bridge_packages/{wsi_name}_{timestamp}
```

### Globus Setup

**HPC (Siku):** Globus endpoint pre-configured on Alliance clusters (`alliancecan#siku`).

**Local Machine:** Install [Globus Connect Personal](https://www.globus.org/globus-connect-personal) to create a personal endpoint.

## Output

The pipeline generates:

- **Tiles**: `data/processed/{wsi_name}/{scale}/*.png`
- **Debug overlays**: `data/debug/{phase}/{wsi_name}/*.png`
- **JSON report**: `data/reports/{wsi_name}_report.json`
- **SQLite manifest**: `data/manifests/manifest.db`

## License

Polyform Noncommercial License 1.0.0. See [LICENSE](LICENSE).

This software is available for research, educational, and personal use. Commercial use requires a separate license agreement.
