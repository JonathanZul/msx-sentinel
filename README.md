# MSX-Sentinel

Anatomically-aware diagnostic framework for MSX (*Haplosporidium nelsoni*) detection in oyster whole slide images (WSIs).

## Overview

MSX-Sentinel employs a multi-scale cascade architecture (1.25x → 10x → 40x) combining YOLO-based candidate detection with VLM morphological confirmation. The system correlates plasmodium detections with anatomical context and hemocyte density biomarkers to produce explainable diagnostic reports.

## Quick Start

```bash
# Setup
git clone <repo-url>
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
| `wsi_path` | Path to OME-TIFF whole slide image |
| `mask_path` | Path to tissue mask PNG (grayscale, non-zero = tissue) |
| `--client-id` | Multi-tenant identifier (default: "default") |
| `--output` | JSON report output path |
| `--skip-tiling` | Skip tile extraction (use existing) |
| `--skip-detection` | Skip YOLO inference |
| `--skip-biomarkers` | Skip hemocyte analysis |
| `--skip-vlm` | Skip VLM verification |
| `-v, --verbose` | Enable debug logging |

### Examples

```bash
# Full pipeline
python scripts/run_pipeline.py slide.ome.tiff mask.png

# Resume from existing tiles
python scripts/run_pipeline.py slide.ome.tiff mask.png --skip-tiling

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
| 2 (Bridge) | Both | Manifest Sync |
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

## Output

The pipeline generates:

- **Tiles**: `data/processed/{wsi_name}/{scale}/*.png`
- **Debug overlays**: `data/debug/{phase}/{wsi_name}/*.png`
- **JSON report**: `data/reports/{wsi_name}_report.json`
- **SQLite manifest**: `data/manifests/manifest.db`

## License

Polyform Noncommercial License 1.0.0. See [LICENSE](LICENSE).

This software is available for research, educational, and personal use. Commercial use requires a separate license agreement.
