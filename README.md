# MSX-Sentinel

Anatomically-aware diagnostic framework for MSX (*Haplosporidium nelsoni*) detection in oyster whole slide images (WSIs).

## Overview

MSX-Sentinel employs a multi-scale cascade architecture (1.25x → 10x → 40x) combining YOLO-based candidate detection with VLM morphological confirmation. The system correlates plasmodium detections with anatomical context and hemocyte density biomarkers to produce explainable diagnostic reports.

## Architecture

```
src/
├── core/       # Configuration, paths, provider abstractions
├── hpc/        # Tiling, YOLO inference, hemocyte counting (Tier 1)
├── bridge/     # DVC, manifest sync, rsync logic (Tier 2)
└── local/      # VLM Eye, LLM Brain, Streamlit UI (Tier 3)
```

### Execution Tiers

| Tier | Environment | Modules |
|------|-------------|---------|
| 1 (Heavy) | HPC/Siku | Tiling, YOLO, Hemocyte Density |
| 2 (Bridge) | Both | DVC, Manifest Sync |
| 3 (Reasoning) | Local/macOS | VLM, LLM, UI |

## Multi-Scale Coordinate System

All coordinates normalize to Level 0 (40x) resolution:

| Scale | Magnification | Purpose |
|-------|---------------|---------|
| 1.25x | Macro | Anatomical segmentation |
| 10x | Mid | YOLO detection, hemocyte density |
| 40x | High | VLM morphological confirmation |

## Requirements

- Python 3.10+
- For HPC: Apptainer/Singularity, CUDA 11.8+
- For local: Ollama or API keys (Anthropic/OpenAI/Gemini)

## Setup

```bash
# Clone repository
git clone <repo-url>
cd msx-sentinel

# Create conda environment
conda env create -f environment.yml
conda activate msx-sentinel

# Copy and configure environment
cp .env.example .env
```

## Configuration

Edit `config.yaml` to set:
- `active_provider`: ollama | anthropic | openai | gemini
- Path aliases for your environment

## License

Polyform Noncommercial License 1.0.0. See [LICENSE](LICENSE).

This software is available for research, educational, and personal use. Commercial use requires a separate license agreement.
