"""YOLO training pipeline for MSX plasmodia detection."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.core.config import Config
from src.core.paths import Paths

logger = logging.getLogger(__name__)

# MSX class mapping: maintain consistency across training and inference
CLASS_NAMES: dict[int, str] = {
    0: "Distractor",
    1: "Plasmodia",
}


@dataclass
class TrainingResult:
    """Training run summary.

    Args:
        best_weights: Path to best.pt weights file.
        final_weights: Path to last.pt weights file.
        epochs_completed: Number of epochs trained.
        best_map50: Best mAP@0.5 achieved.
        best_map50_95: Best mAP@0.5:0.95 achieved.
        wandb_run_id: W&B run ID if tracking enabled, else None.
    """

    best_weights: Path
    final_weights: Path
    epochs_completed: int
    best_map50: float
    best_map50_95: float
    wandb_run_id: str | None


class YOLOTrainer:
    """YOLO training wrapper for MSX plasmodia detection.

    Handles dataset validation, W&B integration, and weight management.
    Enforces the MSX class mapping (0: Distractor, 1: Plasmodia).
    """

    def __init__(
        self,
        dataset_path: Path | None = None,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 512,
        base_model: str = "yolov8n.pt",
    ) -> None:
        """Initialize the trainer.

        Args:
            dataset_path: Root directory of YOLO dataset. If None, reads
                          from config.yaml yolo.dataset_path.
            epochs: Training epochs.
            batch_size: Batch size (adjust for GPU memory).
            imgsz: Input image size.
            base_model: Pretrained YOLO model to fine-tune.
        """
        self._config = Config.get()
        self._paths = Paths.get()

        self._epochs = epochs
        self._batch_size = batch_size
        self._imgsz = imgsz
        self._base_model = base_model

        # Resolve dataset path from config if not provided
        self._dataset_path = dataset_path or self._load_dataset_path()
        self._dataset_yaml = self._dataset_path / "dataset.yaml"

        # Output directory for trained weights
        self._output_dir = self._paths.models_dir / "yolo_scout"

        # W&B configuration
        self._use_wandb = self._load_wandb_config()
        self._wandb_run_id: str | None = None

    def _load_dataset_path(self) -> Path:
        """Load dataset path from config.yaml."""
        config_path = self._paths.project_root / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}. "
                "Set yolo.dataset_path or pass dataset_path directly."
            )

        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        yolo_config = data.get("yolo", {})
        dataset_path = yolo_config.get("dataset_path")

        if not dataset_path:
            raise ValueError(
                "yolo.dataset_path not set in config.yaml. "
                "Expected: yolo:\\n  dataset_path: /path/to/dataset"
            )

        return Path(dataset_path).expanduser()

    def _load_wandb_config(self) -> bool:
        """Check if W&B tracking is enabled in config."""
        config_path = self._paths.project_root / "config.yaml"

        if not config_path.exists():
            return False

        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return data.get("yolo", {}).get("use_wandb", False)

    def _validate_dataset_structure(self) -> None:
        """Verify required dataset directories exist."""
        required_dirs = [
            self._dataset_path / "images" / "train",
            self._dataset_path / "images" / "val",
            self._dataset_path / "labels" / "train",
            self._dataset_path / "labels" / "val",
        ]

        missing = [d for d in required_dirs if not d.exists()]

        if missing:
            raise FileNotFoundError(
                f"Dataset structure incomplete. Missing: {missing}"
            )

        # Count training samples
        train_images = list((self._dataset_path / "images" / "train").glob("*.png"))
        train_images += list((self._dataset_path / "images" / "train").glob("*.jpg"))
        val_images = list((self._dataset_path / "images" / "val").glob("*.png"))
        val_images += list((self._dataset_path / "images" / "val").glob("*.jpg"))

        logger.info(
            f"Dataset validated: {len(train_images)} train, {len(val_images)} val images"
        )

        if len(train_images) == 0:
            raise ValueError("No training images found in images/train/")

    def _generate_dataset_yaml(self) -> Path:
        """Generate dataset.yaml with absolute paths and class mapping."""
        self._validate_dataset_structure()

        dataset_config = {
            "path": str(self._dataset_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES,
        }

        with open(self._dataset_yaml, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset.yaml: {self._dataset_yaml}")
        return self._dataset_yaml

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases tracking."""
        if not self._use_wandb:
            return

        try:
            import wandb

            run = wandb.init(
                project="msx-sentinel",
                name=f"yolo-scout-{self._epochs}ep-{self._imgsz}px",
                config={
                    "epochs": self._epochs,
                    "batch_size": self._batch_size,
                    "imgsz": self._imgsz,
                    "base_model": self._base_model,
                    "dataset": str(self._dataset_path),
                },
                tags=["yolo", "detection", "msx"],
            )
            self._wandb_run_id = run.id
            logger.info(f"W&B initialized: run {self._wandb_run_id}")

        except ImportError:
            logger.warning("wandb not installed. Skipping W&B integration.")
            self._use_wandb = False
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Continuing without tracking.")
            self._use_wandb = False

    def train(self) -> TrainingResult:
        """Execute YOLO training.

        Returns:
            TrainingResult with paths to weights and metrics.
        """
        # Prepare dataset config
        self._generate_dataset_yaml()

        # Initialize tracking
        self._init_wandb()

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Load base model
        model = YOLO(self._base_model)
        logger.info(f"Loaded base model: {self._base_model}")

        # Train
        logger.info(
            f"Starting training: {self._epochs} epochs, "
            f"batch={self._batch_size}, imgsz={self._imgsz}"
        )

        results = model.train(
            data=str(self._dataset_yaml),
            epochs=self._epochs,
            batch=self._batch_size,
            imgsz=self._imgsz,
            project=str(self._output_dir.parent),
            name=self._output_dir.name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
        )

        # Locate output weights
        run_dir = Path(results.save_dir)
        weights_dir = run_dir / "weights"

        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        if not best_pt.exists():
            raise FileNotFoundError(f"Training failed: best.pt not found at {best_pt}")

        # Copy best weights to canonical location
        canonical_best = self._output_dir / "best.pt"
        shutil.copy2(best_pt, canonical_best)
        logger.info(f"Best weights saved: {canonical_best}")

        # Extract metrics from results
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        best_map50 = metrics.get("metrics/mAP50(B)", 0.0)
        best_map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

        # Finalize W&B
        if self._use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        return TrainingResult(
            best_weights=canonical_best,
            final_weights=last_pt if last_pt.exists() else canonical_best,
            epochs_completed=self._epochs,
            best_map50=best_map50,
            best_map50_95=best_map50_95,
            wandb_run_id=self._wandb_run_id,
        )

    def resume(self, checkpoint: Path | None = None) -> TrainingResult:
        """Resume training from a checkpoint.

        Args:
            checkpoint: Path to checkpoint weights. If None, uses
                        yolo_scout/last.pt.

        Returns:
            TrainingResult with updated metrics.
        """
        if checkpoint is None:
            checkpoint = self._output_dir / "weights" / "last.pt"

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        self._generate_dataset_yaml()
        self._init_wandb()

        model = YOLO(str(checkpoint))
        logger.info(f"Resuming from checkpoint: {checkpoint}")

        results = model.train(
            data=str(self._dataset_yaml),
            epochs=self._epochs,
            batch=self._batch_size,
            imgsz=self._imgsz,
            project=str(self._output_dir.parent),
            name=self._output_dir.name,
            exist_ok=True,
            resume=True,
        )

        run_dir = Path(results.save_dir)
        weights_dir = run_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        canonical_best = self._output_dir / "best.pt"
        shutil.copy2(best_pt, canonical_best)

        metrics = results.results_dict if hasattr(results, "results_dict") else {}

        if self._use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        return TrainingResult(
            best_weights=canonical_best,
            final_weights=last_pt if last_pt.exists() else canonical_best,
            epochs_completed=self._epochs,
            best_map50=metrics.get("metrics/mAP50(B)", 0.0),
            best_map50_95=metrics.get("metrics/mAP50-95(B)", 0.0),
            wandb_run_id=self._wandb_run_id,
        )

    def export_onnx(self, weights: Path | None = None) -> Path:
        """Export trained model to ONNX format.

        Args:
            weights: Path to weights file. Defaults to best.pt.

        Returns:
            Path to exported ONNX file.
        """
        if weights is None:
            weights = self._output_dir / "best.pt"

        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        model = YOLO(str(weights))
        export_path = model.export(format="onnx", imgsz=self._imgsz)

        logger.info(f"Exported ONNX model: {export_path}")
        return Path(export_path)
