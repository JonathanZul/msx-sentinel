"""Context-aware YOLO dataset generation from QuPath GeoJSON annotations."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.core.config import Config
from src.core.paths import Paths
from src.hpc.train import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Single point annotation from QuPath GeoJSON.

    Args:
        x: X coordinate at Level 0 (40x).
        y: Y coordinate at Level 0 (40x).
        class_name: Classification label (e.g., "Plasmodia").
        class_id: Mapped class index for YOLO.
    """

    x: int
    y: int
    class_name: str
    class_id: int


@dataclass
class DatasetStats:
    """Dataset generation summary.

    Args:
        total_positives_40x: Number of 40x positive patches.
        total_positives_10x: Number of 10x context patches.
        total_negatives: Number of negative samples.
        train_count: Training set size.
        val_count: Validation set size.
        wsi_count: Number of WSIs processed.
    """

    total_positives_40x: int
    total_positives_10x: int
    total_negatives: int
    train_count: int
    val_count: int
    wsi_count: int


class DatasetGenerator:
    """Context-aware YOLO dataset generator.

    Extracts jittered 40x patches and 10x context patches from annotated
    WSIs. Mines hard negatives from severity-0 slides.
    """

    # 40x patch parameters
    PATCH_SIZE_40X: int = 640
    MAX_JITTER: int = 200
    JITTER_COUNT: int = 2

    # 10x context parameters (extract 2048x2048 at 40x, downsample to 512)
    CONTEXT_SIZE_40X: int = 2048
    OUTPUT_SIZE_10X: int = 512
    DOWNSAMPLE_10X: float = 4.0

    # Estimated plasmodia bbox size at 40x (5-50µm, ~30px typical)
    PLASMODIA_SIZE_40X: int = 30

    # Negative mining
    NEGATIVES_PER_SLIDE: int = 100
    MASK_SCALE: int = 8

    # Train/val split
    VAL_FRACTION: float = 0.15

    def __init__(
        self,
        output_path: Path,
        seed: int = 42,
    ) -> None:
        """Initialize the dataset generator.

        Args:
            output_path: Root directory for YOLO dataset output.
            seed: Random seed for reproducibility.
        """
        self._output_path = output_path
        self._config = Config.get()
        self._paths = Paths.get()
        self._rng = random.Random(seed)

        # Class name to ID mapping (inverse of CLASS_NAMES)
        self._class_map: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

    def generate(
        self,
        wsi_paths: list[Path],
        annotation_paths: list[Path],
        mask_paths: list[Path],
        severity_labels: list[int],
    ) -> DatasetStats:
        """Generate complete YOLO dataset.

        Args:
            wsi_paths: List of WSI file paths.
            annotation_paths: Corresponding QuPath GeoJSON files.
            mask_paths: Corresponding tissue mask PNGs.
            severity_labels: MSX severity (0-3) for each WSI.

        Returns:
            DatasetStats with generation summary.
        """
        if not (len(wsi_paths) == len(annotation_paths) == len(mask_paths) == len(severity_labels)):
            raise ValueError("Input lists must have equal length")

        self._setup_directories()

        total_40x = 0
        total_10x = 0
        total_neg = 0
        all_samples: list[tuple[Path, Path, str]] = []  # (image, label, split)

        with self._progress_bar() as progress:
            task = progress.add_task("Processing WSIs", total=len(wsi_paths))

            for wsi_path, ann_path, mask_path, severity in zip(
                wsi_paths, annotation_paths, mask_paths, severity_labels
            ):
                progress.update(task, advance=1)

                wsi_name = wsi_path.stem
                logger.info(f"Processing {wsi_name} (severity={severity})")

                # Parse annotations
                annotations = self._parse_geojson(ann_path)

                with tifffile.TiffFile(wsi_path) as tif:
                    wsi_dims = (tif.pages[0].shape[1], tif.pages[0].shape[0])

                    # Generate positive samples from annotations
                    if annotations:
                        samples_40x = self._extract_positives_40x(
                            tif, wsi_name, annotations, wsi_dims
                        )
                        samples_10x = self._extract_positives_10x(
                            tif, wsi_name, annotations, wsi_dims
                        )
                        all_samples.extend(samples_40x)
                        all_samples.extend(samples_10x)
                        total_40x += len(samples_40x)
                        total_10x += len(samples_10x)

                    # Mine negatives from severity-0 slides
                    if severity == 0:
                        mask_array = self._load_mask(mask_path)
                        neg_samples = self._mine_negatives(
                            tif, wsi_name, mask_array, wsi_dims, annotations
                        )
                        all_samples.extend(neg_samples)
                        total_neg += len(neg_samples)

        # Split into train/val
        train_samples, val_samples = self._split_dataset(all_samples)

        # Move files to final locations
        self._organize_splits(train_samples, val_samples)

        stats = DatasetStats(
            total_positives_40x=total_40x,
            total_positives_10x=total_10x,
            total_negatives=total_neg,
            train_count=len(train_samples),
            val_count=len(val_samples),
            wsi_count=len(wsi_paths),
        )

        logger.info(
            f"Dataset complete: {stats.train_count} train, {stats.val_count} val "
            f"({total_40x} 40x, {total_10x} 10x, {total_neg} neg)"
        )

        return stats

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        for split in ["train", "val"]:
            (self._output_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self._output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Staging directory for initial extraction
        (self._output_path / "_staging" / "images").mkdir(parents=True, exist_ok=True)
        (self._output_path / "_staging" / "labels").mkdir(parents=True, exist_ok=True)

    def _parse_geojson(self, geojson_path: Path) -> list[Annotation]:
        """Parse QuPath GeoJSON export to annotations.

        Args:
            geojson_path: Path to GeoJSON file.

        Returns:
            List of Annotation objects.
        """
        if not geojson_path.exists():
            logger.warning(f"Annotation file not found: {geojson_path}")
            return []

        with open(geojson_path) as f:
            data = json.load(f)

        annotations: list[Annotation] = []

        for feature in data.get("features", []):
            geom = feature.get("geometry", {})

            # Only process Point annotations
            if geom.get("type") != "Point":
                continue

            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            x, y = int(coords[0]), int(coords[1])

            # Extract classification
            props = feature.get("properties", {})
            classification = props.get("classification", {})
            class_name = classification.get("name", "Unknown")

            # Map to class ID
            class_id = self._class_map.get(class_name, 0)

            annotations.append(Annotation(
                x=x,
                y=y,
                class_name=class_name,
                class_id=class_id,
            ))

        logger.debug(f"Parsed {len(annotations)} annotations from {geojson_path.name}")
        return annotations

    def _extract_positives_40x(
        self,
        tif: tifffile.TiffFile,
        wsi_name: str,
        annotations: list[Annotation],
        wsi_dims: tuple[int, int],
    ) -> list[tuple[Path, Path, str]]:
        """Extract jittered 40x patches around annotations.

        Returns:
            List of (image_path, label_path, "staging") tuples.
        """
        samples: list[tuple[Path, Path, str]] = []
        wsi_w, wsi_h = wsi_dims
        page = tif.pages[0]
        full_array = page.asarray()

        for ann in annotations:
            # Generate centered + jittered positions
            positions = self._generate_jittered_positions(
                ann.x, ann.y, wsi_w, wsi_h, self.PATCH_SIZE_40X
            )

            for idx, (x_patch, y_patch) in enumerate(positions):
                # Extract patch
                patch = full_array[
                    y_patch : y_patch + self.PATCH_SIZE_40X,
                    x_patch : x_patch + self.PATCH_SIZE_40X,
                ]

                if patch.shape[0] != self.PATCH_SIZE_40X or patch.shape[1] != self.PATCH_SIZE_40X:
                    continue

                img = self._array_to_pil(patch)

                # Compute YOLO label (annotation position relative to patch)
                x_rel = (ann.x - x_patch) / self.PATCH_SIZE_40X
                y_rel = (ann.y - y_patch) / self.PATCH_SIZE_40X
                w_norm = self.PLASMODIA_SIZE_40X / self.PATCH_SIZE_40X
                h_norm = self.PLASMODIA_SIZE_40X / self.PATCH_SIZE_40X

                # Skip if annotation falls outside patch bounds
                if not (0 < x_rel < 1 and 0 < y_rel < 1):
                    continue

                # Save to staging
                img_name = f"{wsi_name}_{x_patch}_{y_patch}_40x_j{idx}.png"
                lbl_name = f"{wsi_name}_{x_patch}_{y_patch}_40x_j{idx}.txt"

                img_path = self._output_path / "_staging" / "images" / img_name
                lbl_path = self._output_path / "_staging" / "labels" / lbl_name

                img.save(img_path, "PNG")
                self._write_yolo_label(lbl_path, ann.class_id, x_rel, y_rel, w_norm, h_norm)

                samples.append((img_path, lbl_path, "staging"))

        return samples

    def _extract_positives_10x(
        self,
        tif: tifffile.TiffFile,
        wsi_name: str,
        annotations: list[Annotation],
        wsi_dims: tuple[int, int],
    ) -> list[tuple[Path, Path, str]]:
        """Extract 10x context patches (2048→512 downsample).

        Returns:
            List of (image_path, label_path, "staging") tuples.
        """
        samples: list[tuple[Path, Path, str]] = []
        wsi_w, wsi_h = wsi_dims
        page = tif.pages[0]
        full_array = page.asarray()

        for ann in annotations:
            # Center context region on annotation
            x_ctx = ann.x - self.CONTEXT_SIZE_40X // 2
            y_ctx = ann.y - self.CONTEXT_SIZE_40X // 2

            # Clamp to WSI bounds
            x_ctx = max(0, min(x_ctx, wsi_w - self.CONTEXT_SIZE_40X))
            y_ctx = max(0, min(y_ctx, wsi_h - self.CONTEXT_SIZE_40X))

            # Extract large region
            region = full_array[
                y_ctx : y_ctx + self.CONTEXT_SIZE_40X,
                x_ctx : x_ctx + self.CONTEXT_SIZE_40X,
            ]

            if region.shape[0] != self.CONTEXT_SIZE_40X or region.shape[1] != self.CONTEXT_SIZE_40X:
                continue

            img = self._array_to_pil(region)
            img = img.resize(
                (self.OUTPUT_SIZE_10X, self.OUTPUT_SIZE_10X),
                Image.Resampling.LANCZOS,
            )

            # YOLO label: annotation position in downsampled image
            x_rel = (ann.x - x_ctx) / self.CONTEXT_SIZE_40X
            y_rel = (ann.y - y_ctx) / self.CONTEXT_SIZE_40X

            # Plasmodia appears larger relative to FOV at 10x
            w_norm = (self.PLASMODIA_SIZE_40X * self.DOWNSAMPLE_10X) / self.CONTEXT_SIZE_40X
            h_norm = w_norm

            if not (0 < x_rel < 1 and 0 < y_rel < 1):
                continue

            img_name = f"{wsi_name}_{x_ctx}_{y_ctx}_10x.png"
            lbl_name = f"{wsi_name}_{x_ctx}_{y_ctx}_10x.txt"

            img_path = self._output_path / "_staging" / "images" / img_name
            lbl_path = self._output_path / "_staging" / "labels" / lbl_name

            img.save(img_path, "PNG")
            self._write_yolo_label(lbl_path, ann.class_id, x_rel, y_rel, w_norm, h_norm)

            samples.append((img_path, lbl_path, "staging"))

        return samples

    def _mine_negatives(
        self,
        tif: tifffile.TiffFile,
        wsi_name: str,
        mask_array: np.ndarray,
        wsi_dims: tuple[int, int],
        annotations: list[Annotation],
    ) -> list[tuple[Path, Path, str]]:
        """Extract negative patches from tissue regions without annotations.

        Returns:
            List of (image_path, label_path, "staging") tuples.
        """
        samples: list[tuple[Path, Path, str]] = []
        wsi_w, wsi_h = wsi_dims
        page = tif.pages[0]
        full_array = page.asarray()

        # Build exclusion zones around annotations
        exclusion_radius = self.PATCH_SIZE_40X
        exclusion_set: set[tuple[int, int]] = set()

        for ann in annotations:
            # Grid cells to exclude (at mask resolution)
            cx = ann.x // self.MASK_SCALE
            cy = ann.y // self.MASK_SCALE
            r = exclusion_radius // self.MASK_SCALE

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    exclusion_set.add((cx + dx, cy + dy))

        # Find valid tissue positions
        mask_h, mask_w = mask_array.shape[:2]
        tissue_positions: list[tuple[int, int]] = []

        patch_cells = self.PATCH_SIZE_40X // self.MASK_SCALE

        for my in range(0, mask_h - patch_cells, patch_cells // 2):
            for mx in range(0, mask_w - patch_cells, patch_cells // 2):
                # Skip if in exclusion zone
                if (mx, my) in exclusion_set:
                    continue

                # Check tissue coverage
                region = mask_array[my : my + patch_cells, mx : mx + patch_cells]
                if region.size == 0:
                    continue

                tissue_frac = np.count_nonzero(region) / region.size
                if tissue_frac < 0.5:
                    continue

                # Convert to Level 0 coords
                x0 = mx * self.MASK_SCALE
                y0 = my * self.MASK_SCALE

                if x0 + self.PATCH_SIZE_40X <= wsi_w and y0 + self.PATCH_SIZE_40X <= wsi_h:
                    tissue_positions.append((x0, y0))

        # Sample negatives
        if len(tissue_positions) > self.NEGATIVES_PER_SLIDE:
            tissue_positions = self._rng.sample(tissue_positions, self.NEGATIVES_PER_SLIDE)

        for idx, (x0, y0) in enumerate(tissue_positions):
            patch = full_array[
                y0 : y0 + self.PATCH_SIZE_40X,
                x0 : x0 + self.PATCH_SIZE_40X,
            ]

            if patch.shape[0] != self.PATCH_SIZE_40X or patch.shape[1] != self.PATCH_SIZE_40X:
                continue

            img = self._array_to_pil(patch)

            img_name = f"{wsi_name}_neg_{idx:04d}.png"
            lbl_name = f"{wsi_name}_neg_{idx:04d}.txt"

            img_path = self._output_path / "_staging" / "images" / img_name
            lbl_path = self._output_path / "_staging" / "labels" / lbl_name

            img.save(img_path, "PNG")
            # Empty label file for negative (no objects)
            lbl_path.write_text("")

            samples.append((img_path, lbl_path, "staging"))

        return samples

    def _generate_jittered_positions(
        self,
        cx: int,
        cy: int,
        wsi_w: int,
        wsi_h: int,
        patch_size: int,
    ) -> list[tuple[int, int]]:
        """Generate centered + jittered patch positions.

        Args:
            cx, cy: Annotation center coordinates.
            wsi_w, wsi_h: WSI dimensions for boundary clamping.
            patch_size: Patch dimension.

        Returns:
            List of (x, y) top-left coordinates.
        """
        positions: list[tuple[int, int]] = []

        # Centered position
        x_center = cx - patch_size // 2
        y_center = cy - patch_size // 2
        x_center = max(0, min(x_center, wsi_w - patch_size))
        y_center = max(0, min(y_center, wsi_h - patch_size))
        positions.append((x_center, y_center))

        # Jittered positions
        for _ in range(self.JITTER_COUNT):
            jx = self._rng.randint(-self.MAX_JITTER, self.MAX_JITTER)
            jy = self._rng.randint(-self.MAX_JITTER, self.MAX_JITTER)

            x_jit = cx - patch_size // 2 + jx
            y_jit = cy - patch_size // 2 + jy

            x_jit = max(0, min(x_jit, wsi_w - patch_size))
            y_jit = max(0, min(y_jit, wsi_h - patch_size))

            positions.append((x_jit, y_jit))

        return positions

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load tissue mask as grayscale array."""
        img = Image.open(mask_path)
        if img.mode != "L":
            img = img.convert("L")
        return np.array(img)

    def _array_to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        elif arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        return Image.fromarray(arr)

    def _write_yolo_label(
        self,
        path: Path,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> None:
        """Write YOLO format label file."""
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        path.write_text(line)

    def _split_dataset(
        self,
        samples: list[tuple[Path, Path, str]],
    ) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, Path, str]]]:
        """Split samples into train/val sets."""
        self._rng.shuffle(samples)

        val_count = int(len(samples) * self.VAL_FRACTION)
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]

        return train_samples, val_samples

    def _organize_splits(
        self,
        train_samples: list[tuple[Path, Path, str]],
        val_samples: list[tuple[Path, Path, str]],
    ) -> None:
        """Move staged files to train/val directories."""
        import shutil

        for img_path, lbl_path, _ in train_samples:
            shutil.move(str(img_path), self._output_path / "images" / "train" / img_path.name)
            shutil.move(str(lbl_path), self._output_path / "labels" / "train" / lbl_path.name)

        for img_path, lbl_path, _ in val_samples:
            shutil.move(str(img_path), self._output_path / "images" / "val" / img_path.name)
            shutil.move(str(lbl_path), self._output_path / "labels" / "val" / lbl_path.name)

        # Clean up staging
        staging_dir = self._output_path / "_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    def _progress_bar(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[dataset]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
