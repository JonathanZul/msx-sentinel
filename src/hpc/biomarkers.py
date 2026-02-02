"""Hemocyte density biomarker analysis on 10x tiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.bridge.manifest import ManifestManager, TileRecord
from src.core.config import Config
from src.core.paths import Paths

logger = logging.getLogger(__name__)


@dataclass
class HemocyteResult:
    """Hemocyte analysis result for a single tile.

    Args:
        tile_id: Manifest tile ID.
        count: Number of detected hemocytes.
        tissue_area_mm2: Tissue area in mm².
        density: Hemocytes per mm².
        keypoints: List of (x, y) coordinates for detected hemocytes.
    """
    tile_id: int
    count: int
    tissue_area_mm2: float
    density: float
    keypoints: list[tuple[int, int]]


class BiomarkerAnalyzer:
    """Hemocyte density analyzer for 10x mid-resolution tiles.

    Uses OpenCV blob detection to count hemocyte nuclei and
    calculates density as count / tissue_area_mm².
    """

    # Microns per pixel at 10x (typical for Aperio/Leica scanners)
    UM_PER_PIXEL_10X = 1.0

    # Blob detector parameters tuned for hemocyte nuclei at 10x
    # Hemocytes: 3-7µm diameter → 3-7 pixels at 1µm/px
    MIN_AREA_PX = 7      # π × 1.5² ≈ 7 (min 3µm diameter)
    MAX_AREA_PX = 80     # π × 5² ≈ 78 (exclude >10µm structures)
    MIN_CIRCULARITY = 0.6
    MIN_CONVEXITY = 0.8

    # Tissue detection: saturation threshold in HSV
    TISSUE_SAT_THRESHOLD = 10

    def __init__(self, manifest: ManifestManager | None = None) -> None:
        """Initialize the biomarker analyzer.

        Args:
            manifest: ManifestManager instance. Creates default if None.
        """
        self._manifest = manifest or ManifestManager()
        self._config = Config.get()
        self._paths = Paths.get()
        self._detector = self._create_blob_detector()

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
        """Configure SimpleBlobDetector for hemocyte nuclei."""
        params = cv2.SimpleBlobDetector_Params()

        # Filter by color (detect dark blobs = nuclei)
        params.filterByColor = True
        params.blobColor = 0  # Dark objects

        # Filter by area
        params.filterByArea = True
        params.minArea = self.MIN_AREA_PX
        params.maxArea = self.MAX_AREA_PX

        # Filter by circularity (hemocytes are roughly spherical)
        params.filterByCircularity = True
        params.minCircularity = self.MIN_CIRCULARITY

        # Filter by convexity (reject irregular debris)
        params.filterByConvexity = True
        params.minConvexity = self.MIN_CONVEXITY

        # Disable inertia filter
        params.filterByInertia = False

        return cv2.SimpleBlobDetector_create(params)

    def analyze_wsi(self, wsi_name: str, client_id: str | None = None) -> dict[str, Any]:
        """Run hemocyte density analysis on all mid tiles for a WSI.

        Args:
            wsi_name: WSI filename to process.
            client_id: Optional filter by client.

        Returns:
            Summary dict with tile_count, total_hemocytes, mean_density.
        """
        tiles = self._manifest.get_tiles_by_wsi(wsi_name, client_id)
        mid_tiles = [t for t in tiles if t.scale == "mid"]

        if not mid_tiles:
            logger.warning(f"No mid-scale tiles found for {wsi_name}")
            return {"tile_count": 0, "total_hemocytes": 0, "mean_density": 0.0}

        logger.info(f"Analyzing hemocyte density on {len(mid_tiles)} mid tiles for {wsi_name}")

        # Debug output directory
        debug_dir = None
        if self._config.debug_enabled:
            debug_dir = self._paths.debug_dir / "biomarkers" / wsi_name
            debug_dir.mkdir(parents=True, exist_ok=True)

        results: list[HemocyteResult] = []
        total_count = 0
        total_area = 0.0

        with self._progress_bar() as progress:
            task = progress.add_task("Hemocytes", total=len(mid_tiles))

            for tile in mid_tiles:
                progress.update(task, advance=1)

                result = self._analyze_tile(tile, debug_dir)
                if result is None:
                    continue

                results.append(result)
                total_count += result.count
                total_area += result.tissue_area_mm2

                # Persist to manifest
                self._manifest.update_diagnostic(
                    tile.id,
                    hemocyte_density=result.density,
                )

        mean_density = total_count / total_area if total_area > 0 else 0.0

        logger.info(
            f"Hemocyte analysis complete: {total_count} cells across "
            f"{len(results)} tiles, mean density {mean_density:.1f}/mm²"
        )

        return {
            "wsi_name": wsi_name,
            "tile_count": len(results),
            "total_hemocytes": total_count,
            "total_tissue_area_mm2": round(total_area, 4),
            "mean_density": round(mean_density, 2),
        }

    def _analyze_tile(
        self,
        tile: TileRecord,
        debug_dir: Path | None,
    ) -> HemocyteResult | None:
        """Analyze a single tile for hemocyte density.

        Args:
            tile: TileRecord from manifest.
            debug_dir: Debug output directory, or None.

        Returns:
            HemocyteResult or None if tile cannot be processed.
        """
        # Resolve tile path
        tile_path = (
            self._paths.processed_dir
            / tile.wsi_name
            / tile.scale
            / f"{tile.wsi_name}_{tile.scale}_{tile.x_level0}_{tile.y_level0}.png"
        )

        if not tile_path.exists():
            logger.warning(f"Tile not found: {tile_path}")
            return None

        # Load tile
        img = cv2.imread(str(tile_path))
        if img is None:
            logger.warning(f"Failed to read tile: {tile_path}")
            return None

        # Detect hemocytes
        keypoints = self._detect_hemocytes(img)
        count = len(keypoints)

        # Calculate tissue area
        tissue_area_mm2 = self._calculate_tissue_area(img)

        # Compute density
        density = count / tissue_area_mm2 if tissue_area_mm2 > 0 else 0.0

        result = HemocyteResult(
            tile_id=tile.id,
            count=count,
            tissue_area_mm2=tissue_area_mm2,
            density=density,
            keypoints=keypoints,
        )

        # Save debug overlay
        if debug_dir:
            self._save_debug_overlay(img, result, tile, debug_dir)

        return result

    def _detect_hemocytes(self, img: np.ndarray) -> list[tuple[int, int]]:
        """Detect hemocyte nuclei using blob detection.

        Args:
            img: BGR image array.

        Returns:
            List of (x, y) keypoint coordinates.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold to isolate dark nuclei
        # Invert so nuclei become white (blobs are detected as white regions)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10,
        )

        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Detect blobs on inverted image (detector expects dark blobs on light background)
        inv_thresh = cv2.bitwise_not(thresh)
        keypoints = self._detector.detect(inv_thresh)

        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def _calculate_tissue_area(self, img: np.ndarray) -> float:
        """Calculate tissue area in mm² excluding white space.

        Args:
            img: BGR image array.

        Returns:
            Tissue area in mm².
        """
        # Convert to HSV and threshold on saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Tissue has saturation > threshold (white background has low saturation)
        tissue_mask = saturation > self.TISSUE_SAT_THRESHOLD
        tissue_pixels = np.count_nonzero(tissue_mask)

        # Convert pixels to mm²
        # Area = pixels × (µm/pixel)² × 10⁻⁶
        pixel_area_um2 = self.UM_PER_PIXEL_10X ** 2
        tissue_area_mm2 = tissue_pixels * pixel_area_um2 * 1e-6

        return tissue_area_mm2

    def _save_debug_overlay(
        self,
        img: np.ndarray,
        result: HemocyteResult,
        tile: TileRecord,
        debug_dir: Path,
    ) -> None:
        """Save tile with hemocyte detections overlaid as green dots.

        Args:
            img: Original BGR image.
            result: Hemocyte analysis result.
            tile: Source tile record.
            debug_dir: Output directory.
        """
        overlay = img.copy()

        # Draw green dots at each keypoint
        for x, y in result.keypoints:
            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

        # Add density label
        label = f"D={result.density:.1f}/mm2 N={result.count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # White background for text
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(overlay, (5, 5), (10 + tw, 15 + th), (255, 255, 255), -1)
        cv2.putText(overlay, label, (8, 12 + th), font, font_scale, (0, 100, 0), thickness)

        # Save
        out_path = debug_dir / f"{tile.wsi_name}_{tile.x_level0}_{tile.y_level0}_hemo.png"
        cv2.imwrite(str(out_path), overlay)
        logger.debug(f"Saved hemocyte overlay: {out_path}")

    def _progress_bar(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[biomarker]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
