"""YOLO-based MSX candidate detection on 10x tiles."""

from __future__ import annotations

import json
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
from ultralytics import YOLO

from src.bridge.manifest import ManifestManager, TileRecord
from src.core.config import Config
from src.core.paths import Paths

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single YOLO detection with Level 0 coordinates.

    Args:
        class_id: YOLO class index.
        class_name: Human-readable class label.
        confidence: Detection confidence [0, 1].
        x_level0: Bbox center X in Level 0 coords.
        y_level0: Bbox center Y in Level 0 coords.
        w_level0: Bbox width in Level 0 pixels.
        h_level0: Bbox height in Level 0 pixels.
        x_local: Bbox center X in tile-local coords.
        y_local: Bbox center Y in tile-local coords.
        w_local: Bbox width in tile-local pixels.
        h_local: Bbox height in tile-local pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    x_level0: int
    y_level0: int
    w_level0: int
    h_level0: int
    x_local: int
    y_local: int
    w_local: int
    h_local: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_level0": {
                "x": self.x_level0,
                "y": self.y_level0,
                "w": self.w_level0,
                "h": self.h_level0,
            },
            "bbox_local": {
                "x": self.x_local,
                "y": self.y_local,
                "w": self.w_local,
                "h": self.h_local,
            },
        }


class YOLODetector:
    """YOLO-based detector for MSX candidates on 10x tiles.

    Scans mid-resolution tiles for plasmodia candidates, converts
    detections to Level 0 coordinates, and persists results to manifest.
    """

    def __init__(
        self,
        manifest: ManifestManager,
        model_path: Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            manifest: ManifestManager instance for tile queries and updates.
            model_path: Path to YOLO weights. Defaults to models/yolo_msx.pt.
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: NMS IoU threshold.
        """
        self._manifest = manifest
        self._config = Config.get()
        self._paths = Paths.get()

        self._conf_thresh = confidence_threshold
        self._iou_thresh = iou_threshold

        # Resolve model path
        if model_path is None:
            model_path = self._paths.models_dir / "yolo_msx.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")

        self._model = YOLO(str(model_path))
        logger.info(f"Loaded YOLO model: {model_path}")

        # Mid-scale downsample factor for coordinate conversion
        self._downsample = self._config.get_scale("mid").downsample

    def detect_tiles(
        self,
        wsi_name: str,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Run detection on all mid-scale tiles for a WSI.

        Args:
            wsi_name: WSI filename (without path).
            client_id: Optional filter by client.

        Returns:
            Summary dict with tile_count, detection_count, and detections per tile.
        """
        tiles = self._manifest.get_tiles_by_wsi(wsi_name, client_id)
        mid_tiles = [t for t in tiles if t.scale == "mid"]

        if not mid_tiles:
            logger.warning(f"No mid-scale tiles found for {wsi_name}")
            return {"tile_count": 0, "detection_count": 0, "tiles": {}}

        logger.info(f"Running detection on {len(mid_tiles)} mid tiles for {wsi_name}")

        # Debug output directory
        debug_dir = None
        if self._config.debug_enabled:
            debug_dir = self._paths.debug_dir / "detection" / wsi_name
            debug_dir.mkdir(parents=True, exist_ok=True)

        total_detections = 0
        results_by_tile: dict[int, list[dict[str, Any]]] = {}

        with self._progress_bar() as progress:
            task = progress.add_task("Detecting", total=len(mid_tiles))

            for tile in mid_tiles:
                progress.update(task, advance=1)

                detections = self._process_tile(tile, debug_dir)

                if detections:
                    # Convert to dicts for JSON serialization
                    det_dicts = [d.to_dict() for d in detections]
                    results_by_tile[tile.id] = det_dicts
                    total_detections += len(detections)

                    # Persist to manifest
                    self._manifest.update_diagnostic(
                        tile.id,
                        yolo_box={"detections": det_dicts},
                    )

        logger.info(
            f"Detection complete: {total_detections} candidates "
            f"across {len(results_by_tile)} tiles"
        )

        return {
            "wsi_name": wsi_name,
            "tile_count": len(mid_tiles),
            "tiles_with_detections": len(results_by_tile),
            "detection_count": total_detections,
            "tiles": results_by_tile,
        }

    def _process_tile(
        self,
        tile: TileRecord,
        debug_dir: Path | None,
    ) -> list[Detection]:
        """Run inference on a single tile.

        Args:
            tile: TileRecord from manifest.
            debug_dir: Directory for debug overlays, or None.

        Returns:
            List of Detection objects with Level 0 coordinates.
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
            return []

        # Run YOLO inference
        results = self._model.predict(
            source=str(tile_path),
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return []

        detections = self._extract_detections(results[0], tile)

        # Save debug overlay
        if debug_dir and detections:
            self._save_debug_overlay(tile_path, detections, debug_dir, tile)

        return detections

    def _extract_detections(
        self,
        result: Any,
        tile: TileRecord,
    ) -> list[Detection]:
        """Extract detections from YOLO result and convert coordinates.

        Args:
            result: Single YOLO Results object.
            tile: Source tile record for coordinate offset.

        Returns:
            List of Detection objects.
        """
        detections: list[Detection] = []
        boxes = result.boxes

        for i in range(len(boxes)):
            # xywh format: center_x, center_y, width, height (local coords)
            xywh = boxes.xywh[i].cpu().numpy()
            x_local, y_local, w_local, h_local = map(int, xywh)

            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            # Convert to Level 0 coordinates
            x_l0, y_l0, w_l0, h_l0 = self._convert_to_level0(
                x_local, y_local, w_local, h_local,
                tile.x_level0, tile.y_level0,
            )

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                x_level0=x_l0,
                y_level0=y_l0,
                w_level0=w_l0,
                h_level0=h_l0,
                x_local=x_local,
                y_local=y_local,
                w_local=w_local,
                h_local=h_local,
            ))

        return detections

    def _convert_to_level0(
        self,
        x_local: int,
        y_local: int,
        w_local: int,
        h_local: int,
        tile_x_l0: int,
        tile_y_l0: int,
    ) -> tuple[int, int, int, int]:
        """Convert local tile coordinates to Level 0 global coordinates.

        The tile origin (tile_x_l0, tile_y_l0) is already in Level 0 coords.
        Local pixel positions scale by the downsample factor (4.0 for mid).

        Args:
            x_local, y_local: Detection center in tile pixels.
            w_local, h_local: Detection dimensions in tile pixels.
            tile_x_l0, tile_y_l0: Tile top-left in Level 0 coords.

        Returns:
            (x_level0, y_level0, w_level0, h_level0) in Level 0 pixels.
        """
        ds = self._downsample

        x_l0 = tile_x_l0 + int(x_local * ds)
        y_l0 = tile_y_l0 + int(y_local * ds)
        w_l0 = int(w_local * ds)
        h_l0 = int(h_local * ds)

        return x_l0, y_l0, w_l0, h_l0

    def _save_debug_overlay(
        self,
        tile_path: Path,
        detections: list[Detection],
        debug_dir: Path,
        tile: TileRecord,
    ) -> None:
        """Save tile with detection bounding boxes overlaid.

        Args:
            tile_path: Path to source tile PNG.
            detections: List of detections to draw.
            debug_dir: Output directory for debug images.
            tile: Source tile record for naming.
        """
        img = cv2.imread(str(tile_path))
        if img is None:
            logger.warning(f"Could not read tile for debug: {tile_path}")
            return

        # Box styling
        color = (0, 255, 0)  # Green
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        for det in detections:
            # Convert center coords to top-left for cv2.rectangle
            x1 = det.x_local - det.w_local // 2
            y1 = det.y_local - det.h_local // 2
            x2 = det.x_local + det.w_local // 2
            y2 = det.y_local + det.h_local // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Label with class and confidence
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            # Background for label
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                img, label, (x1, y1 - 2),
                font, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
            )

        out_path = debug_dir / f"{tile.wsi_name}_{tile.x_level0}_{tile.y_level0}_det.png"
        cv2.imwrite(str(out_path), img)
        logger.debug(f"Saved debug overlay: {out_path}")

    def _progress_bar(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[detection]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
