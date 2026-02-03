"""Tile extraction engine for OME-TIFF whole slide images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile
import zarr
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.bridge.manifest import ManifestManager
from src.core.config import Config, ScaleConfig
from src.core.paths import Paths

logger = logging.getLogger(__name__)

# Suppress PIL decompression bomb warning for large masks
Image.MAX_IMAGE_PIXELS = None


class TilingEngine:
    """Multi-scale tile extractor for OME-TIFF WSIs.

    Extracts tiles at macro (1.25x), mid (10x), and high (40x) scales
    with 15% overlap. Filters tiles using a pre-computed tissue mask.
    All coordinates are stored normalized to Level 0 (40x).

    Uses zarr-backed lazy loading to avoid loading entire pyramid levels
    into memory, enabling processing of 100GB+ WSIs on limited RAM.
    """

    OVERLAP_FRACTION: float = 0.15

    def __init__(
        self,
        manifest: ManifestManager,
        client_id: str = "default",
    ) -> None:
        """Initialize the tiling engine.

        Args:
            manifest: ManifestManager instance for tile registration.
            client_id: Multi-tenant client identifier.
        """
        self._manifest = manifest
        self._client_id = client_id
        self._config = Config.get()
        self._paths = Paths.get()

    def process_wsi(
        self,
        wsi_path: Path,
        mask_path: Path,
        scales: list[str] | None = None,
        tissue_threshold: float = 0.1,
    ) -> dict[str, int]:
        """Extract tiles from a WSI at all configured scales.

        Args:
            wsi_path: Path to the OME-TIFF file.
            mask_path: Path to the tissue mask PNG.
            scales: Scale names to process. Defaults to all ("macro", "mid", "high").
            tissue_threshold: Minimum tissue fraction to extract tile.

        Returns:
            Dict mapping scale name to number of tiles extracted.
        """
        if scales is None:
            scales = ["macro", "mid", "high"]

        wsi_name = wsi_path.stem
        mask_array = self._load_mask(mask_path)

        logger.info(f"Processing {wsi_name} at scales: {scales}")

        tile_counts: dict[str, int] = {}

        with tifffile.TiffFile(wsi_path) as tif:
            # Validate OME-TIFF structure
            if not tif.pages:
                raise ValueError(f"No pages found in {wsi_path}")

            level0_page = tif.pages[0]
            wsi_width = level0_page.shape[1]
            wsi_height = level0_page.shape[0]

            logger.info(f"WSI dimensions (Level 0): {wsi_width} x {wsi_height}")

            # Auto-detect mask scaling factor
            mask_scale = self._compute_mask_scale(
                wsi_dims=(wsi_width, wsi_height),
                mask_dims=(mask_array.shape[1], mask_array.shape[0]),
            )
            logger.debug(f"Mask scaling factor: {mask_scale:.1f}x")

            # Build pyramid level map: level_index -> (page_index, downsample)
            pyramid_map = self._build_pyramid_map(tif, wsi_width, wsi_height)

            # Create zarr store for memory-efficient region access
            store = tif.aszarr()

            for scale_name in scales:
                count = self._extract_scale(
                    store=store,
                    tif=tif,
                    wsi_name=wsi_name,
                    scale_name=scale_name,
                    mask_array=mask_array,
                    mask_scale=mask_scale,
                    wsi_dims=(wsi_width, wsi_height),
                    pyramid_map=pyramid_map,
                    tissue_threshold=tissue_threshold,
                )
                tile_counts[scale_name] = count

            store.close()

        return tile_counts

    def _compute_mask_scale(
        self,
        wsi_dims: tuple[int, int],
        mask_dims: tuple[int, int],
    ) -> float:
        """Compute the scaling factor between WSI Level 0 and mask.

        Args:
            wsi_dims: (width, height) of WSI at Level 0.
            mask_dims: (width, height) of mask image.

        Returns:
            Average scaling factor (WSI pixels per mask pixel).
        """
        scale_x = wsi_dims[0] / mask_dims[0]
        scale_y = wsi_dims[1] / mask_dims[1]

        # Warn if scales differ significantly (anisotropic mask)
        if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.1:
            logger.warning(
                f"Anisotropic mask scaling: X={scale_x:.1f}, Y={scale_y:.1f}. "
                "Using average."
            )

        return (scale_x + scale_y) / 2

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load tissue mask as single-channel array.

        Converts RGB to grayscale if necessary. Non-zero values indicate tissue.
        """
        img = Image.open(mask_path)

        # Force single channel
        if img.mode != "L":
            img = img.convert("L")

        mask = np.array(img)
        logger.debug(f"Loaded mask {mask_path.name}: {mask.shape}, dtype={mask.dtype}")
        return mask

    def _build_pyramid_map(
        self,
        tif: tifffile.TiffFile,
        base_width: int,
        base_height: int,
    ) -> dict[int, tuple[int, float]]:
        """Map pyramid levels to page indices and downsample factors.

        Returns:
            Dict mapping level_index -> (page_index, actual_downsample).
        """
        pyramid_map: dict[int, tuple[int, float]] = {}

        for page_idx, page in enumerate(tif.pages):
            page_width = page.shape[1]
            downsample = base_width / page_width

            # Find closest integer level
            level = int(np.log2(downsample)) if downsample > 1 else 0
            pyramid_map[level] = (page_idx, downsample)

        logger.debug(f"Pyramid map: {pyramid_map}")
        return pyramid_map

    def _extract_scale(
        self,
        store: zarr.storage.FSStore,
        tif: tifffile.TiffFile,
        wsi_name: str,
        scale_name: str,
        mask_array: np.ndarray,
        mask_scale: float,
        wsi_dims: tuple[int, int],
        pyramid_map: dict[int, tuple[int, float]],
        tissue_threshold: float,
    ) -> int:
        """Extract all tiles for a single scale.

        Returns:
            Number of tiles extracted.
        """
        scale_config = self._config.get_scale(scale_name)
        wsi_width, wsi_height = wsi_dims

        # Output directory
        out_dir = self._paths.processed_dir / wsi_name / scale_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Calculate stride with overlap
        tile_size = scale_config.tile_size
        stride = int(tile_size * (1 - self.OVERLAP_FRACTION))

        # Tile dimensions in Level 0 coordinates
        tile_size_l0 = int(tile_size * scale_config.downsample)
        stride_l0 = int(stride * scale_config.downsample)

        # Generate tile positions
        positions = list(self._generate_positions(
            wsi_width, wsi_height, tile_size_l0, stride_l0,
        ))

        # Check for native pyramid level
        native_level = self._find_native_level(scale_config, pyramid_map)

        # Open zarr array for the appropriate pyramid level
        zarr_array = zarr.open(store, mode="r")

        tile_count = 0

        with self._progress_bar(f"[{scale_name}]") as progress:
            task = progress.add_task(f"Extracting {scale_name}", total=len(positions))

            for x0, y0 in positions:
                progress.update(task, advance=1)

                # Skip if insufficient tissue
                if not self._is_tissue(
                    mask_array, x0, y0, tile_size_l0, tile_size_l0,
                    mask_scale, tissue_threshold,
                ):
                    continue

                # Extract tile using zarr (memory-efficient)
                tile = self._extract_tile_zarr(
                    zarr_array=zarr_array,
                    tif=tif,
                    x0=x0,
                    y0=y0,
                    tile_size=tile_size,
                    scale_config=scale_config,
                    native_level=native_level,
                    pyramid_map=pyramid_map,
                )

                if tile is None:
                    continue

                # Save tile
                tile_path = out_dir / f"{wsi_name}_{scale_name}_{x0}_{y0}.png"
                tile.save(tile_path, "PNG")

                # Register in manifest
                self._manifest.register_tile(
                    client_id=self._client_id,
                    wsi_name=wsi_name,
                    scale=scale_name,
                    x_level0=x0,
                    y_level0=y0,
                    width=tile_size,
                    height=tile_size,
                )

                tile_count += 1

        logger.info(f"[{scale_name}] Extracted {tile_count} tiles to {out_dir}")
        return tile_count

    def _generate_positions(
        self,
        wsi_width: int,
        wsi_height: int,
        tile_size_l0: int,
        stride_l0: int,
    ) -> Iterator[tuple[int, int]]:
        """Generate tile positions in Level 0 coordinates."""
        for y0 in range(0, wsi_height - tile_size_l0 + 1, stride_l0):
            for x0 in range(0, wsi_width - tile_size_l0 + 1, stride_l0):
                yield x0, y0

    def _find_native_level(
        self,
        scale_config: ScaleConfig,
        pyramid_map: dict[int, tuple[int, float]],
    ) -> int | None:
        """Find native pyramid level matching the scale config.

        Returns:
            Pyramid level index if available, None if downsampling required.
        """
        target_level = scale_config.level

        if target_level in pyramid_map:
            _, actual_downsample = pyramid_map[target_level]
            # Accept if within 5% of expected downsample
            if abs(actual_downsample - scale_config.downsample) / scale_config.downsample < 0.05:
                return target_level

        return None

    def _extract_tile_zarr(
        self,
        zarr_array: zarr.Array | zarr.Group,
        tif: tifffile.TiffFile,
        x0: int,
        y0: int,
        tile_size: int,
        scale_config: ScaleConfig,
        native_level: int | None,
        pyramid_map: dict[int, tuple[int, float]],
    ) -> Image.Image | None:
        """Extract a single tile using zarr for memory-efficient access.

        Args:
            zarr_array: Zarr array opened from tifffile store.
            tif: Open TiffFile handle (for metadata).
            x0, y0: Top-left corner in Level 0 coordinates.
            tile_size: Output tile dimension in pixels.
            scale_config: Scale configuration.
            native_level: Native pyramid level or None.
            pyramid_map: Pyramid level mapping.

        Returns:
            PIL Image or None if extraction failed.
        """
        try:
            # Determine if zarr_array is a Group (multi-level) or Array (single level)
            is_group = hasattr(zarr_array, "keys") and callable(zarr_array.keys)

            if native_level is not None and scale_config.downsample > 1:
                # Extract from native pyramid level
                page_idx, actual_downsample = pyramid_map[native_level]
                x_scaled = int(x0 / actual_downsample)
                y_scaled = int(y0 / actual_downsample)

                # Zarr lazy slice - only loads requested region
                if is_group:
                    # Multi-level pyramid stored as group (zarr v3 uses string keys)
                    level_array = zarr_array[str(page_idx)]
                    region = np.array(level_array[
                        y_scaled:y_scaled + tile_size,
                        x_scaled:x_scaled + tile_size,
                    ])
                else:
                    # Single array or different structure
                    region = np.array(zarr_array[
                        y_scaled:y_scaled + tile_size,
                        x_scaled:x_scaled + tile_size,
                    ])
            else:
                # Extract from Level 0 and downsample
                read_size = int(tile_size * scale_config.downsample)

                # Zarr lazy slice from Level 0
                if is_group:
                    level0 = zarr_array["0"]
                    region = np.array(level0[
                        y0:y0 + read_size,
                        x0:x0 + read_size,
                    ])
                else:
                    region = np.array(zarr_array[
                        y0:y0 + read_size,
                        x0:x0 + read_size,
                    ])

            # Handle empty or invalid regions
            if region.size == 0:
                return None

            # Convert to PIL
            if region.ndim == 2:
                img = Image.fromarray(region, mode="L")
            elif region.ndim == 3 and region.shape[2] == 3:
                img = Image.fromarray(region, mode="RGB")
            elif region.ndim == 3 and region.shape[2] == 4:
                img = Image.fromarray(region, mode="RGBA")
            else:
                img = Image.fromarray(region)

            # Downsample if extracted from Level 0
            if native_level is None and scale_config.downsample > 1:
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            return img

        except Exception as e:
            logger.warning(f"Failed to extract tile at ({x0}, {y0}): {e}")
            return None

    def _is_tissue(
        self,
        mask_array: np.ndarray,
        x0: int,
        y0: int,
        w0: int,
        h0: int,
        mask_scale: float,
        threshold: float,
    ) -> bool:
        """Check if tile region contains sufficient tissue.

        Args:
            mask_array: Single-channel mask at reduced resolution.
            x0, y0: Top-left corner in Level 0 coordinates.
            w0, h0: Tile dimensions in Level 0 pixels.
            mask_scale: WSI-to-mask scaling factor (auto-detected).
            threshold: Minimum tissue fraction.

        Returns:
            True if tissue coverage >= threshold.
        """
        # Map Level 0 coords to mask coords using dynamic scale
        x_m = int(x0 / mask_scale)
        y_m = int(y0 / mask_scale)
        w_m = max(1, int(w0 / mask_scale))
        h_m = max(1, int(h0 / mask_scale))

        # Clamp to mask bounds
        mask_h, mask_w = mask_array.shape[:2]
        x_end = min(x_m + w_m, mask_w)
        y_end = min(y_m + h_m, mask_h)

        if x_m >= mask_w or y_m >= mask_h:
            return False

        region = mask_array[y_m:y_end, x_m:x_end]

        if region.size == 0:
            return False

        # Count non-zero pixels (tissue) - mask is single-channel
        tissue_fraction = np.count_nonzero(region) / region.size
        return tissue_fraction >= threshold

    def _progress_bar(self, description: str) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn(description),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
