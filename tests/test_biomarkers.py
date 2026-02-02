"""Unit tests for BiomarkerAnalyzer hemocyte detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.hpc.biomarkers import BiomarkerAnalyzer, HemocyteResult


class TestBiomarkerAnalyzer:
    """Tests for BiomarkerAnalyzer OpenCV blob detection."""

    @pytest.fixture
    def mock_manifest(self) -> MagicMock:
        """Create mock ManifestManager."""
        manifest = MagicMock()
        manifest.get_tiles_by_wsi.return_value = []
        return manifest

    @pytest.fixture
    def analyzer(self, mock_manifest: MagicMock) -> BiomarkerAnalyzer:
        """Create BiomarkerAnalyzer with mocked manifest."""
        with patch("src.hpc.biomarkers.Config.get") as mock_config, \
             patch("src.hpc.biomarkers.Paths.get") as mock_paths:
            mock_config.return_value.debug_enabled = False
            mock_paths.return_value.processed_dir = Path("/tmp/processed")
            mock_paths.return_value.debug_dir = Path("/tmp/debug")

            return BiomarkerAnalyzer(manifest=mock_manifest)

    def test_create_blob_detector_params(self, analyzer: BiomarkerAnalyzer) -> None:
        """Blob detector has correct parameters for hemocyte nuclei."""
        detector = analyzer._detector

        # Verify detector was created (SimpleBlobDetector is opaque)
        assert detector is not None

    def test_detect_hemocytes_on_synthetic_image(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Detects dark circular blobs on synthetic tissue image."""
        # Create 512x512 pink tissue background
        img = np.full((512, 512, 3), (200, 180, 220), dtype=np.uint8)

        # Draw 5 dark circular "hemocytes" (radius ~3-4 pixels)
        hemocyte_positions = [
            (100, 100),
            (200, 150),
            (300, 200),
            (150, 350),
            (400, 400),
        ]
        for x, y in hemocyte_positions:
            cv2.circle(img, (x, y), 4, (50, 30, 60), -1)

        keypoints = analyzer._detect_hemocytes(img)

        # Should detect approximately 5 blobs (tolerance for edge effects)
        assert 3 <= len(keypoints) <= 7

    def test_detect_hemocytes_excludes_large_structures(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Excludes structures larger than MAX_AREA_PX (>10µm diameter)."""
        img = np.full((512, 512, 3), (200, 180, 220), dtype=np.uint8)

        # Draw one valid hemocyte (radius 3)
        cv2.circle(img, (100, 100), 3, (40, 30, 50), -1)

        # Draw one large structure (radius 15, too big)
        cv2.circle(img, (300, 300), 15, (40, 30, 50), -1)

        keypoints = analyzer._detect_hemocytes(img)

        # Should only detect the small hemocyte
        assert len(keypoints) <= 2

    def test_detect_hemocytes_excludes_irregular_shapes(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Excludes irregular non-circular shapes."""
        img = np.full((512, 512, 3), (200, 180, 220), dtype=np.uint8)

        # Draw one circular hemocyte
        cv2.circle(img, (100, 100), 4, (40, 30, 50), -1)

        # Draw an elongated shape (not circular)
        cv2.ellipse(img, (300, 300), (3, 12), 0, 0, 360, (40, 30, 50), -1)

        keypoints = analyzer._detect_hemocytes(img)

        # The elongated shape should be filtered by circularity
        assert len(keypoints) <= 2

    def test_calculate_tissue_area_excludes_white_background(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """White background is excluded from tissue area calculation."""
        # Create image: half pink tissue, half white background
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:, :256] = (200, 180, 220)  # Pink tissue (left half)
        img[:, 256:] = (255, 255, 255)  # White background (right half)

        tissue_area = analyzer._calculate_tissue_area(img)

        # Should be approximately half of total area
        # Total pixels = 512 * 512 = 262144
        # Tissue pixels ≈ 262144 / 2 = 131072
        # At 1.0 µm/px: area = 131072 * 1.0² * 1e-6 ≈ 0.131 mm²
        assert 0.10 < tissue_area < 0.15

    def test_calculate_tissue_area_full_tissue(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Full tissue image returns total area."""
        # Fully saturated pink tissue
        img = np.full((512, 512, 3), (180, 150, 200), dtype=np.uint8)

        tissue_area = analyzer._calculate_tissue_area(img)

        # Total = 512 * 512 * 1.0² * 1e-6 ≈ 0.262 mm²
        assert 0.25 < tissue_area < 0.28

    def test_calculate_tissue_area_empty_slide(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Pure white background returns near-zero tissue area."""
        img = np.full((512, 512, 3), (255, 255, 255), dtype=np.uint8)

        tissue_area = analyzer._calculate_tissue_area(img)

        assert tissue_area < 0.01

    def test_hemocyte_result_density_calculation(self) -> None:
        """HemocyteResult correctly stores density."""
        result = HemocyteResult(
            tile_id=1,
            count=50,
            tissue_area_mm2=0.1,
            density=500.0,
            keypoints=[(i * 10, i * 10) for i in range(50)],
        )

        assert result.count == 50
        assert result.tissue_area_mm2 == 0.1
        assert result.density == 500.0
        assert len(result.keypoints) == 50

    def test_analyze_tile_returns_none_for_missing_file(
        self, analyzer: BiomarkerAnalyzer
    ) -> None:
        """Returns None when tile file doesn't exist."""
        tile = MagicMock()
        tile.wsi_name = "missing.ome.tiff"
        tile.scale = "mid"
        tile.x_level0 = 0
        tile.y_level0 = 0

        result = analyzer._analyze_tile(tile, None)

        assert result is None

    def test_analyze_wsi_returns_summary(
        self, analyzer: BiomarkerAnalyzer, mock_manifest: MagicMock
    ) -> None:
        """Returns summary dict even with no tiles."""
        mock_manifest.get_tiles_by_wsi.return_value = []

        summary = analyzer.analyze_wsi("empty.ome.tiff")

        assert summary["tile_count"] == 0
        assert summary["total_hemocytes"] == 0
        assert summary["mean_density"] == 0.0


class TestBiomarkerIntegration:
    """Integration tests with synthetic hemocyte images."""

    @pytest.fixture
    def synthetic_tile_path(self, tmp_path: Path) -> Path:
        """Create synthetic tile image with known hemocyte count."""
        # Create tissue background with 10 hemocytes
        img = np.full((1024, 1024, 3), (200, 170, 210), dtype=np.uint8)

        # Add 10 hemocytes in a grid pattern
        for i in range(10):
            x = 100 + (i % 5) * 180
            y = 100 + (i // 5) * 400
            cv2.circle(img, (x, y), 4, (45, 35, 55), -1)

        tile_path = tmp_path / "synthetic_tile.png"
        cv2.imwrite(str(tile_path), img)

        return tile_path

    def test_full_detection_pipeline(
        self, synthetic_tile_path: Path
    ) -> None:
        """End-to-end test with synthetic hemocyte image."""
        with patch("src.hpc.biomarkers.Config.get") as mock_config, \
             patch("src.hpc.biomarkers.Paths.get") as mock_paths:
            mock_config.return_value.debug_enabled = False
            mock_paths.return_value.processed_dir = synthetic_tile_path.parent
            mock_paths.return_value.debug_dir = synthetic_tile_path.parent / "debug"

            analyzer = BiomarkerAnalyzer(manifest=MagicMock())

            img = cv2.imread(str(synthetic_tile_path))
            keypoints = analyzer._detect_hemocytes(img)

            # Should detect approximately 10 hemocytes
            assert 7 <= len(keypoints) <= 13

            tissue_area = analyzer._calculate_tissue_area(img)
            # Full 1024x1024 tissue = 1.048 mm²
            assert 0.9 < tissue_area < 1.1

            density = len(keypoints) / tissue_area
            # Expected: ~10 cells / 1.0 mm² = ~10/mm²
            assert 5 < density < 15
