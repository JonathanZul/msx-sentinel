"""Unit tests for VLMEye and DiagnosticBrain reasoning modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.local.reasoning import VLMEye, Candidate, VLM_SYSTEM_PROMPT, DEFAULT_VLM_RESPONSE
from src.local.brain import DiagnosticBrain, CaseEvidence, DiagnosticReport


class TestVLMEye:
    """Tests for VLMEye morphological verification."""

    @pytest.fixture
    def mock_manifest(self) -> MagicMock:
        """Create mock ManifestManager."""
        manifest = MagicMock()
        manifest.get_tiles_by_wsi.return_value = []
        return manifest

    @pytest.fixture
    def vlm_eye(self, mock_manifest: MagicMock) -> VLMEye:
        """Create VLMEye with mocked dependencies."""
        with patch("src.local.reasoning.providers.get_vlm") as mock_get_vlm:
            mock_vlm = MagicMock()
            mock_get_vlm.return_value = mock_vlm
            eye = VLMEye(manifest=mock_manifest)
            eye._vlm = mock_vlm
            return eye

    def test_get_unverified_candidates_empty(
        self, vlm_eye: VLMEye, mock_manifest: MagicMock
    ) -> None:
        """Returns empty list when no tiles have YOLO boxes."""
        mock_manifest.get_tiles_by_wsi.return_value = []

        candidates = vlm_eye.get_unverified_candidates("test.ome.tiff")

        assert candidates == []
        mock_manifest.get_tiles_by_wsi.assert_called_once_with("test.ome.tiff")

    def test_get_unverified_candidates_filters_verified(
        self, vlm_eye: VLMEye, mock_manifest: MagicMock
    ) -> None:
        """Excludes tiles that already have vlm_description."""
        tile_verified = MagicMock()
        tile_verified.yolo_box = {"detections": [{"bbox_level0": {"x": 100, "y": 200}}]}
        tile_verified.vlm_description = '{"organism_present": false}'

        tile_unverified = MagicMock()
        tile_unverified.id = 42
        tile_unverified.wsi_name = "test.ome.tiff"
        tile_unverified.organ_type = "Gills"
        tile_unverified.yolo_box = {
            "detections": [
                {
                    "bbox_level0": {"x": 500, "y": 600, "w": 64, "h": 64},
                    "confidence": 0.85,
                }
            ]
        }
        tile_unverified.vlm_description = None

        mock_manifest.get_tiles_by_wsi.return_value = [tile_verified, tile_unverified]

        candidates = vlm_eye.get_unverified_candidates("test.ome.tiff")

        assert len(candidates) == 1
        assert candidates[0].tile_id == 42
        assert candidates[0].x_level0 == 500
        assert candidates[0].confidence == 0.85
        assert candidates[0].organ_type == "Gills"

    def test_parse_vlm_response_valid_json(self, vlm_eye: VLMEye) -> None:
        """Parses valid JSON response correctly."""
        response = json.dumps({
            "organism_present": True,
            "multinucleated": True,
            "nucleus_count": 5,
            "cytoplasm_texture": "granular",
            "parasite_diameter_um": 25.5,
            "confidence": 0.92,
        })

        result = vlm_eye._parse_vlm_response(response)

        assert result["organism_present"] is True
        assert result["multinucleated"] is True
        assert result["nucleus_count"] == 5
        assert result["cytoplasm_texture"] == "granular"
        assert result["parasite_diameter_um"] == 25.5
        assert result["confidence"] == 0.92

    def test_parse_vlm_response_markdown_wrapped(self, vlm_eye: VLMEye) -> None:
        """Handles JSON wrapped in markdown code blocks."""
        response = """```json
{
    "organism_present": true,
    "multinucleated": false,
    "nucleus_count": 1,
    "cytoplasm_texture": "smooth",
    "parasite_diameter_um": 8.0,
    "confidence": 0.65
}
```"""

        result = vlm_eye._parse_vlm_response(response)

        assert result["organism_present"] is True
        assert result["multinucleated"] is False

    def test_parse_vlm_response_invalid_json_returns_defaults(
        self, vlm_eye: VLMEye
    ) -> None:
        """Returns default response when JSON parsing fails."""
        response = "This is not valid JSON at all."

        result = vlm_eye._parse_vlm_response(response)

        assert result == DEFAULT_VLM_RESPONSE

    def test_parse_vlm_response_clamps_confidence(self, vlm_eye: VLMEye) -> None:
        """Clamps confidence to [0.0, 1.0] range."""
        response = json.dumps({
            "organism_present": True,
            "confidence": 1.5,
        })

        result = vlm_eye._parse_vlm_response(response)

        assert result["confidence"] == 1.0

    def test_parse_vlm_response_invalid_texture_uses_default(
        self, vlm_eye: VLMEye
    ) -> None:
        """Ignores invalid cytoplasm_texture values."""
        response = json.dumps({
            "organism_present": True,
            "cytoplasm_texture": "invalid_value",
        })

        result = vlm_eye._parse_vlm_response(response)

        assert result["cytoplasm_texture"] == "artifact"

    def test_build_prompt_includes_system_prompt(self, vlm_eye: VLMEye) -> None:
        """Prompt includes the system instructions."""
        prompt = vlm_eye._build_prompt()

        assert "veterinary histopathologist" in prompt
        assert "Haplosporidium nelsoni" in prompt
        assert "JSON" in prompt


class TestDiagnosticBrain:
    """Tests for DiagnosticBrain LLM synthesis."""

    @pytest.fixture
    def mock_manifest(self) -> MagicMock:
        """Create mock ManifestManager."""
        manifest = MagicMock()
        manifest.get_tiles_by_wsi.return_value = []
        return manifest

    @pytest.fixture
    def brain(self, mock_manifest: MagicMock) -> DiagnosticBrain:
        """Create DiagnosticBrain with mocked dependencies."""
        with patch("src.local.brain.providers.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            b = DiagnosticBrain(manifest=mock_manifest)
            b._llm = mock_llm
            return b

    def test_aggregate_evidence_empty(
        self, brain: DiagnosticBrain, mock_manifest: MagicMock
    ) -> None:
        """Handles WSI with no tiles."""
        mock_manifest.get_tiles_by_wsi.return_value = []

        evidence = brain._aggregate_evidence("empty.ome.tiff", None)

        assert evidence.wsi_name == "empty.ome.tiff"
        assert evidence.yolo_count == 0
        assert evidence.vlm_confirmed == 0
        assert evidence.mean_density == 0.0

    def test_aggregate_evidence_counts_yolo_detections(
        self, brain: DiagnosticBrain, mock_manifest: MagicMock
    ) -> None:
        """Counts YOLO detections across tiles."""
        tile1 = MagicMock()
        tile1.yolo_box = {"detections": [{"conf": 0.9}, {"conf": 0.8}]}
        tile1.organ_type = "Gills"
        tile1.vlm_description = None
        tile1.hemocyte_density = None

        tile2 = MagicMock()
        tile2.yolo_box = {"detections": [{"conf": 0.7}]}
        tile2.organ_type = "Mantle"
        tile2.vlm_description = None
        tile2.hemocyte_density = None

        mock_manifest.get_tiles_by_wsi.return_value = [tile1, tile2]

        evidence = brain._aggregate_evidence("test.ome.tiff", None)

        assert evidence.yolo_count == 3
        assert evidence.tiles_with_detections == 2
        assert evidence.organ_counts == {"Gills": 2, "Mantle": 1}

    def test_aggregate_evidence_counts_vlm_confirmed(
        self, brain: DiagnosticBrain, mock_manifest: MagicMock
    ) -> None:
        """Counts VLM-confirmed organisms."""
        tile_positive = MagicMock()
        tile_positive.yolo_box = None
        tile_positive.vlm_description = json.dumps({
            "organism_present": True,
            "multinucleated": True,
            "confidence": 0.9,
        })
        tile_positive.hemocyte_density = None
        tile_positive.organ_type = None

        tile_negative = MagicMock()
        tile_negative.yolo_box = None
        tile_negative.vlm_description = json.dumps({
            "organism_present": False,
            "confidence": 0.8,
        })
        tile_negative.hemocyte_density = None
        tile_negative.organ_type = None

        mock_manifest.get_tiles_by_wsi.return_value = [tile_positive, tile_negative]

        evidence = brain._aggregate_evidence("test.ome.tiff", None)

        assert evidence.vlm_analyzed == 2
        assert evidence.vlm_confirmed == 1
        assert evidence.multinucleated_count == 1
        assert evidence.vlm_mean_confidence == pytest.approx(0.85)

    def test_aggregate_evidence_hemocyte_stats(
        self, brain: DiagnosticBrain, mock_manifest: MagicMock
    ) -> None:
        """Calculates hemocyte density statistics."""
        tile1 = MagicMock()
        tile1.yolo_box = None
        tile1.vlm_description = None
        tile1.hemocyte_density = 300.0
        tile1.organ_type = None

        tile2 = MagicMock()
        tile2.yolo_box = None
        tile2.vlm_description = None
        tile2.hemocyte_density = 700.0
        tile2.organ_type = None

        mock_manifest.get_tiles_by_wsi.return_value = [tile1, tile2]

        evidence = brain._aggregate_evidence("test.ome.tiff", None)

        assert evidence.mean_density == 500.0
        assert evidence.peak_density == 700.0
        assert evidence.elevated_regions == 1  # Only tile2 > 500

    def test_parse_llm_response_valid_json(self, brain: DiagnosticBrain) -> None:
        """Parses valid LLM JSON response."""
        response = json.dumps({
            "final_severity_score": 2,
            "confidence_score": 0.85,
            "justification": "Multiple plasmodia confirmed in Gills and Mantle.",
            "suggested_next_steps": "Histological confirmation recommended.",
        })

        result = brain._parse_llm_response(response)

        assert result["final_severity_score"] == 2
        assert result["confidence_score"] == 0.85
        assert "plasmodia" in result["justification"]

    def test_parse_llm_response_clamps_severity(self, brain: DiagnosticBrain) -> None:
        """Clamps severity to [0, 3] range."""
        response = json.dumps({
            "final_severity_score": 5,
            "confidence_score": 0.9,
            "justification": "Test",
            "suggested_next_steps": "Test",
        })

        result = brain._parse_llm_response(response)

        assert result["final_severity_score"] == 3

    def test_parse_llm_response_handles_markdown(self, brain: DiagnosticBrain) -> None:
        """Strips markdown code blocks from response."""
        response = """```
{
    "final_severity_score": 1,
    "confidence_score": 0.7,
    "justification": "Light infection.",
    "suggested_next_steps": "Monitor."
}
```"""

        result = brain._parse_llm_response(response)

        assert result["final_severity_score"] == 1

    def test_default_result_heuristic_severity_1(
        self, brain: DiagnosticBrain
    ) -> None:
        """Heuristic returns severity 1 for 1-5 confirmed organisms."""
        evidence = CaseEvidence(
            wsi_name="test.ome.tiff",
            vlm_confirmed=3,
            organ_counts={"Gills": 3},
        )

        result = brain._default_result(evidence)

        assert result["final_severity_score"] == 1

    def test_default_result_heuristic_severity_2(
        self, brain: DiagnosticBrain
    ) -> None:
        """Heuristic returns severity 2 for multiple organs affected."""
        evidence = CaseEvidence(
            wsi_name="test.ome.tiff",
            vlm_confirmed=7,
            organ_counts={"Gills": 4, "Mantle": 3},
        )

        result = brain._default_result(evidence)

        assert result["final_severity_score"] == 2

    def test_default_result_heuristic_severity_3(
        self, brain: DiagnosticBrain
    ) -> None:
        """Heuristic returns severity 3 for systemic infection."""
        evidence = CaseEvidence(
            wsi_name="test.ome.tiff",
            vlm_confirmed=15,
            organ_counts={"Gills": 5, "Mantle": 5, "Digestive Gland": 5},
        )

        result = brain._default_result(evidence)

        assert result["final_severity_score"] == 3

    def test_diagnostic_report_to_dict(self) -> None:
        """DiagnosticReport serializes to JSON-compatible dict."""
        evidence = CaseEvidence(
            wsi_name="test.ome.tiff",
            yolo_count=10,
            vlm_confirmed=5,
            mean_density=450.0,
            peak_density=800.0,
            organ_counts={"Gills": 3, "Mantle": 2},
        )

        report = DiagnosticReport(
            wsi_name="test.ome.tiff",
            final_severity_score=2,
            confidence_score=0.85,
            justification="Test justification.",
            suggested_next_steps="Test steps.",
            evidence=evidence,
        )

        d = report.to_dict()

        assert d["wsi_name"] == "test.ome.tiff"
        assert d["final_severity_score"] == 2
        assert d["evidence_summary"]["yolo_count"] == 10
        assert d["evidence_summary"]["organ_distribution"] == {"Gills": 3, "Mantle": 2}
