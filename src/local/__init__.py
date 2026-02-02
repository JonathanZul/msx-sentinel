"""Local reasoning modules (Tier 3: VLM Eye, LLM Brain)."""

from .brain import CaseEvidence, DiagnosticBrain, DiagnosticReport
from .reasoning import Candidate, VLMEye

__all__ = [
    "Candidate",
    "CaseEvidence",
    "DiagnosticBrain",
    "DiagnosticReport",
    "VLMEye",
]
