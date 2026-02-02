"""HPC-tier modules: heavy compute for tiling, detection, and biomarkers."""

from src.hpc.biomarkers import BiomarkerAnalyzer, HemocyteResult
from src.hpc.dataset import Annotation, DatasetGenerator, DatasetStats
from src.hpc.detection import Detection, YOLODetector
from src.hpc.tiling import TilingEngine
from src.hpc.train import CLASS_NAMES, TrainingResult, YOLOTrainer

__all__ = [
    "Annotation",
    "BiomarkerAnalyzer",
    "CLASS_NAMES",
    "DatasetGenerator",
    "DatasetStats",
    "Detection",
    "HemocyteResult",
    "TilingEngine",
    "TrainingResult",
    "YOLODetector",
    "YOLOTrainer",
]
