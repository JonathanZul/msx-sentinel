"""HPC-tier modules: heavy compute for tiling and detection."""

from src.hpc.dataset import Annotation, DatasetGenerator, DatasetStats
from src.hpc.detection import Detection, YOLODetector
from src.hpc.tiling import TilingEngine
from src.hpc.train import CLASS_NAMES, TrainingResult, YOLOTrainer

__all__ = [
    "Annotation",
    "CLASS_NAMES",
    "DatasetGenerator",
    "DatasetStats",
    "Detection",
    "TilingEngine",
    "TrainingResult",
    "YOLODetector",
    "YOLOTrainer",
]
