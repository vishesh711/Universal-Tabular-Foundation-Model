"""Task-specific heads for downstream applications."""

from .classification import (
    ClassificationHead,
    BinaryClassificationHead,
    MultiClassClassificationHead,
    MultiLabelClassificationHead
)
from .regression import (
    RegressionHead,
    MultiTargetRegressionHead,
    QuantileRegressionHead
)
from .anomaly_detection import (
    AnomalyDetectionHead,
    ReconstructionAnomalyHead,
    OneClassSVMHead,
    IsolationForestHead
)
from .survival import SurvivalHead
from .base import (
    BaseTaskHead,
    TaskOutput,
    TaskType
)

__all__ = [
    # Base classes
    "BaseTaskHead",
    "TaskOutput", 
    "TaskType",
    
    # Classification
    "ClassificationHead",
    "BinaryClassificationHead",
    "MultiClassClassificationHead",
    "MultiLabelClassificationHead",
    
    # Regression
    "RegressionHead",
    "MultiTargetRegressionHead",
    "QuantileRegressionHead",
    
    # Anomaly Detection
    "AnomalyDetectionHead",
    "ReconstructionAnomalyHead",
    "OneClassSVMHead",
    "IsolationForestHead",
    
    # Survival Analysis
    "SurvivalHead",
]