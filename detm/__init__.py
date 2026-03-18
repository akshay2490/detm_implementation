"""
detm — Dynamic Embedded Topic Model
=====================================

Public API re-exports.  Import from here for stable names::

    from detm import DETMConfig, DETM, DETMTrainer, TopicEvaluator
    from detm import DataPreprocessor, EmbeddingGenerator
    from detm import DETMDataset, create_dataloaders
"""

from detm.config import DETMConfig
from detm.data import (
    DataPreprocessor,
    DETMDataset,
    EmbeddingGenerator,
    create_dataloaders,
)
from detm.evaluate import TopicEvaluator
from detm.model import (
    DETM,
    DETMDecoder,
    DocumentTopicEncoder,
    ReconstructionLoss,
    TemporalBaselineEncoder,
)
from detm.train import DETMTrainer

__all__ = [
    # Config
    "DETMConfig",
    # Data
    "DataPreprocessor",
    "EmbeddingGenerator",
    "DETMDataset",
    "create_dataloaders",
    # Model components
    "DocumentTopicEncoder",
    "DETMDecoder",
    "TemporalBaselineEncoder",
    "ReconstructionLoss",
    "DETM",
    # Training
    "DETMTrainer",
    # Evaluation
    "TopicEvaluator",
]
