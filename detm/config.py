"""
detm/config.py
--------------
Central configuration dataclass for the Dynamic Embedded Topic Model.

Usage
-----
    from detm.config import DETMConfig
    cfg = DETMConfig()                   # defaults
    cfg = DETMConfig(NUM_TOPICS=25, BATCH_SIZE=500)  # override
    cfg = DETMConfig.from_dict({...})    # from a JSON/dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class DETMConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path("data")
    MODELS_DIR: Path = Path("models")
    OUTPUTS_DIR: Path = Path("outputs")

    # ── Data preprocessing ────────────────────────────────────────────────────
    MIN_DF: int = 30           # Min document frequency gate
    MAX_DF: float = 0.3        # Max document frequency gate (fraction)
    MAX_VOCAB_SIZE: int = 10_000
    MIN_DOC_LENGTH: int = 15   # Min vocab-token count after cleaning

    # ── Word2Vec embeddings ───────────────────────────────────────────────────
    EMBEDDING_DIM: int = 300
    W2V_WINDOW: int = 5
    W2V_EPOCHS: int = 10
    W2V_WORKERS: int = 4

    # ── Temporal Baseline Encoder (η LSTM) ────────────────────────────────────
    COMPRESSION_DIM: int = 400
    LSTM_LAYERS: int = 4
    LSTM_HIDDEN: int = 400

    # ── Document Topic Encoder (θ MLP) ────────────────────────────────────────
    NUM_TOPICS: int = 50
    DOC_HIDDEN_DIM: int = 800
    DOC_DROPOUT: float = 0.0   # 0.0 matches original — dropout in VAE encoders causes collapse

    # ── Topic Embeddings mean-field (α) ───────────────────────────────────────
    INIT_ALPHA_STD: float = 1.0    # Matches original torch.randn default
    INIT_ALPHA_LOGVAR: float = 0.0  # log(1) = 0, consistent with std=1 init

    # ── Random-walk prior variances ───────────────────────────────────────────
    ETA_PRIOR_VARIANCE: float = 0.005    # δ²: p(η_t | η_{t-1}) = N(η_{t-1}, δ²I)
    ALPHA_PRIOR_VARIANCE: float = 0.005  # γ²: p(α_t | α_{t-1}) = N(α_{t-1}, γ²I)

    # ── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 1000
    NUM_EPOCHS: int = 400
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 1.2e-6
    CLIP_GRAD: float = 2.0

    # ── Loss weights ─────────────────────────────────────────────────────────
    RECON_WEIGHT: float = 1.0
    KL_THETA_WEIGHT: float = 1.0
    KL_ETA_WEIGHT: float = 1.0
    KL_ALPHA_WEIGHT: float = 1.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    TOP_N_WORDS: List[int] = field(default_factory=lambda: [10, 15, 20])
    COHERENCE_METRICS: List[str] = field(default_factory=lambda: ["c_v", "c_npmi"])

    # ── Checkpointing & early stopping ────────────────────────────────────────
    SAVE_EVERY: int = 10
    PATIENCE: int = 15

    # ── Data split ────────────────────────────────────────────────────────────
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1

    def __post_init__(self):
        # Convert str paths to Path objects (useful when loading from JSON)
        self.DATA_DIR = Path(self.DATA_DIR)
        self.MODELS_DIR = Path(self.MODELS_DIR)
        self.OUTPUTS_DIR = Path(self.OUTPUTS_DIR)

    def make_dirs(self) -> None:
        """Create data/model/output directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Serialisation helpers ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert Path objects to strings for JSON compatibility
        for k in ("DATA_DIR", "MODELS_DIR", "OUTPUTS_DIR"):
            d[k] = str(d[k])
        return d

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DETMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path | str) -> "DETMConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
