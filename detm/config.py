"""
detm/config.py
--------------
Central configuration dataclass for the Dynamic Embedded Topic Model.

Designed to be domain-agnostic (UN debates, finance, news, medical, etc.).
All text-column names, stopword sources, splitting behaviour, and split
strategy are configurable so no application-specific code leaks into the
default settings.

Usage
-----
    from detm.config import DETMConfig

    cfg = DETMConfig()                          # all defaults
    cfg = DETMConfig(NUM_TOPICS=25, BATCH_SIZE=500)
    cfg = DETMConfig.from_dict({...})           # from a JSON/dict
    cfg = DETMConfig.load("outputs/config.json")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DETMConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path("data")
    MODELS_DIR: Path = Path("models")
    OUTPUTS_DIR: Path = Path("outputs")

    # ── Input column names (change for any domain) ────────────────────────────
    # Set TIME_COLUMN = "" to disable temporal modelling (single time step).
    TEXT_COLUMN: str = "text"
    TIME_COLUMN: str = "year"

    # ── Document splitting ────────────────────────────────────────────────────
    # SPLIT_DELIMITER: each document is split on this string before processing.
    #   ".\n"  — sentence/paragraph boundary (original DETM; UN debates, news)
    #   "\n\n" — blank-line paragraph split
    #   ""     — no splitting; one CSV row = one document
    SPLIT_DELIMITER: str = ".\n"

    # ── Data split ────────────────────────────────────────────────────────────
    # SPLIT_STRATEGY:
    #   "random"         — shuffle with SEED then slice (avoids temporal bias)
    #   "chronological"  — first TRAIN_SPLIT of time-sorted docs → train, etc.
    SPLIT_STRATEGY: str = "random"
    TRAIN_SPLIT: float = 0.85
    VAL_SPLIT: float = 0.05

    # ── Stopwords ─────────────────────────────────────────────────────────────
    # Compose the stopword set from three independent sources:
    #   1. NLTK English stopwords  (USE_NLTK_STOPWORDS = True/False)
    #   2. A plain-text file       (one word per line; STOPWORDS_FILE path or "")
    #   3. An inline list          (EXTRA_STOPWORDS)
    # Leave all empty for no stopword removal.
    USE_NLTK_STOPWORDS: bool = True
    STOPWORDS_FILE: str = ""            # e.g. "stop_words.txt" or ""
    EXTRA_STOPWORDS: List[str] = field(default_factory=list)

    # ── Text cleaning ─────────────────────────────────────────────────────────
    # MIN_WORD_LENGTH: minimum character count for a token to be kept.
    #   1 → keep all alpha tokens (a, I, …)
    #   2 → keep tokens ≥ 2 chars (notebook default: >1 char, keeps EU/UK/UN)
    #   3 → discard 2-char tokens
    MIN_WORD_LENGTH: int = 2

    # LEMMATIZE: apply WordNet lemmatisation after tokenisation.
    #   False — original DETM paper behaviour (no lemmatisation)
    #   True  — collapses morphological variants
    LEMMATIZE: bool = False

    # ── Data preprocessing ────────────────────────────────────────────────────
    MIN_DF: int = 10            # Min document frequency (absolute count)
    MAX_DF: float = 0.7         # Max document frequency (fraction of docs)
    # MAX_VOCAB_SIZE:
    #   0     — keep ALL words passing df gates, sorted alphabetically
    #           (matches original DETM and the notebook)
    #   N > 0 — keep top-N words by IDF score (useful for very large vocab)
    MAX_VOCAB_SIZE: int = 0
    MIN_DOC_LENGTH: int = 5     # Min cleaned-token count to keep a document

    # ── Word2Vec embeddings ───────────────────────────────────────────────────
    EMBEDDING_DIM: int = 300
    W2V_WINDOW: int = 5
    W2V_EPOCHS: int = 10
    W2V_WORKERS: int = 4
    TRAIN_WORD_EMBEDDINGS: bool = True  # Allow ρ to be fine-tuned during training

    # ── Temporal Baseline Encoder (η LSTM) ────────────────────────────────────
    COMPRESSION_DIM: int = 200
    LSTM_LAYERS: int = 3
    LSTM_HIDDEN: int = 200

    # ── Document Topic Encoder (θ MLP) ────────────────────────────────────────
    NUM_TOPICS: int = 50
    DOC_HIDDEN_DIM: int = 800
    DOC_DROPOUT: float = 0.0     # 0.0 — dropout on VAE encoders causes collapse

    # ── Topic Embeddings mean-field (α) ───────────────────────────────────────
    INIT_ALPHA_STD: float = 1.0              # randn scale for α_mu init

    # ── Random-walk prior variances ───────────────────────────────────────────
    ETA_PRIOR_VARIANCE: float = 0.005    # δ²: p(η_t | η_{t-1}) = N(η_{t-1}, δ²I)
    ALPHA_PRIOR_VARIANCE: float = 0.005  # γ²: p(α_t | α_{t-1}) = N(α_{t-1}, γ²I)

    # ── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 700
    NUM_EPOCHS: int = 1000
    LEARNING_RATE: float = 0.0001       # Matches Dieng et al. 2019 default
    WEIGHT_DECAY: float = 1.2e-6
    CLIP_GRAD: float = 0.0              # 0.0 disables clipping (original default)

    # ── Loss weights ─────────────────────────────────────────────────────────
    RECON_WEIGHT: float = 1.0
    KL_THETA_WEIGHT: float = 1.0
    KL_ETA_WEIGHT: float = 1.0
    KL_ALPHA_WEIGHT: float = 1.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    TOP_N_WORDS: List[int] = field(default_factory=lambda: [10, 15, 20])
    COHERENCE_METRICS: List[str] = field(default_factory=lambda: ["c_v", "c_npmi"])

    # ── Checkpointing & LR annealing ──────────────────────────────────────────
    SAVE_EVERY: int = 10
    LR_ANNEAL_PATIENCE: int = 10    # Plateau epochs before dividing LR by 4

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED: int = 42

    # --------------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.DATA_DIR = Path(self.DATA_DIR)
        self.MODELS_DIR = Path(self.MODELS_DIR)
        self.OUTPUTS_DIR = Path(self.OUTPUTS_DIR)

    def make_dirs(self) -> None:
        """Create data / model / output directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("DATA_DIR", "MODELS_DIR", "OUTPUTS_DIR"):
            d[k] = str(d[k])
        return d

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DETMConfig":
        data = dict(d)
        # Legacy field migrations
        if "PATIENCE" in data and "LR_ANNEAL_PATIENCE" not in data:
            data["LR_ANNEAL_PATIENCE"] = data.pop("PATIENCE")
        if "ANNEAL_PATIENCE" in data and "LR_ANNEAL_PATIENCE" not in data:
            data["LR_ANNEAL_PATIENCE"] = data.pop("ANNEAL_PATIENCE")
        # Drop unknown / deprecated fields silently
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: Path | str) -> "DETMConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ── Cache fingerprint ─────────────────────────────────────────────────────

    def preprocessing_fingerprint(self) -> str:
        """
        SHA-256 fingerprint of every preprocessing-affecting config field.

        When the fingerprint changes between runs the cached
        ``preprocessed_data.pkl``, ``word2vec.model``, and
        ``word_embeddings.npy`` are stale and must be regenerated.

        Covered fields
        --------------
        Text cleaning   : MIN_WORD_LENGTH, LEMMATIZE
        Vocabulary gates: MIN_DF, MAX_DF, MAX_VOCAB_SIZE, MIN_DOC_LENGTH
        Stopwords       : USE_NLTK_STOPWORDS, STOPWORDS_FILE, EXTRA_STOPWORDS
        Splitting       : SPLIT_DELIMITER
        Column names    : TEXT_COLUMN, TIME_COLUMN
        Vocab split     : TRAIN_SPLIT, SEED
        """
        fields = {
            "MIN_WORD_LENGTH": self.MIN_WORD_LENGTH,
            "LEMMATIZE": self.LEMMATIZE,
            "MIN_DF": self.MIN_DF,
            "MAX_DF": self.MAX_DF,
            "MAX_VOCAB_SIZE": self.MAX_VOCAB_SIZE,
            "MIN_DOC_LENGTH": self.MIN_DOC_LENGTH,
            "USE_NLTK_STOPWORDS": self.USE_NLTK_STOPWORDS,
            "STOPWORDS_FILE": self.STOPWORDS_FILE,
            "EXTRA_STOPWORDS": sorted(self.EXTRA_STOPWORDS),
            "SPLIT_DELIMITER": self.SPLIT_DELIMITER,
            "TEXT_COLUMN": self.TEXT_COLUMN,
            "TIME_COLUMN": self.TIME_COLUMN,
            "TRAIN_SPLIT": self.TRAIN_SPLIT,
            "SEED": self.SEED,
        }
        blob = json.dumps(fields, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()
