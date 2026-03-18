"""
detm/data.py
------------
Data preprocessing and loading for the Dynamic Embedded Topic Model.

Public API
----------
    DataPreprocessor   — cleans text, builds vocab, creates BoW + temporal index
    EmbeddingGenerator — trains / loads Word2Vec skip-gram, extracts embedding matrix
    DETMDataset        — torch Dataset wrapping (BoW, time_index) pairs
    create_dataloaders — splits processed data into train/val/test DataLoaders
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from detm.config import DETMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Preprocesses a corpus DataFrame for DETM.

    Expected DataFrame columns
    --------------------------
    - ``text``  : raw speech/document text (str)
    - ``year``  : integer year (used to build temporal time-step index)

    Pipeline
    --------
    1. Lowercase → word_tokenize → alpha-only → stopword removal → lemmatize
    2. Apply ``MIN_DOC_LENGTH`` filter on cleaned tokens
    3. Build vocabulary (MIN_DF / MAX_DF gates, IDF-ranked selection)
    4. Convert to bag-of-words matrix
    5. Sort by year, build temporal index

    UN-specific boilerplate stopwords are included by default; pass
    ``extra_stopwords`` to extend them for other corpora.
    """

    UN_STOPWORDS: frozenset = frozenset({
        # Organisational / procedural
        "united", "nations", "assembly", "general", "security", "council",
        "president", "secretary", "minister", "delegation", "representative",
        "distinguished", "session", "resolution", "agenda", "committee",
        "conference", "plenary", "headquarter", "secretariat",
        # Diplomatic boilerplate
        "reaffirm", "reiterate", "congratulate", "behalf", "honour", "welcome",
        "express", "wish", "convey", "utilize", "hereby",
        "take", "made", "shall", "would", "could", "must", "may",
        # Overly generic
        "international", "world", "countries", "states", "government",
        "republic", "peoples", "nation", "global", "member",
        "also", "well", "one", "two", "new", "first", "many",
        "year", "years", "within", "since", "among", "like",
        # Common filler
        "said", "make", "need", "call", "called", "issue", "issues",
        "important", "ensure", "support", "continue", "including",
        "area", "areas", "level", "levels", "part", "role",
    })

    def __init__(
        self,
        config: DETMConfig,
        extra_stopwords: Optional[set] = None,
    ):
        self.config = config
        sw = set(stopwords.words("english")) | self.UN_STOPWORDS
        if extra_stopwords:
            sw |= extra_stopwords
        self.stopwords = sw
        self.lemmatizer = WordNetLemmatizer()
        # Populated after preprocess_corpus()
        self.vocabulary: Optional[List[str]] = None
        self.word2idx: Optional[Dict[str, int]] = None
        self.idx2word: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def preprocess_corpus(self, df: pd.DataFrame, save: bool = True) -> Dict:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        df   : DataFrame with ``text`` and ``year`` columns
        save : persist processed data to ``config.DATA_DIR/preprocessed_data.pkl``

        Returns
        -------
        Dict with keys:
            bow_matrix    – (N, V) float32 array
            tokens_list   – list of cleaned token lists
            vocabulary_info – vocab / word2idx / idf_scores / ...
            metadata      – filtered & sorted DataFrame
            num_docs      – N
            vocab_size    – V
            temporal_info – dict with doc_to_time, avg_bow_per_time, etc.
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING CORPUS")
        print("=" * 60)

        # 1. Tokenise
        tokens_raw = [self.clean_text(t) for t in tqdm(df["text"], desc="Tokenizing")]

        # 2. Length filter
        min_len = self.config.MIN_DOC_LENGTH
        keep = [len(t) >= min_len for t in tokens_raw]
        df_f = df[keep].reset_index(drop=True)
        tokens = [t for t, k in zip(tokens_raw, keep) if k]
        print(f"Docs after length filter (≥{min_len}): {len(df_f)} "
              f"(removed {sum(1 for k in keep if not k)})")

        # 3. Build vocabulary
        vocab_info = self._build_vocabulary(tokens)

        # 4. BoW conversion — drop empty docs
        bows, valid_idx = [], []
        for i, tok in enumerate(tqdm(tokens, desc="Building BoW")):
            b = self._doc_to_bow(tok)
            if b.sum() > 0:
                bows.append(b)
                valid_idx.append(i)
        bow_matrix = np.array(bows, dtype=np.float32)
        df_f = df_f.iloc[valid_idx].reset_index(drop=True)
        tokens = [tokens[i] for i in valid_idx]
        print(f"Final documents: {len(bow_matrix)}, vocabulary: {bow_matrix.shape[1]}")

        # 5. Sort by year
        if "year" in df_f.columns:
            order = df_f["year"].argsort()
            bow_matrix = bow_matrix[order]
            df_f = df_f.iloc[order].reset_index(drop=True)
            tokens = [tokens[i] for i in order]

        # 6. Temporal index
        temporal_info = self._create_temporal_index(df_f, bow_matrix)

        processed = {
            "bow_matrix": bow_matrix,
            "tokens_list": tokens,
            "vocabulary_info": vocab_info,
            "metadata": df_f,
            "num_docs": len(bow_matrix),
            "vocab_size": len(self.vocabulary),
            "temporal_info": temporal_info,
        }

        if save:
            path = self.config.DATA_DIR / "preprocessed_data.pkl"
            with open(path, "wb") as f:
                pickle.dump(processed, f)
            print(f"Preprocessed data saved → {path}")

        return processed

    def clean_text(self, text: str) -> List[str]:
        """Lowercase → tokenize → alpha-only → stopword removal → lemmatize."""
        text = str(text).lower()
        tokens = word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok.isalpha() and len(tok) > 2 and tok not in self.stopwords
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vocabulary(self, documents: List[List[str]]) -> Dict:
        print("Building vocabulary...")
        word_freq: Counter = Counter()
        doc_freq: Counter = Counter()
        for doc in tqdm(documents, desc="Counting"):
            word_freq.update(doc)
            doc_freq.update(set(doc))

        N = len(documents)
        min_df = self.config.MIN_DF
        max_df = int(self.config.MAX_DF * N)

        valid = [w for w, df in doc_freq.items() if min_df <= df <= max_df]
        idf = {w: np.log((N + 1) / (doc_freq[w] + 1)) for w in valid}
        top_words = sorted(valid, key=lambda w: idf[w], reverse=True)[: self.config.MAX_VOCAB_SIZE]

        self.word2idx = {w: i for i, w in enumerate(top_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocabulary = top_words

        print(f"Vocabulary size: {len(top_words)} "
              f"(from {len(word_freq)} unique tokens, {len(valid)} pass df gates)")
        return {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocabulary": self.vocabulary,
            "word_freq": {w: word_freq[w] for w in top_words},
            "doc_freq": {w: doc_freq[w] for w in top_words},
            "idf_scores": {w: idf[w] for w in top_words},
        }

    def _doc_to_bow(self, tokens: List[str]) -> np.ndarray:
        bow = np.zeros(len(self.vocabulary), dtype=np.float32)
        for t in tokens:
            if t in self.word2idx:
                bow[self.word2idx[t]] += 1
        return bow

    def _create_temporal_index(self, df: pd.DataFrame, bow_matrix: np.ndarray) -> Dict:
        if "year" not in df.columns:
            return {
                "time_steps": [0],
                "doc_to_time": np.zeros(len(df), dtype=np.int64),
                "time_to_docs": {0: list(range(len(df)))},
                "avg_bow_per_time": bow_matrix.mean(axis=0, keepdims=True),
                "num_time_steps": 1,
            }

        time_steps = sorted(df["year"].unique())
        t2i = {y: i for i, y in enumerate(time_steps)}
        doc_to_time = np.array([t2i[y] for y in df["year"]], dtype=np.int64)

        t2docs: Dict[int, List[int]] = {i: [] for i in range(len(time_steps))}
        for doc_i, t_i in enumerate(doc_to_time):
            t2docs[t_i].append(doc_i)

        avg_bow = np.zeros((len(time_steps), bow_matrix.shape[1]), dtype=np.float32)
        for t_i, idxs in t2docs.items():
            if idxs:
                rows = bow_matrix[idxs]
                norms = rows.sum(axis=1, keepdims=True) + 1e-10
                avg_bow[t_i] = (rows / norms).mean(axis=0)

        print(f"Temporal structure: {len(time_steps)} time steps "
              f"({time_steps[0]}–{time_steps[-1]}), "
              f"avg {len(df)/len(time_steps):.1f} docs/step")
        return {
            "time_steps": time_steps,
            "doc_to_time": doc_to_time,
            "time_to_docs": t2docs,
            "avg_bow_per_time": avg_bow,
            "num_time_steps": len(time_steps),
        }


# ---------------------------------------------------------------------------
# EmbeddingGenerator
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    Trains (or loads cached) Word2Vec skip-gram embeddings aligned with the
    DETM vocabulary.

    Word2Vec skip-gram is used because its inner-product training objective
    directly matches DETM's β = softmax(α · ρᵀ) decoder geometry, giving
    angularly spread embeddings (avg pairwise cosine ~0.05–0.20) compared to
    the narrow cone of transformer-based encoders (~0.65–0.80).

    Caching
    -------
    Saved to ``config.DATA_DIR/word2vec.model`` on first run; automatically
    reloaded on subsequent runs.  Pass ``force_retrain=True`` to skip cache.
    """

    W2V_FILENAME = "word2vec.model"
    EMB_FILENAME = "word_embeddings.npy"

    def __init__(self, config: DETMConfig):
        self.config = config
        self.w2v_model: Optional[Word2Vec] = None

    def generate_vocabulary_embeddings(
        self,
        vocabulary: List[str],
        tokens_list: Optional[List[List[str]]] = None,
        force_retrain: bool = False,
    ) -> np.ndarray:
        """
        Return (vocab_size, embedding_dim) float32 embedding matrix.

        Loads cached model if available, otherwise trains from ``tokens_list``.
        OOV words receive random unit vectors.
        """
        w2v_path = self.config.DATA_DIR / self.W2V_FILENAME
        if not force_retrain and w2v_path.exists():
            self.w2v_model = Word2Vec.load(str(w2v_path))
            print(f"Word2Vec loaded from cache ({len(self.w2v_model.wv):,} words)")
        else:
            if tokens_list is None:
                raise ValueError("tokens_list required to train Word2Vec (no cache found).")
            self._train(tokens_list)

        dim = self.config.EMBEDDING_DIM
        embeddings = np.zeros((len(vocabulary), dim), dtype=np.float32)
        oov = []
        for i, word in enumerate(vocabulary):
            if word in self.w2v_model.wv:
                embeddings[i] = self.w2v_model.wv[word]
            else:
                v = np.random.randn(dim).astype(np.float32)
                embeddings[i] = v / (np.linalg.norm(v) + 1e-10)
                oov.append(word)
        if oov:
            print(f"OOV words ({len(oov)}): {oov[:10]}{'...' if len(oov) > 10 else ''}")

        emb_path = self.config.DATA_DIR / self.EMB_FILENAME
        np.save(emb_path, embeddings)
        print(f"Embeddings ({embeddings.shape}) saved → {emb_path}")
        return embeddings

    def _train(self, tokens_list: List[List[str]]) -> None:
        np.random.seed(_SEED)
        self.w2v_model = Word2Vec(
            sentences=tokens_list,
            vector_size=self.config.EMBEDDING_DIM,
            window=self.config.W2V_WINDOW,
            min_count=self.config.MIN_DF,
            workers=self.config.W2V_WORKERS,
            sg=1,
            seed=_SEED,
            epochs=self.config.W2V_EPOCHS,
        )
        path = self.config.DATA_DIR / self.W2V_FILENAME
        self.w2v_model.save(str(path))
        print(f"Word2Vec trained ({len(self.w2v_model.wv):,} words) → {path}")


# ---------------------------------------------------------------------------
# Dataset & DataLoaders
# ---------------------------------------------------------------------------

class DETMDataset(Dataset):
    """PyTorch Dataset wrapping (BoW, time_index) pairs."""

    def __init__(
        self,
        bow_matrix: np.ndarray,
        time_indices: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
    ):
        self.bow = torch.FloatTensor(bow_matrix)
        self.time_indices = torch.LongTensor(time_indices)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.bow)

    def __getitem__(self, idx: int) -> Dict:
        item = {"bow": self.bow[idx], "time_idx": self.time_indices[idx]}
        if self.metadata is not None:
            item["metadata"] = self.metadata.iloc[idx].to_dict()
        return item


def create_dataloaders(
    processed_data: Dict,
    config: DETMConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split processed data into train / val / test DataLoaders.

    Split is chronological (documents are pre-sorted by year), matching the
    evaluation protocol of the original DETM paper.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    bow = processed_data["bow_matrix"]
    meta = processed_data["metadata"]
    time_idx = processed_data["temporal_info"]["doc_to_time"]

    n = len(bow)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)

    def _make(slc) -> DETMDataset:
        return DETMDataset(
            bow[slc],
            time_idx[slc],
            meta.iloc[slc] if meta is not None else None,
        )

    train_ds = _make(slice(None, n_train))
    val_ds = _make(slice(n_train, n_train + n_val))
    test_ds = _make(slice(n_train + n_val, None))

    def _loader(ds, shuffle) -> DataLoader:
        return DataLoader(ds, batch_size=config.BATCH_SIZE,
                          shuffle=shuffle, num_workers=0)

    print(f"Dataset splits — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
