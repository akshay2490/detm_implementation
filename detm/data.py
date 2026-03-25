"""
detm/data.py
------------
Data preprocessing and loading for the Dynamic Embedded Topic Model.

Fully domain-agnostic: no UN-specific stopwords or column names are
hard-coded.  Domain customisation is driven entirely by ``DETMConfig``.

Public API
----------
    DataPreprocessor   — cleans text, splits documents, builds vocab, BoW, temporal index
    EmbeddingGenerator — trains / loads Word2Vec skip-gram, extracts embedding matrix
    DETMDataset        — torch Dataset wrapping (BoW, time_index) pairs
    create_dataloaders — splits processed data into train/val/test DataLoaders

Cache invalidation
------------------
``preprocess_corpus`` writes a SHA-256 fingerprint of vocab-affecting config
fields alongside the processed pickle.  On subsequent runs the fingerprint is
compared; a mismatch triggers automatic deletion of the stale preprocessing
cache *and* any cached word2vec / embedding files, then re-runs from scratch.
"""

from __future__ import annotations

import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from detm.config import DETMConfig

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_stopwords_file(path: str) -> Set[str]:
    """Read one lowercased word per line from a plain-text stopwords file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stopwords file not found: {p}")
    with open(p) as f:
        return {line.strip().lower() for line in f if line.strip()}


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Preprocesses a corpus DataFrame for DETM.

    Configuration is driven entirely by ``DETMConfig`` — no domain-specific
    stopwords or column names are hard-coded here.

    Expected DataFrame columns
    --------------------------
    - ``config.TEXT_COLUMN``  : raw document text (str)
    - ``config.TIME_COLUMN``  : optional temporal key (e.g. int year)
                                 Set ``config.TIME_COLUMN = ""`` to disable.

    Pipeline
    --------
    1. Split documents on ``config.SPLIT_DELIMITER`` (if non-empty)
    2. Lowercase → word_tokenize → alpha-only → length filter → stopword removal
       → optional lemmatisation (controlled by ``config.LEMMATIZE``)
    3. Apply ``MIN_DOC_LENGTH`` filter on cleaned tokens
    4. Build vocabulary from training split only (prevents val/test leakage)
    5. Convert all documents to bag-of-words sparse matrix
    6. Sort by time column (if present) and build temporal index

    Stopwords
    ---------
    The stopword set is composed from three independent sources (all optional):
      1. NLTK English stopwords   (``config.USE_NLTK_STOPWORDS``)
      2. A plain-text file        (``config.STOPWORDS_FILE``)
      3. An inline list           (``config.EXTRA_STOPWORDS``)
    """

    _PREPROCESSED_FNAME = "preprocessed_data.pkl"

    def __init__(self, config: DETMConfig) -> None:
        self.config = config

        # Build stopword set from config sources only
        sw: Set[str] = set()
        if config.USE_NLTK_STOPWORDS:
            from nltk.corpus import stopwords as _nltk_sw
            sw |= set(_nltk_sw.words("english"))
        if config.STOPWORDS_FILE:
            sw |= _load_stopwords_file(config.STOPWORDS_FILE)
        if config.EXTRA_STOPWORDS:
            sw |= {w.lower() for w in config.EXTRA_STOPWORDS}
        self.stopwords: Set[str] = sw

        # Lemmatizer — only instantiated if needed
        self._lemmatizer = None
        if config.LEMMATIZE:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()

        # Populated after preprocess_corpus()
        self.vocabulary: Optional[List[str]] = None
        self.word2idx: Optional[Dict[str, int]] = None
        self.idx2word: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def preprocess_corpus(
        self,
        df: pd.DataFrame,
        save: bool = True,
        force_retrain: bool = False,
    ) -> Dict:
        """
        Full preprocessing pipeline with fingerprint-based cache invalidation.

        Parameters
        ----------
        df            : DataFrame with text and (optionally) time columns
        save          : persist processed data to
                        ``config.DATA_DIR / preprocessed_data.pkl``
        force_retrain : ignore cache and rerun preprocessing unconditionally

        Returns
        -------
        Dict with keys:
            bow_matrix    – (N, V) scipy CSR float32 sparse matrix
            tokens_list   – list of cleaned token lists (post-split)
            vocabulary_info – vocab / word2idx / word_freq / doc_freq
            metadata      – filtered & sorted DataFrame
            num_docs      – N
            vocab_size    – V
            temporal_info – dict with doc_to_time, avg_bow_per_time, etc.
        """
        cache_path = self.config.DATA_DIR / self._PREPROCESSED_FNAME
        fingerprint = self.config.preprocessing_fingerprint()

        # ── Cache load ────────────────────────────────────────────────────────
        if not force_retrain and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            cached_fp = cached.get("_fingerprint", "")
            if cached_fp == fingerprint:
                processed = cached["processed_data"]
                self.vocabulary = cached["vocabulary"]
                self.word2idx = cached["word2idx"]
                self.idx2word = cached["idx2word"]
                print(f"✓ Loaded preprocessed data from cache ({cache_path})")
                print(f"  Docs: {processed['num_docs']:,}  Vocab: {processed['vocab_size']:,}")
                return processed
            else:
                print(
                    "⚠ Config fingerprint mismatch — preprocessing cache is stale. "
                    "Deleting and re-running …"
                )
                self._invalidate_caches()

        # ── Full preprocessing ─────────────────────────────────────────────────
        processed = self._run_pipeline(df)

        if save:
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "_fingerprint": fingerprint,
                        "processed_data": processed,
                        "vocabulary": self.vocabulary,
                        "word2idx": self.word2idx,
                        "idx2word": self.idx2word,
                    },
                    f,
                )
            print(f"Preprocessed data saved → {cache_path}")

        return processed

    def clean_text(self, text: str) -> List[str]:
        """Lowercase → tokenize → alpha-only → length filter → stopwords → [lemmatize]."""
        from nltk.tokenize import word_tokenize

        text = str(text).lower()
        tokens = word_tokenize(text)
        min_len = self.config.MIN_WORD_LENGTH
        result = []
        for tok in tokens:
            if not tok.isalpha():
                continue
            if len(tok) < min_len:
                continue
            if tok in self.stopwords:
                continue
            if self._lemmatizer is not None:
                tok = self._lemmatizer.lemmatize(tok)
            result.append(tok)
        return result

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _invalidate_caches(self) -> None:
        """Delete stale preprocessing pickle and any cached W2V / embedding files."""
        to_delete = [
            self.config.DATA_DIR / self._PREPROCESSED_FNAME,
            self.config.DATA_DIR / EmbeddingGenerator.W2V_FILENAME,
            self.config.DATA_DIR / EmbeddingGenerator.EMB_FILENAME,
        ]
        for p in to_delete:
            if p.exists():
                p.unlink()
                print(f"  Deleted stale cache: {p}")

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, df: pd.DataFrame) -> Dict:
        print("\n" + "=" * 60)
        print("PREPROCESSING CORPUS")
        print("=" * 60)
        print(f"  Stopwords   : {len(self.stopwords)} total "
              f"(NLTK={self.config.USE_NLTK_STOPWORDS}, "
              f"file={bool(self.config.STOPWORDS_FILE)}, "
              f"extra={len(self.config.EXTRA_STOPWORDS)})")
        print(f"  Lemmatize   : {self.config.LEMMATIZE}")
        print(f"  Min word len: {self.config.MIN_WORD_LENGTH}")
        print(f"  Delimiter   : {repr(self.config.SPLIT_DELIMITER)}")

        text_col = self.config.TEXT_COLUMN
        time_col = self.config.TIME_COLUMN

        # Step 1 — paragraph / sentence splitting
        if self.config.SPLIT_DELIMITER:
            print(f"\nStep 1 — Splitting documents on {repr(self.config.SPLIT_DELIMITER)} …")
            df = self._split_documents(df, text_col)
            print(f"  Expanded to {len(df):,} segments")
        else:
            print("\nStep 1 — No splitting (one row = one document)")

        # Step 2 — tokenise
        print("\nStep 2 — Tokenising …")
        tokens_raw = [self.clean_text(t) for t in tqdm(df[text_col], desc="Tokenizing")]

        # Step 3 — minimum length filter
        min_len = self.config.MIN_DOC_LENGTH
        keep = [len(t) >= min_len for t in tokens_raw]
        df = df[keep].reset_index(drop=True)
        tokens_all = [t for t, k in zip(tokens_raw, keep) if k]
        print(f"  Docs after length filter (≥{min_len}): {len(df):,} "
              f"(removed {sum(1 for k in keep if not k):,})")

        # Step 4 — build vocabulary from training split only
        print("\nStep 4 — Building vocabulary from training split only …")
        n_total = len(tokens_all)
        rng = np.random.default_rng(self.config.SEED)
        perm = rng.permutation(n_total)
        n_train = int(self.config.TRAIN_SPLIT * n_total)
        train_tokens = [tokens_all[i] for i in perm[:n_train]]
        vocab_info = self._build_vocabulary(train_tokens)

        # Step 5 — BoW conversion (sparse)
        print("\nStep 5 — Building sparse BoW matrix …")
        row_list = []
        valid_idx = []
        for i, tok in enumerate(tqdm(tokens_all, desc="BoW")):
            b = self._doc_to_bow(tok)
            if b.sum() > 0:
                row_list.append(b)
                valid_idx.append(i)
        bow_matrix = sp.csr_matrix(np.stack(row_list, axis=0).astype(np.float32))
        df = df.iloc[valid_idx].reset_index(drop=True)
        tokens_all = [tokens_all[i] for i in valid_idx]
        print(f"  Final docs: {len(row_list):,}  Vocab size: {bow_matrix.shape[1]:,}")

        # Step 6 — sort by time (if present)
        if time_col and time_col in df.columns:
            print(f"\nStep 6 — Sorting by '{time_col}' …")
            order = df[time_col].argsort().values
            bow_matrix = bow_matrix[order]
            df = df.iloc[order].reset_index(drop=True)
            tokens_all = [tokens_all[i] for i in order]
        else:
            print("\nStep 6 — No time column; skipping sort")

        # Step 7 — temporal index
        print("\nStep 7 — Building temporal index …")
        temporal_info = self._create_temporal_index(df, bow_matrix, time_col)

        return {
            "bow_matrix": bow_matrix,
            "tokens_list": tokens_all,
            "vocabulary_info": vocab_info,
            "metadata": df,
            "num_docs": len(df),
            "vocab_size": len(self.vocabulary),
            "temporal_info": temporal_info,
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split_documents(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Expand each row into multiple rows by splitting on SPLIT_DELIMITER."""
        delim = self.config.SPLIT_DELIMITER
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting"):
            segments = [s.strip() for s in str(row[text_col]).split(delim) if s.strip()]
            if not segments:
                segments = [str(row[text_col])]
            for seg in segments:
                new_row = row.to_dict()
                new_row[text_col] = seg
                rows.append(new_row)
        return pd.DataFrame(rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocabulary(self, documents: List[List[str]]) -> Dict:
        print("  Counting frequencies …")
        word_freq: Counter = Counter()
        doc_freq: Counter = Counter()
        for doc in tqdm(documents, desc="Counting", leave=False):
            word_freq.update(doc)
            doc_freq.update(set(doc))

        N = len(documents)
        min_df = self.config.MIN_DF
        max_df = int(self.config.MAX_DF * N)
        valid = [w for w, cnt in doc_freq.items() if min_df <= cnt <= max_df]

        if self.config.MAX_VOCAB_SIZE > 0:
            # IDF-ranked selection (for very large corpora)
            idf = {w: np.log((N + 1) / (doc_freq[w] + 1)) for w in valid}
            top_words = sorted(valid, key=lambda w: idf[w], reverse=True)[
                : self.config.MAX_VOCAB_SIZE
            ]
            print(f"  Vocab: {len(top_words):,} (IDF-ranked top-{self.config.MAX_VOCAB_SIZE} "
                  f"from {len(valid):,} passing df gates)")
        else:
            # Alphabetical — matches original DETM paper and notebook
            top_words = sorted(valid)
            print(f"  Vocab: {len(top_words):,} (all words passing df gates, alphabetical)")

        self.word2idx = {w: i for i, w in enumerate(top_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocabulary = top_words

        return {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocabulary": self.vocabulary,
            "word_freq": {w: word_freq[w] for w in top_words},
            "doc_freq": {w: doc_freq[w] for w in top_words},
        }

    def _doc_to_bow(self, tokens: List[str]) -> np.ndarray:
        bow = np.zeros(len(self.vocabulary), dtype=np.float32)
        for t in tokens:
            if t in self.word2idx:
                bow[self.word2idx[t]] += 1
        return bow

    # ------------------------------------------------------------------
    # Temporal index
    # ------------------------------------------------------------------

    def _create_temporal_index(
        self, df: pd.DataFrame, bow_matrix: sp.csr_matrix, time_col: str
    ) -> Dict:
        if not time_col or time_col not in df.columns:
            avg_bow = np.asarray(bow_matrix.mean(axis=0)).ravel().astype(np.float32)
            print("  Single time step (no time column)")
            return {
                "time_steps": [0],
                "doc_to_time": np.zeros(len(df), dtype=np.int64),
                "time_to_docs": {0: list(range(len(df)))},
                "avg_bow_per_time": avg_bow[np.newaxis, :],
                "num_time_steps": 1,
            }

        time_steps = sorted(df[time_col].unique())
        t2i = {y: i for i, y in enumerate(time_steps)}
        doc_to_time = np.array([t2i[y] for y in df[time_col]], dtype=np.int64)

        t2docs: Dict[int, List[int]] = {i: [] for i in range(len(time_steps))}
        for doc_i, t_i in enumerate(doc_to_time):
            t2docs[t_i].append(doc_i)

        V = bow_matrix.shape[1]
        avg_bow = np.zeros((len(time_steps), V), dtype=np.float32)
        for t_i, idxs in t2docs.items():
            if idxs:
                # Raw count average (length-weighted) — matches original DETM paper.
                # Do NOT L1-normalise before averaging; that loses length information.
                time_docs = bow_matrix[idxs]
                avg_bow[t_i] = np.asarray(time_docs.mean(axis=0)).ravel()

        print(f"  Time steps: {len(time_steps)} ({time_steps[0]}–{time_steps[-1]}), "
              f"avg {len(df) / len(time_steps):.1f} docs/step")
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

    def __init__(self, config: DETMConfig) -> None:
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

        Loads cached model if available (and not force_retrain).
        If cache is absent, trains from ``tokens_list``.
        OOV words receive random unit vectors.
        """
        w2v_path = self.config.DATA_DIR / self.W2V_FILENAME

        if not force_retrain and w2v_path.exists():
            self.w2v_model = Word2Vec.load(str(w2v_path))
            print(f"Word2Vec loaded from cache ({len(self.w2v_model.wv):,} words)")
        else:
            if tokens_list is None:
                raise ValueError(
                    "tokens_list is required to train Word2Vec (no cache found)."
                )
            self._train(tokens_list)

        rng = np.random.default_rng(self.config.SEED)
        dim = self.config.EMBEDDING_DIM
        embeddings = np.zeros((len(vocabulary), dim), dtype=np.float32)
        oov = []
        for i, word in enumerate(vocabulary):
            if word in self.w2v_model.wv:
                embeddings[i] = self.w2v_model.wv[word]
            else:
                v = rng.standard_normal(dim).astype(np.float32)
                embeddings[i] = v / (np.linalg.norm(v) + 1e-10)
                oov.append(word)
        if oov:
            print(f"OOV words ({len(oov)}): {oov[:10]}{'…' if len(oov) > 10 else ''}")

        emb_path = self.config.DATA_DIR / self.EMB_FILENAME
        np.save(emb_path, embeddings)
        print(f"Embeddings {embeddings.shape} saved → {emb_path}")
        return embeddings

    def _train(self, tokens_list: List[List[str]]) -> None:
        np.random.seed(self.config.SEED)
        self.w2v_model = Word2Vec(
            sentences=tokens_list,
            vector_size=self.config.EMBEDDING_DIM,
            window=self.config.W2V_WINDOW,
            min_count=self.config.MIN_DF,
            workers=self.config.W2V_WORKERS,
            sg=1,
            seed=self.config.SEED,
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
        bow_matrix: sp.csr_matrix,
        time_indices: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        # Store sparse matrix; densify per-sample in __getitem__ to save RAM
        self._bow_sparse = bow_matrix
        self.time_indices = torch.LongTensor(time_indices)
        self.metadata = metadata

    def __len__(self) -> int:
        return self._bow_sparse.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        bow_dense = np.asarray(self._bow_sparse[idx].todense()).ravel().astype(np.float32)
        item: Dict = {
            "bow": torch.FloatTensor(bow_dense),
            "time_idx": self.time_indices[idx],
        }
        if self.metadata is not None:
            item["metadata"] = self.metadata.iloc[idx].to_dict()
        return item


def create_dataloaders(
    processed_data: Dict,
    config: DETMConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split processed data into train / val / test DataLoaders.

    Split strategy (``config.SPLIT_STRATEGY``):
    - ``"random"``        — shuffle with ``config.SEED``, then slice.
                           Avoids temporal bias; val/test cover all periods.
    - ``"chronological"`` — deterministic slice of time-sorted docs.
                           Useful for strict temporal out-of-sample evaluation.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    bow = processed_data["bow_matrix"]
    meta = processed_data.get("metadata")
    time_idx_arr = processed_data["temporal_info"]["doc_to_time"]

    n = bow.shape[0]
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)

    if config.SPLIT_STRATEGY == "random":
        rng = np.random.default_rng(config.SEED)
        perm = rng.permutation(n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        # Sort indices so sparse row access is sequential (much faster)
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)
    else:  # chronological
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n)

    def _make(idx: np.ndarray) -> DETMDataset:
        return DETMDataset(
            bow[idx],
            time_idx_arr[idx],
            meta.iloc[idx].reset_index(drop=True) if meta is not None else None,
        )

    train_ds = _make(train_idx)
    val_ds = _make(val_idx)
    test_ds = _make(test_idx)

    def _loader(ds: DETMDataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=config.BATCH_SIZE,
                          shuffle=shuffle, num_workers=0)

    print(
        f"Dataset splits ({config.SPLIT_STRATEGY}) — "
        f"train: {len(train_ds):,}, val: {len(val_ds):,}, test: {len(test_ds):,}"
    )
    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
