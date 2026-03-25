# Dynamic Embedded Topic Model (DETM)

A generalized, domain-agnostic implementation of the [Dynamic Embedded Topic Model](https://arxiv.org/abs/1907.05545) (Dieng, Ruiz & Blei, 2019).

Tracks how topics evolve over time by combining Word2Vec skip-gram embeddings with a temporal Gaussian random-walk prior.  Works with any domain — UN debates, financial earnings calls, news archives, medical literature, etc. — by routing all domain-specific settings through configuration and CLI flags.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Domain Examples](#domain-examples)
6. [Cache Invalidation](#cache-invalidation)
7. [Python API](#python-api)
8. [Configuration Reference](#configuration-reference)
9. [Model Architecture](#model-architecture)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Output Files](#output-files)

---

## Project Structure

```
detm_implementation/
├── detm/                       ← installable Python package
│   ├── __init__.py             ← public re-exports
│   ├── config.py               ← DETMConfig dataclass (all hyperparameters)
│   ├── data.py                 ← DataPreprocessor · EmbeddingGenerator · create_dataloaders
│   ├── model.py                ← DETM · DocumentTopicEncoder · TemporalBaselineEncoder
│   ├── train.py                ← DETMTrainer (Adam · LR annealing · TensorBoard · checkpointing)
│   └── evaluate.py             ← TopicEvaluator (coherence · diversity · perplexity)
├── train_detm.py               ← CLI entry point (end-to-end pipeline)
├── notebooks/
│   └── detm_un_debates.ipynb   ← interactive notebook (UN debates)
├── data/                       ← CSV input + auto-generated caches
├── models/                     ← checkpoints (detm_best.pt, detm_epoch_N.pt)
├── outputs/                    ← results.json · topics_evolution.txt · TensorBoard logs
├── stop_words.txt              ← example domain stopwords file
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Activate the virtual environment
source detm_env/bin/activate

# 2. (If starting fresh) install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"
```

---

## Quick Start

### UN General Debates (default settings)

```bash
# Download dataset first (requires Kaggle credentials)
kaggle datasets download -d unitednations/un-general-debates -p data --unzip

# Run full pipeline: preprocess → embed → train → evaluate
python train_detm.py --data_path data/un-general-debates.csv
```

### Finance corpus (custom columns, domain stopwords)

```bash
python train_detm.py \
    --data_path data/earnings_calls.csv \
    --text_column transcript \
    --time_column quarter \
    --split_delimiter "" \
    --stopwords_file finance_stopwords.txt \
    --no_nltk_stopwords \
    --config_overrides '{"NUM_TOPICS": 30}'
```

### Resume training from a checkpoint

```bash
python train_detm.py \
    --data_path data/corpus.csv \
    --checkpoint models/detm_best.pt
```

---

## CLI Reference

```
python train_detm.py --data_path PATH [options]
```

### Data

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | *(required)* | Path to input CSV file |
| `--text_column` | `text` | Column name containing document text |
| `--time_column` | `year` | Column name for temporal key (int or str). Pass `""` to disable temporal modelling |

### Preprocessing

| Flag | Default | Description |
|------|---------|-------------|
| `--split_delimiter` | `".\n"` | Split each row into segments on this string before tokenising. `""` = no splitting (one row = one document) |
| `--split_strategy` | `random` | `random` — shuffle with seed then split (avoids temporal bias). `chronological` — first N% of time-sorted docs → train |
| `--lemmatize` | off | Enable WordNet lemmatisation |
| `--force_retrain` | off | Ignore all caches and rerun preprocessing + Word2Vec from scratch |

### Stopwords

| Flag | Default | Description |
|------|---------|-------------|
| `--stopwords_file` | none | Path to plain-text file, one stopword per line |
| `--extra_stopwords` | none | Space-separated inline stopwords, e.g. `--extra_stopwords foo bar baz` |
| `--no_nltk_stopwords` | off | Disable NLTK English stopwords (useful for non-English corpora or when supplying a custom list) |

> **Default stopword behaviour** with no flags: NLTK English stopwords only.  
> All three sources can be combined freely.

### Config

| Flag | Description |
|------|-------------|
| `--config_path PATH` | Load a saved `config.json` (overrides all defaults) |
| `--config_overrides JSON` | Inline JSON overrides, e.g. `'{"NUM_TOPICS": 30, "BATCH_SIZE": 500}'` |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | none | Resume training from a `.pt` checkpoint |
| `--epochs N` | from config | Override `NUM_EPOCHS` |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:0`, or `auto` |
| `--seed` | `42` | Random seed |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--results_path PATH` | `outputs/results.json` | Where to write evaluation results JSON |
| `--skip_eval` | off | Skip post-training evaluation |

---

## Domain Examples

### UN / Diplomatic corpus
```bash
python train_detm.py \
    --data_path data/un-general-debates.csv \
    --stopwords_file stop_words.txt \
    --split_delimiter ".\n" \
    --config_overrides '{"NUM_TOPICS": 50, "MIN_DF": 10, "MAX_DF": 0.7}'
```
Paragraph splitting on `".\n"` converts each speech into many shorter documents, which improves topic granularity.

### Financial earnings calls
```bash
python train_detm.py \
    --data_path data/earnings_calls.csv \
    --text_column transcript \
    --time_column quarter \
    --split_delimiter "" \
    --no_nltk_stopwords \
    --stopwords_file finance_stopwords.txt \
    --extra_stopwords thank thanks good morning afternoon \
    --config_overrides '{"NUM_TOPICS": 25, "MIN_DF": 5}'
```
No paragraph splitting (each call is already a document), no NLTK stopwords (domain-specific list is better).

### News archive (no temporal structure)
```bash
python train_detm.py \
    --data_path data/news.csv \
    --text_column body \
    --time_column "" \
    --split_delimiter "\n\n" \
    --config_overrides '{"NUM_TOPICS": 40}'
```
Setting `--time_column ""` disables temporal modelling — the model runs as a static ETM.

### Medical abstracts (with lemmatisation)
```bash
python train_detm.py \
    --data_path data/pubmed.csv \
    --text_column abstract \
    --time_column year \
    --lemmatize \
    --split_delimiter "" \
    --config_overrides '{"MIN_DF": 20, "MAX_VOCAB_SIZE": 8000}'
```
Lemmatisation collapses morphological variants (`"patients"` → `"patient"`), which tightens the vocabulary for medical text.

---

## Cache Invalidation

The preprocessed pickle (`data/preprocessed_data.pkl`) and Word2Vec model (`data/word2vec.model`) are expensive to recompute.  The pipeline caches them and reuses them on subsequent runs — **automatically invalidating them when the configuration changes**.

### How it works

A SHA-256 fingerprint of all preprocessing-affecting config fields is stored inside the pickle.  On every run, the current fingerprint is compared to the stored one:

```
Config field changes (MIN_DF, stopwords, delimiter, …)
        │
        └── fingerprint changes
                │
                ├── preprocessed_data.pkl  → deleted & re-generated
                ├── word2vec.model         → deleted & re-trained
                ├── word_embeddings.npy    → deleted & re-extracted
                └── preprocessing re-runs from the raw CSV
```

### Fields that affect the fingerprint

`MIN_DF`, `MAX_DF`, `MAX_VOCAB_SIZE`, `MIN_DOC_LENGTH`, `MIN_WORD_LENGTH`, `LEMMATIZE`, `USE_NLTK_STOPWORDS`, `STOPWORDS_FILE`, `EXTRA_STOPWORDS`, `SPLIT_DELIMITER`, `TEXT_COLUMN`, `TIME_COLUMN`, `TRAIN_SPLIT`, `SEED`

### Forcing a retrain manually

```bash
python train_detm.py --data_path data/corpus.csv --force_retrain
```

### Important: vocab change invalidates model checkpoints

If the vocabulary size changes (because preprocessing changed), existing model checkpoints become incompatible.  Clear them before retraining:

```bash
rm models/detm_*.pt
```

---

## Python API

Use the `detm` package directly in notebooks or other scripts:

```python
from detm import (
    DETMConfig,
    DataPreprocessor,
    EmbeddingGenerator,
    create_dataloaders,
    DETM,
    DETMTrainer,
    TopicEvaluator,
)
import torch, pandas as pd

# ── 1. Config ──────────────────────────────────────────────────────────────
cfg = DETMConfig(
    TEXT_COLUMN="transcript",
    TIME_COLUMN="quarter",
    SPLIT_DELIMITER="",           # no splitting
    USE_NLTK_STOPWORDS=False,
    STOPWORDS_FILE="finance_sw.txt",
    NUM_TOPICS=25,
    MIN_DF=5,
)
cfg.make_dirs()

# ── 2. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("data/earnings_calls.csv")

# ── 3. Preprocess (cached on disk; auto-invalidated when config changes) ────
preprocessor = DataPreprocessor(cfg)
processed_data = preprocessor.preprocess_corpus(df)

# ── 4. Word embeddings (Word2Vec; cached after first run) ───────────────────
emb_gen = EmbeddingGenerator(cfg)
embeddings_np = emb_gen.generate_vocabulary_embeddings(
    preprocessor.vocabulary,
    tokens_list=processed_data["tokens_list"],
)
word_embeddings = torch.FloatTensor(embeddings_np)

# ── 5. DataLoaders ──────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = create_dataloaders(processed_data, cfg)

# ── 6. Model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DETM(
    cfg,
    word_embeddings,
    num_time_steps=processed_data["temporal_info"]["num_time_steps"],
    avg_bow_timeline=processed_data["temporal_info"]["avg_bow_per_time"],
    num_train_docs=len(train_loader.dataset),
)
model.idx2word = preprocessor.idx2word
model = model.to(device)

# ── 7. Train ────────────────────────────────────────────────────────────────
trainer = DETMTrainer(model, cfg, train_loader, val_loader, device)
history = trainer.train()

# ── 8. Evaluate ─────────────────────────────────────────────────────────────
evaluator = TopicEvaluator(processed_data["tokens_list"], preprocessor.vocabulary)
metrics = evaluator.evaluate_topics(model, top_n_words=[10, 15, 20])
ppl = TopicEvaluator.compute_perplexity(model, test_loader, device)
print(metrics)
print(f"Perplexity: {ppl:.2f}")

# ── 9. Inspect topics ───────────────────────────────────────────────────────
# Topics at the last time step
for k, topic in enumerate(model.get_topics(time_idx=-1, top_n=10)):
    words = [w for w, _ in topic]
    print(f"Topic {k}: {', '.join(words)}")
```

### Config JSON round-trip (save / load)

```python
cfg.save("outputs/config.json")

cfg2 = DETMConfig.load("outputs/config.json")
```

### Inline config overrides

```python
cfg = DETMConfig.from_dict({
    "NUM_TOPICS": 30,
    "LEARNING_RATE": 0.0001,
    "STOPWORDS_FILE": "my_stopwords.txt",
})
```

### Preprocessing fingerprint (cache key)

```python
print(cfg.preprocessing_fingerprint())  # SHA-256 hex string
# Changes whenever any vocab-affecting field changes
```

---

## Configuration Reference

All fields belong to `DETMConfig` (see [detm/config.py](detm/config.py)).

### Paths

| Field | Default | Description |
|-------|---------|-------------|
| `DATA_DIR` | `data/` | Cache files directory |
| `MODELS_DIR` | `models/` | Checkpoint output directory |
| `OUTPUTS_DIR` | `outputs/` | Results, TensorBoard, topics output |

### Input columns

| Field | Default | Description |
|-------|---------|-------------|
| `TEXT_COLUMN` | `"text"` | CSV column for document text |
| `TIME_COLUMN` | `"year"` | CSV column for temporal key; `""` = static (no time) |

### Document splitting

| Field | Default | Description |
|-------|---------|-------------|
| `SPLIT_DELIMITER` | `".\n"` | Segment delimiter; `""` = no splitting |

### Data split

| Field | Default | Description |
|-------|---------|-------------|
| `SPLIT_STRATEGY` | `"random"` | `"random"` or `"chronological"` |
| `TRAIN_SPLIT` | `0.85` | Fraction of data for training |
| `VAL_SPLIT` | `0.05` | Fraction for validation (remainder → test) |

### Stopwords

| Field | Default | Description |
|-------|---------|-------------|
| `USE_NLTK_STOPWORDS` | `True` | Include NLTK English stopwords |
| `STOPWORDS_FILE` | `""` | Path to plain-text stopwords file |
| `EXTRA_STOPWORDS` | `[]` | Inline list of additional stopwords |

### Text cleaning

| Field | Default | Description |
|-------|---------|-------------|
| `MIN_WORD_LENGTH` | `2` | Minimum token character count (keeps "EU", "UK") |
| `LEMMATIZE` | `False` | WordNet lemmatisation (reduces vocab, broader topics) |

### Vocabulary

| Field | Default | Description |
|-------|---------|-------------|
| `MIN_DF` | `10` | Minimum document frequency (absolute count) |
| `MAX_DF` | `0.7` | Maximum document frequency (fraction) |
| `MAX_VOCAB_SIZE` | `0` | `0` = keep all words passing df gates (alphabetical). `N > 0` = keep IDF-ranked top-N |
| `MIN_DOC_LENGTH` | `5` | Minimum cleaned-token count to keep a document |

### Word2Vec embeddings

| Field | Default | Description |
|-------|---------|-------------|
| `EMBEDDING_DIM` | `300` | Embedding dimensionality |
| `W2V_WINDOW` | `5` | Skip-gram context window |
| `W2V_EPOCHS` | `10` | Training epochs |
| `W2V_WORKERS` | `4` | Parallel workers |
| `TRAIN_WORD_EMBEDDINGS` | `True` | Fine-tune embeddings during DETM training |

### Model architecture

| Field | Default | Description |
|-------|---------|-------------|
| `NUM_TOPICS` | `50` | Number of topics K |
| `DOC_HIDDEN_DIM` | `800` | θ-encoder MLP hidden size |
| `COMPRESSION_DIM` | `200` | η-LSTM input compression dim |
| `LSTM_LAYERS` | `3` | η-LSTM layers |
| `LSTM_HIDDEN` | `200` | η-LSTM hidden size |

### Prior variances (random walk)

| Field | Default | Description |
|-------|---------|-------------|
| `ETA_PRIOR_VARIANCE` | `0.005` | δ²: step variance for η_t random walk |
| `ALPHA_PRIOR_VARIANCE` | `0.005` | γ²: step variance for α_t random walk |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `BATCH_SIZE` | `700` | Documents per batch |
| `NUM_EPOCHS` | `1000` | Maximum training epochs |
| `LEARNING_RATE` | `0.0001` | Adam initial learning rate |
| `WEIGHT_DECAY` | `1.2e-6` | L2 regularisation |
| `CLIP_GRAD` | `0.0` | Gradient clip norm; `0.0` = disabled |
| `LR_ANNEAL_PATIENCE` | `10` | Plateau epochs before LR ÷ 4 |

### Evaluation

| Field | Default | Description |
|-------|---------|-------------|
| `TOP_N_WORDS` | `[10, 15, 20]` | Top-N word counts for coherence/diversity |
| `COHERENCE_METRICS` | `["c_v", "c_npmi"]` | Gensim coherence measures |

### Checkpointing

| Field | Default | Description |
|-------|---------|-------------|
| `SAVE_EVERY` | `10` | Save periodic checkpoint every N epochs |
| `SEED` | `42` | Global random seed |

---

## Model Architecture

### Generative Process

```
For each time step t = 1 … T:
    α_t ~ N(α_{t-1}, γ²I)          # topic embeddings evolve (random walk)
    η_t ~ N(η_{t-1}, δ²I)          # global topic baseline evolves

For each document d at time t:
    θ_d ~ LogisticNormal(η_t, I)   # document topic proportions
    β_t = softmax(α_t · ρᵀ)        # topic-word distributions (V×K)
    For each token w in d:
        w ~ Categorical(θ_d · β_t) # word generated from mixture
```

### Variational Inference

| Component | Network | Output |
|-----------|---------|--------|
| `TemporalBaselineEncoder` | LSTM over mean BoW timeline | η_t, KL_η |
| `DocumentTopicEncoder` | MLP on (BoW‖η_t) | θ_d, KL_θ |
| mean-field α | learned parameters | α_t, KL_α |

### Loss (corpus-scale ELBO)

```
L = Σ_d E_q[log p(w_d | θ_d, α_{t_d}, ρ)]
    − KL(q(θ) ‖ p(θ))
    − KL(q(η) ‖ p(η))
    − KL(q(α) ‖ p(α))
```

All four terms are placed on the same corpus-scale (`O(D)`) magnitude so their gradients contribute proportionally.

---

## Evaluation Metrics

| Metric | Range | Better |
|--------|-------|--------|
| **Coherence C_v** | 0 – 1 | ↑ higher |
| **Coherence NPMI** | −1 – 1 | ↑ higher (typical range ~0.0 – 0.2) |
| **Topic Diversity** | 0 – 1 | ↑ higher (1 = all words unique across topics) |
| **Perplexity** | > 1 | ↓ lower |

Coherence is averaged over 5 evenly-spaced time steps (t=0, T/4, T/2, 3T/4, T−1) for a robust estimate across the temporal range.

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/results.json` | Coherence, diversity, perplexity metrics |
| `outputs/config.json` | Full config used for the run (for reproducibility) |
| `outputs/topics_evolution.txt` | Top-10 words per topic at each decade |
| `outputs/tensorboard_logs/` | TensorBoard event files (loss, KL terms, LR, grad norm, α diagnostics) |
| `models/detm_best.pt` | Best checkpoint (by validation loss) |
| `models/detm_epoch_N.pt` | Periodic checkpoints every `SAVE_EVERY` epochs |
| `data/preprocessed_data.pkl` | Cached preprocessed corpus (auto-invalidated on config change) |
| `data/word2vec.model` | Cached Word2Vec model |
| `data/word_embeddings.npy` | Cached embedding matrix (vocab_size × EMBEDDING_DIM) |

### TensorBoard

```bash
tensorboard --logdir outputs/tensorboard_logs
```

Logged scalars: `Batch/Loss`, `Batch/Reconstruction`, `Batch/KL_theta`, `Batch/KL_eta`, `Batch/KL_alpha`, `Batch/Grad_Norm`, `Epoch/Train/*`, `Epoch/Val/*`, `Epoch/Learning_Rate`, `Epoch/Gradient_Norm`, `Epoch/Alpha/PosteriorVar_mean`, `Epoch/Alpha/Logvar_mean`.

---

## Reference

> Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2019).  
> **Topic Modeling in Embedding Spaces.**  
> *arXiv:1907.05545*
