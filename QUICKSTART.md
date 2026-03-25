# DETM Quick Start

Get from zero to a trained model in four commands.

## Prerequisites

```bash
source detm_env/bin/activate
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"
```

## UN General Debates (default domain)

```bash
# 1. Download dataset (requires Kaggle credentials)
kaggle datasets download -d unitednations/un-general-debates -p data --unzip

# 2. Train (paragraph split + NLTK stopwords + stop_words.txt)
python train_detm.py \
    --data_path data/un-general-debates.csv \
    --stopwords_file stop_words.txt

# Results in outputs/results.json · models/detm_best.pt
```

## Any Other Domain

```bash
# Minimum: just supply a CSV with text + time columns
python train_detm.py \
    --data_path data/my_corpus.csv \
    --text_column body \
    --time_column date

# Finance — disable paragraph split, use domain stopwords only
python train_detm.py \
    --data_path data/earnings.csv \
    --text_column transcript \
    --time_column quarter \
    --split_delimiter "" \
    --no_nltk_stopwords \
    --stopwords_file finance_sw.txt

# No temporal structure (static ETM)
python train_detm.py \
    --data_path data/articles.csv \
    --time_column ""
```

## Resume / Reload

```bash
# Resume training from the best checkpoint
python train_detm.py \
    --data_path data/my_corpus.csv \
    --checkpoint models/detm_best.pt

# Evaluate only (skip training)
python train_detm.py \
    --data_path data/my_corpus.csv \
    --checkpoint models/detm_best.pt \
    --epochs 0
```

## Override Hyperparameters

```bash
# Inline JSON overrides
python train_detm.py \
    --data_path data/my_corpus.csv \
    --config_overrides '{"NUM_TOPICS": 30, "BATCH_SIZE": 500, "MIN_DF": 5}'

# Load a full saved config (e.g. from a previous run)
python train_detm.py \
    --data_path data/my_corpus.csv \
    --config_path outputs/config.json
```

## Monitor Training

```bash
tensorboard --logdir outputs/tensorboard_logs
```

## Key Output Files

| File | What it contains |
|------|-----------------|
| `outputs/results.json` | coherence, diversity, perplexity |
| `outputs/config.json` | full config snapshot for reproducibility |
| `outputs/topics_evolution.txt` | top words per topic at each time step |
| `models/detm_best.pt` | best checkpoint (lowest validation loss) |

---

See [README.md](README.md) for the full CLI reference and Python API.
