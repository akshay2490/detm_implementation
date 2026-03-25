#!/usr/bin/env python3
"""
train_detm.py
=============
Domain-agnostic command-line entry point for training a DETM model end-to-end.

Works for any domain (UN debates, finance news, medical literature, etc.) by
routing all domain-specific settings through CLI flags that map to DETMConfig.

Quick start
-----------
    # UN debates — default settings (paragraph split, NLTK stopwords)
    python train_detm.py --data_path data/un-general-debates.csv

    # Finance — custom columns, no paragraph splitting, domain stopwords
    python train_detm.py \\
        --data_path data/earnings_calls.csv \\
        --text_column transcript \\
        --time_column quarter \\
        --split_delimiter "" \\
        --stopwords_file finance_stopwords.txt \\
        --no_nltk_stopwords

    # Resume from a checkpoint
    python train_detm.py --data_path data/corpus.csv \\
        --checkpoint models/detm_best.pt

    # Use a saved config + override a few hyperparameters inline
    python train_detm.py --data_path data/corpus.csv \\
        --config_path outputs/config.json \\
        --config_overrides '{"NUM_TOPICS": 30, "LEARNING_RATE": 0.0001}'

    # Force retrain preprocessing even if cache exists
    python train_detm.py --data_path data/corpus.csv --force_retrain

Cache invalidation
------------------
The preprocessed data pickle contains a SHA-256 fingerprint of all
vocab-affecting config fields.  If the fingerprint changes from a previous
run, the stale pickles (preprocessed_data.pkl, word2vec.model,
word_embeddings.npy) are deleted automatically and preprocessing re-runs.
This ensures vocab and embeddings are always consistent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Dynamic Embedded Topic Model (DETM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument(
        "--data_path", required=True,
        help="Path to the input CSV file.",
    )
    data.add_argument(
        "--text_column", default=None,
        help="CSV column containing document text. Overrides config.TEXT_COLUMN.",
    )
    data.add_argument(
        "--time_column", default=None,
        help="CSV column containing the temporal key (e.g. year, quarter). "
             "Set to '' to disable temporal modelling. Overrides config.TIME_COLUMN.",
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    pre = p.add_argument_group("Preprocessing")
    pre.add_argument(
        "--split_delimiter", default=None,
        help="String on which to split each document into segments before "
             "processing (e.g. '.\\n' for paragraph split, '' for none). "
             "Overrides config.SPLIT_DELIMITER.",
    )
    pre.add_argument(
        "--split_strategy", choices=["random", "chronological"], default=None,
        help="How to split documents into train/val/test. "
             "'random' shuffles before splitting (avoids temporal bias); "
             "'chronological' uses time-sorted order. Overrides config.SPLIT_STRATEGY.",
    )
    pre.add_argument(
        "--lemmatize", action="store_true", default=None,
        help="Enable WordNet lemmatisation during tokenisation.",
    )
    pre.add_argument(
        "--force_retrain", action="store_true",
        help="Ignore cached preprocessed data and word2vec; rerun from scratch.",
    )

    # ── Stopwords ─────────────────────────────────────────────────────────────
    sw = p.add_argument_group("Stopwords")
    sw.add_argument(
        "--stopwords_file", default=None,
        help="Path to a plain-text file with one stopword per line. "
             "Overrides config.STOPWORDS_FILE.",
    )
    sw.add_argument(
        "--extra_stopwords", nargs="*", default=None,
        help="Additional stopwords as space-separated tokens. "
             "Overrides config.EXTRA_STOPWORDS.",
    )
    sw.add_argument(
        "--no_nltk_stopwords", action="store_true",
        help="Disable NLTK English stopwords. "
             "Useful for non-English corpora or when you supply your own list.",
    )

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = p.add_argument_group("Config")
    cfg.add_argument(
        "--config_path", default=None,
        help="Path to a saved config JSON (overrides all defaults).",
    )
    cfg.add_argument(
        "--config_overrides", default=None,
        help="JSON string of config overrides, e.g. '{\"NUM_TOPICS\": 30}'.",
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt = p.add_argument_group("Checkpointing")
    ckpt.add_argument(
        "--checkpoint", default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )

    # ── Training control ──────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=None,
                       help="Override config.NUM_EPOCHS.")
    train.add_argument("--device", default="auto",
                       help="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--results_path", default=None,
                     help="Where to write results JSON (default: outputs/results.json).")
    out.add_argument("--skip_eval", action="store_true",
                     help="Skip evaluation after training.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    # ── Deferred imports ──────────────────────────────────────────────────────
    import pandas as pd
    from detm import (
        DETM,
        DETMConfig,
        DETMTrainer,
        DataPreprocessor,
        EmbeddingGenerator,
        TopicEvaluator,
        create_dataloaders,
    )

    # ── Config assembly ───────────────────────────────────────────────────────
    if args.config_path:
        config = DETMConfig.load(args.config_path)
        print(f"Config loaded from {args.config_path}.")
    else:
        config = DETMConfig()
        print("Using default config.")

    if args.config_overrides:
        overrides: dict = json.loads(args.config_overrides)
        for k, v in overrides.items():
            if not hasattr(config, k):
                print(f"WARNING: unknown config key '{k}' — skipped.", file=sys.stderr)
                continue
            setattr(config, k, v)
        print(f"Applied {len(overrides)} config override(s).")

    # ── CLI flag → config field propagation ───────────────────────────────────
    if args.text_column is not None:
        config.TEXT_COLUMN = args.text_column
    if args.time_column is not None:
        config.TIME_COLUMN = args.time_column
    if args.split_delimiter is not None:
        config.SPLIT_DELIMITER = args.split_delimiter
    if args.split_strategy is not None:
        config.SPLIT_STRATEGY = args.split_strategy
    if args.lemmatize:
        config.LEMMATIZE = True
    if args.stopwords_file is not None:
        config.STOPWORDS_FILE = args.stopwords_file
    if args.extra_stopwords is not None:
        config.EXTRA_STOPWORDS = args.extra_stopwords
    if args.no_nltk_stopwords:
        config.USE_NLTK_STOPWORDS = False
    if args.seed != 42:
        config.SEED = args.seed

    config.make_dirs()
    device = pick_device(args.device)
    print(f"Device: {device}")

    # Preview active stopword config
    print(f"\nStopword sources:"
          f" NLTK={config.USE_NLTK_STOPWORDS},"
          f" file={repr(config.STOPWORDS_FILE) if config.STOPWORDS_FILE else 'none'},"
          f" extra={len(config.EXTRA_STOPWORDS)} words")
    print(f"Text column: '{config.TEXT_COLUMN}'   Time column: '{config.TIME_COLUMN}'")
    print(f"Split delimiter: {repr(config.SPLIT_DELIMITER)}   Strategy: {config.SPLIT_STRATEGY}")

    # ── Load raw data ─────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    print(f"\n[1/6] Loading data from {data_path} …")
    df = pd.read_csv(data_path)

    # Validate required columns
    if config.TEXT_COLUMN not in df.columns:
        sys.exit(
            f"ERROR: text column '{config.TEXT_COLUMN}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}. "
            f"Use --text_column to override."
        )
    if config.TIME_COLUMN and config.TIME_COLUMN not in df.columns:
        print(
            f"WARNING: time column '{config.TIME_COLUMN}' not found — "
            "disabling temporal modelling (single time step).",
            file=sys.stderr,
        )
        config.TIME_COLUMN = ""

    n_raw = len(df)
    time_info = ""
    if config.TIME_COLUMN and config.TIME_COLUMN in df.columns:
        time_info = (
            f", {config.TIME_COLUMN} range "
            f"{df[config.TIME_COLUMN].min()}–{df[config.TIME_COLUMN].max()}"
        )
    print(f"  {n_raw:,} rows{time_info}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing corpus …")
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess_corpus(df, force_retrain=args.force_retrain)
    vocab = preprocessor.vocabulary

    # ── Word embeddings ───────────────────────────────────────────────────────
    print("\n[3/6] Generating word embeddings …")
    emb_gen = EmbeddingGenerator(config)
    embeddings_np = emb_gen.generate_vocabulary_embeddings(
        vocab,
        tokens_list=processed_data["tokens_list"],
        force_retrain=args.force_retrain,
    )
    word_embeddings = torch.FloatTensor(embeddings_np)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n[4/6] Creating data loaders …")
    train_loader, val_loader, test_loader = create_dataloaders(processed_data, config)
    print(
        f"  Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[5/6] Building model …")
    num_train_docs = len(train_loader.dataset)  # type: ignore[arg-type]
    model = DETM(
        config,
        word_embeddings,
        num_time_steps=processed_data["temporal_info"]["num_time_steps"],
        avg_bow_timeline=processed_data["temporal_info"]["avg_bow_per_time"],
        num_train_docs=num_train_docs,
    )
    model.idx2word = preprocessor.idx2word
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {trainable_p:,}")
    print(f"  num_train_docs  : {num_train_docs:,}")

    # Optional checkpoint resume
    start_epoch = 0
    checkpoint_data = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found at {ckpt_path} — starting fresh.", file=sys.stderr)
        else:
            checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            start_epoch = checkpoint_data.get("epoch", 0)
            print(
                f"  Resumed from epoch {start_epoch} "
                f"(val_loss={checkpoint_data.get('best_val_loss', float('nan')):.4e})"
            )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[6/6] Training …")
    trainer = DETMTrainer(model, config, train_loader, val_loader, device)

    if checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data:
            trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if isinstance(checkpoint_data.get("history"), dict):
            trainer.history = checkpoint_data["history"]
        trainer.best_val_loss = checkpoint_data.get("best_val_loss", float("inf"))
        trainer.plateau_counter = checkpoint_data.get("plateau_counter", 0)
        trainer.global_step = checkpoint_data.get(
            "global_step", start_epoch * len(train_loader)
        )
        print(f"  Trainer state restored:")
        print(f"    best_val_loss   : {trainer.best_val_loss:.4e}")
        print(f"    plateau_counter : {trainer.plateau_counter}")
        print(f"    global_step     : {trainer.global_step}")

    history = trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results: dict = {"history": history}

    if not args.skip_eval:
        print("\n── Evaluation ──────────────────────────────────────────────────")

        # Reload best weights before evaluating
        best_ckpt_path = config.MODELS_DIR / "detm_best.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(
                f"Loaded best checkpoint (epoch={best_ckpt.get('epoch', '?')}, "
                f"val_loss={best_ckpt.get('best_val_loss', float('nan')):.4e})"
            )

        evaluator = TopicEvaluator(processed_data["tokens_list"], vocab)
        metrics = evaluator.evaluate_topics(model, top_n_words=config.TOP_N_WORDS)
        perplexity = TopicEvaluator.compute_perplexity(model, test_loader, device)

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Test Perplexity: {perplexity:.2f}")

        results["metrics"] = metrics
        results["test_perplexity"] = perplexity

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = (
        Path(args.results_path) if args.results_path else config.OUTPUTS_DIR / "results.json"
    )
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults saved → {results_path}")

    config_out = results_path.parent / "config.json"
    config.save(config_out)
    print(f"Config saved   → {config_out}")


if __name__ == "__main__":
    main()
