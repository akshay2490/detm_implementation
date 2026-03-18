#!/usr/bin/env python3
"""
train_detm.py
=============
Command-line entry point for training a DETM model end-to-end.

Quick start
-----------
    # Full pipeline — preprocess + embed + train + evaluate
    python train_detm.py --data_path data/un-general-debates.csv

    # Resume from a checkpoint
    python train_detm.py --data_path data/un-general-debates.csv \\
        --checkpoint models/detm_best.pt

    # Use a saved config
    python train_detm.py --data_path data/un-general-debates.csv \\
        --config_path outputs/config.json

    # Override a few hyperparameters inline (JSON dict)
    python train_detm.py --data_path data/un-general-debates.csv \\
        --config_overrides '{"NUM_TOPICS": 30, "LEARNING_RATE": 0.003}'
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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

    # --- Data ---
    p.add_argument(
        "--data_path",
        required=True,
        help="Path to the input CSV file (must have 'text' and 'year' columns).",
    )

    # --- Config ---
    p.add_argument(
        "--config_path",
        default=None,
        help="Path to a saved config JSON (overrides all defaults).",
    )
    p.add_argument(
        "--config_overrides",
        default=None,
        help='JSON string of config overrides, e.g. \'{"NUM_TOPICS": 30}\'.',
    )

    # --- Checkpointing ---
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )

    # --- Training control ---
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config.NUM_EPOCHS.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # --- Output ---
    p.add_argument(
        "--results_path",
        default=None,
        help="Where to write results JSON (default: outputs/results.json from config).",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    # ── Imports (deferred so --help is fast) ──────────────────────────────────
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

    # ── Config ────────────────────────────────────────────────────────────────
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

    config.make_dirs()

    device = pick_device(args.device)
    print(f"Device: {device}")

    # ── Load raw data ─────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    print(f"\n[1/6] Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    required_cols = {"text", "year"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}")
    print(f"  {len(df):,} documents, years {df['year'].min()}–{df['year'].max()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing corpus …")
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess_corpus(df)
    vocab_size = len(preprocessor.vocabulary)
    num_time_steps = len(preprocessor.time_steps)
    print(f"  Vocabulary size : {vocab_size:,}")
    print(f"  Time steps      : {num_time_steps}")
    print(f"  Documents kept  : {len(processed_data['bow_list']):,}")

    # ── Word embeddings ───────────────────────────────────────────────────────
    print("\n[3/6] Generating Word2Vec embeddings …")
    emb_gen = EmbeddingGenerator(config)
    embeddings_np = emb_gen.generate_vocabulary_embeddings(preprocessor.vocabulary)
    word_embeddings = torch.FloatTensor(embeddings_np)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n[4/6] Creating data loaders …")
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data, batch_size=config.BATCH_SIZE
    )
    print(
        f"  Train batches: {len(train_loader)},"
        f" Val batches: {len(val_loader)},"
        f" Test batches: {len(test_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[5/6] Building model …")
    # Inject runtime sizes not known at config-definition time
    config.VOCAB_SIZE = vocab_size
    config.NUM_TIME_STEPS = num_time_steps

    num_train_docs = len(
        [b for b in train_loader.dataset]  # type: ignore[arg-type]
    )
    model = DETM(config, word_embeddings, num_train_docs=num_train_docs)
    model.idx2word = preprocessor.idx2word
    model.avg_bow_timeline = processed_data["avg_bow_timeline"].to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Optionally resume from checkpoint
    start_epoch = 0
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found at {ckpt_path} — starting fresh.", file=sys.stderr)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"  Resumed from epoch {start_epoch} (val_loss={ckpt.get('best_val_loss', float('nan')):.4f})")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[6/6] Training …")
    trainer = DETMTrainer(model, config, train_loader, val_loader, device)

    # If resuming, fast-forward the optimizer state too
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    num_epochs = args.epochs  # None → trainer uses config.NUM_EPOCHS
    history = trainer.train(num_epochs=num_epochs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results: dict = {"history": history}

    if not args.skip_eval:
        print("\n── Evaluation ──────────────────────────────────────────")

        # Reload best checkpoint before evaluating
        best_ckpt_path = config.MODELS_DIR / "detm_best.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(
                f"Loaded best checkpoint (epoch={best_ckpt.get('epoch','?')},"
                f" val_loss={best_ckpt.get('best_val_loss', float('nan')):.4f})"
            )

        model.to(device)

        evaluator = TopicEvaluator(processed_data["tokens_list"], preprocessor.vocabulary)
        metrics = evaluator.evaluate_topics(model, top_n_words=config.TOP_N_WORDS)
        perplexity = TopicEvaluator.compute_perplexity(model, test_loader, device)

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Test Perplexity: {perplexity:.4f}")

        results["metrics"] = metrics
        results["test_perplexity"] = perplexity

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = Path(args.results_path) if args.results_path else config.OUTPUTS_DIR / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults saved → {results_path}")

    # Save config alongside results
    config_out = results_path.parent / "config.json"
    config.save(config_out)
    print(f"Config saved   → {config_out}")


if __name__ == "__main__":
    main()
