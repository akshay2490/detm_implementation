"""
detm/train.py
-------------
Training infrastructure for the Dynamic Embedded Topic Model.

Public API
----------
    DETMTrainer — manages the full train/validate/checkpoint/early-stop loop

Usage
-----
    from detm.train import DETMTrainer
    trainer = DETMTrainer(model, config, train_loader, val_loader, device)
    history = trainer.train()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from detm.config import DETMConfig
from detm.model import DETM


class DETMTrainer:
    """
    Training loop for DETM.

    Features
    --------
    - Adam optimiser with constant LR and L2 weight decay (matches Dieng et al. 2019)
    - Gradient clipping
    - TensorBoard logging (batch and epoch metrics); each run gets a timestamped subdir
    - Periodic checkpointing + best-model checkpoint
    - Early stopping on validation loss

    Parameters
    ----------
    model        : DETM instance (will be moved to ``device``)
    config       : DETMConfig
    train_loader : training DataLoader
    val_loader   : validation DataLoader
    device       : torch.device
    log_dir      : override TensorBoard log directory (default: auto-timestamped)
    """

    def __init__(
        self,
        model: DETM,
        config: DETMConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        log_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # TensorBoard — timestamped subdir so multiple runs overlay cleanly
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_dir is None:
            log_dir = str(config.OUTPUTS_DIR / "tensorboard_logs" / ts)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard: {log_dir}")
        print(f"  → tensorboard --logdir={config.OUTPUTS_DIR / 'tensorboard_logs'}")

        # Adam with L2 weight-decay — NOT AdamW (decoupled decay differs from the paper)
        self.optimizer = Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_recon": [],
            "train_kl_theta": [], "train_kl_eta": [], "train_kl_alpha": [],
            "val_loss": [], "val_recon": [],
            "val_kl_theta": [], "val_kl_eta": [], "val_kl_alpha": [],
            "learning_rate": [],
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Run the full training loop.

        Parameters
        ----------
        num_epochs : override config.NUM_EPOCHS

        Returns
        -------
        history dict (same as self.history)
        """
        epochs = num_epochs or self.config.NUM_EPOCHS
        lr = self.config.LEARNING_RATE

        print("\n" + "=" * 60)
        print("TRAINING DETM")
        print("=" * 60)
        print(f"  Epochs      : {epochs}")
        print(f"  LR          : {lr}  (constant)")
        print(f"  Weight decay: {self.config.WEIGHT_DECAY}")
        print(f"  Grad clip   : {self.config.CLIP_GRAD}")
        print(f"  Device      : {self.device}")
        print(f"  Patience    : {self.config.PATIENCE}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            train_m = self._train_epoch()
            val_m = self._validate()

            # History
            self.history["train_loss"].append(train_m["train_loss"])
            self.history["train_recon"].append(train_m["train_recon"])
            self.history["train_kl_theta"].append(train_m["train_kl_theta"])
            self.history["train_kl_eta"].append(train_m["train_kl_eta"])
            self.history["train_kl_alpha"].append(train_m["train_kl_alpha"])
            self.history["val_loss"].append(val_m["val_loss"])
            self.history["val_recon"].append(val_m["val_recon"])
            self.history["val_kl_theta"].append(val_m["val_kl_theta"])
            self.history["val_kl_eta"].append(val_m["val_kl_eta"])
            self.history["val_kl_alpha"].append(val_m["val_kl_alpha"])
            self.history["learning_rate"].append(lr)

            # TensorBoard
            self._log_epoch(epoch, train_m, val_m, lr)

            # Console summary
            print(
                f"\nEpoch {epoch}/{epochs} — "
                f"train={train_m['train_loss']:.4f} "
                f"(recon={train_m['train_recon']:.4f}, "
                f"KL_θ={train_m['train_kl_theta']:.4f}, "
                f"KL_η={train_m['train_kl_eta']:.4f}, "
                f"KL_α={train_m['train_kl_alpha']:.4f})  |  "
                f"val={val_m['val_loss']:.4f}"
            )

            # Checkpoint / early stop
            is_best = val_m["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["val_loss"]
                self.patience_counter = 0
                print(f"  ✓ New best val_loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config.PATIENCE})")

            self._save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        self.writer.close()
        print(f"\n{'='*60}\nTraining complete. Best val_loss: {self.best_val_loss:.4f}\n{'='*60}")
        return self.history

    # ------------------------------------------------------------------
    # Private — train / validate
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {k: 0.0 for k in ("loss", "recon", "kl_theta", "kl_eta", "kl_alpha")}
        n = 0

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            bow = batch["bow"].to(self.device)
            t_idx = batch["time_idx"].to(self.device)

            out = self.model(bow, t_idx, compute_loss=True)

            self.optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRAD)
            self.optimizer.step()

            for k in totals:
                totals[k] += out[{"loss": "loss", "recon": "recon_loss",
                                   "kl_theta": "kl_theta", "kl_eta": "kl_eta",
                                   "kl_alpha": "kl_alpha"}[k]].item()

            if self.global_step % 10 == 0:
                self._log_batch(out)
            self.global_step += 1
            n += 1

        return {f"train_{k}": v / n for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        totals = {k: 0.0 for k in ("loss", "recon", "kl_theta", "kl_eta", "kl_alpha")}
        n = 0

        for batch in tqdm(self.val_loader, desc="Val  ", leave=False):
            bow = batch["bow"].to(self.device)
            t_idx = batch["time_idx"].to(self.device)
            out = self.model(bow, t_idx, compute_loss=True)
            # Unscaled loss — no KL annealing / weighting during validation
            # so the metric reflects the true ELBO components.
            totals["recon"] += out["recon_loss"].item()
            totals["kl_theta"] += out["kl_theta"].item()
            totals["kl_eta"] += out["kl_eta"].item()
            totals["kl_alpha"] += out["kl_alpha"].item()
            totals["loss"] += (
                out["recon_loss"] + out["kl_theta"] + out["kl_eta"] + out["kl_alpha"]
            ).item()
            n += 1

        return {f"val_{k}": v / n for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Private — checkpointing & logging
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config.to_dict(),
        }
        if epoch % self.config.SAVE_EVERY == 0:
            path = self.config.MODELS_DIR / f"detm_epoch_{epoch}.pt"
            torch.save(ckpt, path)
            print(f"  Checkpoint → {path}")
        if is_best:
            path = self.config.MODELS_DIR / "detm_best.pt"
            torch.save(ckpt, path)
            print(f"  Best model → {path}")

    def _log_batch(self, out: Dict) -> None:
        s = self.global_step
        self.writer.add_scalar("Batch/Loss", out["loss"].item(), s)
        self.writer.add_scalar("Batch/Reconstruction", out["recon_loss"].item(), s)
        self.writer.add_scalar("Batch/KL_theta", out["kl_theta"].item(), s)
        self.writer.add_scalar("Batch/KL_eta", out["kl_eta"].item(), s)
        self.writer.add_scalar("Batch/KL_alpha", out["kl_alpha"].item(), s)

    def _log_epoch(
        self,
        epoch: int,
        train_m: Dict,
        val_m: Dict,
        lr: float,
    ) -> None:
        w = self.writer
        for tag, val in [
            ("Epoch/Train/Loss", train_m["train_loss"]),
            ("Epoch/Train/Reconstruction", train_m["train_recon"]),
            ("Epoch/Train/KL_theta", train_m["train_kl_theta"]),
            ("Epoch/Train/KL_eta", train_m["train_kl_eta"]),
            ("Epoch/Train/KL_alpha", train_m["train_kl_alpha"]),
            ("Epoch/Val/Loss", val_m["val_loss"]),
            ("Epoch/Val/Reconstruction", val_m["val_recon"]),
            ("Epoch/Val/KL_theta", val_m["val_kl_theta"]),
            ("Epoch/Val/KL_eta", val_m["val_kl_eta"]),
            ("Epoch/Val/KL_alpha", val_m["val_kl_alpha"]),
            ("Epoch/Learning_Rate", lr),
        ]:
            w.add_scalar(tag, val, epoch)

        # Gradient norm
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5
        w.add_scalar("Epoch/Gradient_Norm", total_norm, epoch)
