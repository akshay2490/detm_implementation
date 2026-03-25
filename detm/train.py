"""
detm/train.py
-------------
Training infrastructure for the Dynamic Embedded Topic Model.

Public API
----------
    DETMTrainer — manages the full train/validate/checkpoint loop

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
    - Adam optimiser with plateau-based LR annealing (÷4) and L2 weight decay
    - NaN/Inf batch skipping to prevent model corruption
    - Gradient clipping (disabled when ``config.CLIP_GRAD == 0``)
    - TensorBoard logging (batch loss/components/grad-norm; epoch metrics; alpha diagnostics)
    - Periodic checkpointing + best-model checkpoint (by validation loss)

    Parameters
    ----------
    model        : DETM instance
    config       : DETMConfig
    train_loader : training DataLoader
    val_loader   : validation DataLoader
    device       : torch.device
    log_dir      : override TensorBoard log directory (auto-timestamped by default)
    """

    def __init__(
        self,
        model: DETM,
        config: DETMConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        log_dir: Optional[str] = None,
    ) -> None:
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

        # Adam with L2 weight-decay
        self.optimizer = Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.plateau_counter = 0
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

    def train(
        self,
        num_epochs: Optional[int] = None,
        start_epoch: int = 0,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Parameters
        ----------
        num_epochs  : override config.NUM_EPOCHS (None = use config)
        start_epoch : last completed epoch when resuming from checkpoint

        Returns
        -------
        history dict (same reference as self.history)
        """
        epochs = num_epochs or self.config.NUM_EPOCHS
        lr = self.optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 60)
        print("TRAINING DETM")
        print("=" * 60)
        print(f"  Epochs       : {epochs}")
        print(f"  Start epoch  : {start_epoch + 1}")
        print(f"  LR           : {lr}")
        print(f"  Weight decay : {self.config.WEIGHT_DECAY}")
        print(f"  Grad clip    : {self.config.CLIP_GRAD} "
              f"({'disabled' if self.config.CLIP_GRAD == 0 else 'enabled'})")
        print(f"  Device       : {self.device}")
        print(f"  Anneal every : {self.config.LR_ANNEAL_PATIENCE} plateau epoch(s)")
        if start_epoch > 0:
            print(f"  Resuming from completed epoch : {start_epoch}")
            print(f"  Best val_loss so far          : {self.best_val_loss:.4f}")
            print(f"  Plateau counter               : {self.plateau_counter}")
        print("=" * 60)

        if start_epoch >= epochs:
            self.writer.close()
            print("No training run: checkpoint already reached the requested epoch count.")
            return self.history

        for epoch in range(start_epoch + 1, epochs + 1):
            train_m = self._train_epoch()
            val_m = self._validate()

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

            self._log_epoch(epoch, train_m, val_m, lr)

            print(
                f"\nEpoch {epoch}/{epochs} — "
                f"train={train_m['train_loss']:.4e} "
                f"(recon={train_m['train_recon']:.4e}, "
                f"KL_θ={train_m['train_kl_theta']:.4e}, "
                f"KL_η={train_m['train_kl_eta']:.4e}, "
                f"KL_α={train_m['train_kl_alpha']:.4e})  |  "
                f"val={val_m['val_loss']:.4e}"
            )

            is_best = val_m["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["val_loss"]
                self.plateau_counter = 0
                print(f"  ✓ New best val_loss: {self.best_val_loss:.4e}")
            else:
                self.plateau_counter += 1
                print(
                    f"  No improvement ({self.plateau_counter}/"
                    f"{self.config.LR_ANNEAL_PATIENCE})"
                )
                if self.plateau_counter % self.config.LR_ANNEAL_PATIENCE == 0:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] /= 4.0
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  LR annealed → {lr:.2e}")

            self._save_checkpoint(epoch, is_best)

        self.writer.close()
        print(
            f"\n{'='*60}\n"
            f"Training complete. Best val_loss: {self.best_val_loss:.4e}\n"
            f"{'='*60}"
        )
        return self.history

    # ------------------------------------------------------------------
    # Private — train / validate
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {k: 0.0 for k in ("loss", "recon", "kl_theta", "kl_eta", "kl_alpha")}
        n_ok = 0
        n_nan = 0

        key_map = {
            "loss": "loss", "recon": "recon_loss",
            "kl_theta": "kl_theta", "kl_eta": "kl_eta", "kl_alpha": "kl_alpha",
        }

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            bow = batch["bow"].to(self.device)
            t_idx = batch["time_idx"].to(self.device)

            out = self.model(bow, t_idx, compute_loss=True)
            loss = out["loss"]

            # ── NaN/Inf guard ─────────────────────────────────────────────────
            if not torch.isfinite(loss):
                n_nan += 1
                self.optimizer.zero_grad()
                if n_nan <= 3:
                    print(f"\n  ⚠ NaN/Inf loss in batch — skipping (total: {n_nan})")
                self.global_step += 1
                continue

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (or just norm-logging when 0 = disabled)
            clip = self.config.CLIP_GRAD if self.config.CLIP_GRAD > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            if not torch.isfinite(grad_norm):
                n_nan += 1
                self.optimizer.zero_grad()
                if n_nan <= 3:
                    print(f"\n  ⚠ NaN gradient norm — skipping update (total: {n_nan})")
                self.global_step += 1
                continue

            self.optimizer.step()

            for k, out_k in key_map.items():
                totals[k] += out[out_k].item()
            n_ok += 1

            if self.global_step % 10 == 0:
                self._log_batch(out, grad_norm)
            self.global_step += 1

            pbar.set_postfix({
                "loss":   f"{loss.item():.3e}",
                "recon":  f"{out['recon_loss'].item():.3e}",
                "KL_θ":   f"{out['kl_theta'].item():.3e}",
                "skip":   n_nan,
            })

        if n_nan > 0:
            print(f"  ⚠ Epoch had {n_nan} skipped NaN/Inf batches")
        if n_ok == 0:
            return {f"train_{k}": float("nan") for k in totals}

        return {f"train_{k}": v / n_ok for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        totals = {k: 0.0 for k in ("loss", "recon", "kl_theta", "kl_eta", "kl_alpha")}
        n = 0

        for batch in tqdm(self.val_loader, desc="Val  ", leave=False):
            bow = batch["bow"].to(self.device)
            t_idx = batch["time_idx"].to(self.device)
            out = self.model(bow, t_idx, compute_loss=True)
            if not torch.isfinite(out["loss"]):
                continue
            totals["recon"] += out["recon_loss"].item()
            totals["kl_theta"] += out["kl_theta"].item()
            totals["kl_eta"] += out["kl_eta"].item()
            totals["kl_alpha"] += out["kl_alpha"].item()
            totals["loss"] += (
                out["recon_loss"] + out["kl_theta"] + out["kl_eta"] + out["kl_alpha"]
            ).item()
            n += 1

        if n == 0:
            return {f"val_{k}": float("nan") for k in totals}
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
            "plateau_counter": self.plateau_counter,
            "global_step": self.global_step,
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

    def _log_batch(self, out: Dict, grad_norm: torch.Tensor) -> None:
        s = self.global_step
        self.writer.add_scalar("Batch/Loss", out["loss"].item(), s)
        self.writer.add_scalar("Batch/Reconstruction", out["recon_loss"].item(), s)
        self.writer.add_scalar("Batch/KL_theta", out["kl_theta"].item(), s)
        self.writer.add_scalar("Batch/KL_eta", out["kl_eta"].item(), s)
        self.writer.add_scalar("Batch/KL_alpha", out["kl_alpha"].item(), s)
        self.writer.add_scalar("Batch/Grad_Norm", grad_norm.item(), s)

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

        # Gradient norm (epoch-end snapshot)
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5
        w.add_scalar("Epoch/Gradient_Norm", total_norm, epoch)

        # α posterior diagnostics
        with torch.no_grad():
            alpha_var = self.model.alpha_logvar.exp()
            w.add_scalar("Epoch/Alpha/PosteriorVar_mean", alpha_var.mean().item(), epoch)
            w.add_scalar("Epoch/Alpha/PosteriorVar_max", alpha_var.max().item(), epoch)
            w.add_scalar("Epoch/Alpha/Logvar_mean", self.model.alpha_logvar.mean().item(), epoch)
