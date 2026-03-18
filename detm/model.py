"""
detm/model.py
-------------
PyTorch modules implementing the Dynamic Embedded Topic Model
(Dieng, Ruiz & Blei 2019 — https://arxiv.org/abs/1907.05545).

Module hierarchy
----------------
    DocumentTopicEncoder   — amortized inference for θ_d  (MLP, logistic-normal)
    DETMDecoder            — generative decoder  β = softmax(α · ρᵀ)
    TemporalBaselineEncoder— structured inference for η_t  (LSTM)
    ReconstructionLoss     — multinomial / Poisson NLL
    DETM                   — full model combining all components

Importing
---------
    from detm.model import DETM
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detm.config import DETMConfig


# ---------------------------------------------------------------------------
# Document Topic Encoder  (amortized inference for θ_d)
# ---------------------------------------------------------------------------

class DocumentTopicEncoder(nn.Module):
    """
    Amortized variational inference for per-document topic proportions θ_d.

    Architecture
    ------------
    Input: [L1-normalised BoW (V), η_{t_d} (K)]  →  (V+K)
    MLP:   Linear(V+K→H) → ReLU → Linear(H→H) → ReLU
    Heads: μ_θ ∈ R^K,  log σ²_θ ∈ R^K
    Output: θ_d = softmax(μ_θ + ε⊙exp(0.5·log σ²_θ))   [logistic-normal]

    No dropout (enc_drop=0.0 in original) — dropout on VAE encoders causes
    posterior collapse by injecting noise that competes with reparameterization.
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        hidden_dim: int = 800,
        dropout: float = 0.0,   # kept for API compatibility; unused when 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics

        input_dim = vocab_size + num_topics
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, num_topics)
        self.fc_logvar = nn.Linear(hidden_dim, num_topics)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self,
        bow: torch.Tensor,
        eta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        bow : (B, V)  — raw word counts
        eta : (B, K)  — temporal baseline for each document's year

        Returns
        -------
        theta  : (B, K) on simplex
        mu     : (B, K) pre-softmax mean
        logvar : (B, K) pre-softmax log-variance
        """
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        h = self.encoder(torch.cat([bow_norm, eta], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        theta = F.softmax(self.reparameterize(mu, logvar), dim=-1)
        return theta, mu, logvar


# ---------------------------------------------------------------------------
# DETM Decoder
# ---------------------------------------------------------------------------

class DETMDecoder(nn.Module):
    """
    Generative decoder: β_k = softmax(α_k · ρᵀ), p(w|d) = θᵀ · β.

    Word embeddings ρ are stored as raw W2V vectors (no L2-normalisation) and
    are trainable by default, matching the original implementation.
    L2-normalisation compresses logit dynamic range and — combined with small α
    init — produces β ≈ uniform for every topic → posterior collapse.
    """

    def __init__(
        self,
        num_topics: int,
        vocab_size: int,
        embedding_dim: int,
        word_embeddings: torch.Tensor,
        train_word_embeddings: bool = True,
    ):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Raw W2V embeddings stored as parameter so model.to(device) moves them.
        self.word_embeddings = nn.Parameter(
            word_embeddings.clone(), requires_grad=train_word_embeddings
        )

    def get_beta(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute topic-word distributions.

        Parameters
        ----------
        alpha : (K, L) or (B, K, L)

        Returns
        -------
        beta : (K, V) or (B, K, V)
        """
        if alpha.dim() == 2:
            return F.softmax(torch.mm(alpha, self.word_embeddings.t()), dim=-1)
        return F.softmax(torch.matmul(alpha, self.word_embeddings.t()), dim=-1)

    def forward(self, theta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        theta : (B, K)
        alpha : (B, K, L)

        Returns
        -------
        word_dist : (B, V)
        """
        beta = self.get_beta(alpha)          # (B, K, V)
        return torch.bmm(theta.unsqueeze(1), beta).squeeze(1)  # (B, V)


# ---------------------------------------------------------------------------
# Temporal Baseline Encoder  (structured inference for η_t)
# ---------------------------------------------------------------------------

class TemporalBaselineEncoder(nn.Module):
    """
    LSTM-based structured variational inference for global temporal baselines η_t.

    For each time step t the LSTM output is concatenated with η_{t-1} and fed to
    linear heads that produce μ_η and log σ²_η.  A fresh η_t sample is drawn and
    the KL against the random-walk prior p(η_t | η_{t-1}) = N(η_{t-1}, δ²I) is
    accumulated.  Computing KL inside the loop keeps η_{t-1} in the live graph,
    enabling correct Monte Carlo gradients.
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        compression_dim: int = 400,
        lstm_layers: int = 4,
        lstm_hidden: int = 400,
        delta_sq: float = 0.005,
    ):
        super().__init__()
        self.num_topics = num_topics
        self.delta_sq = delta_sq

        self.compress = nn.Linear(vocab_size, compression_dim)
        self.lstm = nn.LSTM(
            input_size=compression_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,   # eta_dropout=0.0 in original
        )
        self.fc_mu = nn.Linear(lstm_hidden + num_topics, num_topics)
        self.fc_logvar = nn.Linear(lstm_hidden + num_topics, num_topics)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(
        self,
        avg_bow_timeline: torch.Tensor,
        return_params: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        avg_bow_timeline : (T, V) — per-time-step average normalised BoW
        return_params    : if True, also return (mu_timeline, logvar_timeline)

        Returns
        -------
        eta_timeline : (T, K)
        kl_eta       : scalar
        [mu_timeline, logvar_timeline] — only when return_params=True
        """
        T = avg_bow_timeline.shape[0]
        device = avg_bow_timeline.device

        lstm_out, _ = self.lstm(self.compress(avg_bow_timeline).unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)  # (T, lstm_hidden)

        eta_prev = torch.zeros(1, self.num_topics, device=device)
        eta_list, mu_list, logvar_list, kl_terms = [], [], [], []

        for t in range(T):
            inp = torch.cat([lstm_out[t:t+1], eta_prev], dim=1)
            mu_t = self.fc_mu(inp)
            logvar_t = self.fc_logvar(inp)
            eta_t = self.reparameterize(mu_t, logvar_t)

            var_t = logvar_t.exp()
            kl_t = 0.5 * torch.sum(
                var_t / self.delta_sq
                + (mu_t - eta_prev).pow(2) / self.delta_sq
                - 1.0
                - torch.log(var_t / self.delta_sq)
            )
            kl_terms.append(kl_t)
            eta_prev = eta_t
            eta_list.append(eta_t)
            mu_list.append(mu_t)
            logvar_list.append(logvar_t)

        eta_timeline = torch.cat(eta_list, dim=0)     # (T, K)
        kl_eta = sum(kl_terms)

        if return_params:
            return (
                eta_timeline,
                torch.cat(mu_list, dim=0),
                torch.cat(logvar_list, dim=0),
                kl_eta,
            )
        return eta_timeline, kl_eta


# ---------------------------------------------------------------------------
# Reconstruction Loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """Multinomial or Poisson negative log-likelihood."""

    def __init__(self, loss_type: str = "multinomial"):
        super().__init__()
        if loss_type not in ("multinomial", "poisson"):
            raise ValueError(f"Unknown loss_type: {loss_type!r}")
        self.loss_type = loss_type

    def forward(
        self,
        bow: torch.Tensor,
        word_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch-mean NLL scalar."""
        if self.loss_type == "multinomial":
            nll = -(bow * torch.log(word_dist.clamp(min=1e-10))).sum(dim=-1)
        else:  # poisson
            doc_len = bow.sum(dim=-1, keepdim=True)
            rates = (word_dist * doc_len).clamp(min=1e-10)
            nll = (rates - bow * rates.log()).sum(dim=-1)
        return nll.mean()


# ---------------------------------------------------------------------------
# DETM  (full model)
# ---------------------------------------------------------------------------

class DETM(nn.Module):
    """
    Dynamic Embedded Topic Model (Dieng et al. 2019).

    Three variational families
    --------------------------
    1. TemporalBaselineEncoder — LSTM structured inference for η_t
    2. DocumentTopicEncoder    — MLP amortized inference for θ_d (logistic-normal)
    3. Mean-field parameters   — α^(t) ∈ R^{T×K×L}

    ELBO
    ----
    E_q[log p(w|θ,α,ρ)] − KL[q(θ)||p(θ)] − KL[q(η)||p(η)] − KL[q(α)||p(α)]

    Loss normalisation
    ------------------
    recon_loss and kl_theta are per-document means.
    kl_eta and kl_alpha are global sums divided by ``num_train_docs`` to bring
    all terms to the same per-document scale.
    """

    def __init__(
        self,
        config: DETMConfig,
        word_embeddings: torch.Tensor,
        num_time_steps: int,
        avg_bow_timeline: np.ndarray,
        num_train_docs: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_time_steps = num_time_steps
        self.num_train_docs = num_train_docs

        vocab_size, emb_dim = word_embeddings.shape
        self.register_buffer("avg_bow_timeline", torch.FloatTensor(avg_bow_timeline))

        self.temporal_baseline_encoder = TemporalBaselineEncoder(
            vocab_size=vocab_size,
            num_topics=config.NUM_TOPICS,
            compression_dim=config.COMPRESSION_DIM,
            lstm_layers=config.LSTM_LAYERS,
            lstm_hidden=config.LSTM_HIDDEN,
            delta_sq=config.ETA_PRIOR_VARIANCE,
        )
        self.doc_encoder = DocumentTopicEncoder(
            vocab_size=vocab_size,
            num_topics=config.NUM_TOPICS,
            hidden_dim=config.DOC_HIDDEN_DIM,
            dropout=config.DOC_DROPOUT,
        )
        self.alpha_mu = nn.Parameter(
            torch.randn(num_time_steps, config.NUM_TOPICS, emb_dim) * config.INIT_ALPHA_STD
        )
        self.alpha_logvar = nn.Parameter(
            torch.ones(num_time_steps, config.NUM_TOPICS, emb_dim) * config.INIT_ALPHA_LOGVAR
        )
        self.decoder = DETMDecoder(
            num_topics=config.NUM_TOPICS,
            vocab_size=vocab_size,
            embedding_dim=emb_dim,
            word_embeddings=word_embeddings,
            train_word_embeddings=True,
        )
        self.recon_loss_fn = ReconstructionLoss(loss_type="multinomial")

        # Set by caller (e.g. trainer) for get_topics()
        self.idx2word: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Alpha sampling
    # ------------------------------------------------------------------

    def _sample_alpha(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Sample α^(t) once per unique time step and broadcast to all docs at t.

        This matches the global-latent semantics of α: one embedding set per
        year, shared by all documents from that year.  Per-document independent
        noise would add destructive gradient variance.
        """
        if not self.training:
            return self.alpha_mu[time_indices]  # (B, K, L) — posterior mean

        out = torch.empty(
            len(time_indices), self.alpha_mu.shape[1], self.alpha_mu.shape[2],
            device=self.alpha_mu.device, dtype=self.alpha_mu.dtype,
        )
        for t in time_indices.unique():
            mask = time_indices == t
            mu_t = self.alpha_mu[t]
            eps = torch.randn_like(mu_t)
            out[mask] = mu_t + eps * torch.exp(0.5 * self.alpha_logvar[t])
        return out

    # ------------------------------------------------------------------
    # Alpha KL
    # ------------------------------------------------------------------

    def _compute_alpha_kl(self) -> torch.Tensor:
        """
        KL(q(α) || p(α)) summed over all T × K × L elements.

        Handles the expectation over the random prior mean α_{t-1} by adding
        var_{t-1}/γ² — the term missing from the reference implementation.
        Caller divides by num_train_docs to normalise to per-document scale.
        """
        gamma_sq = self.config.ALPHA_PRIOR_VARIANCE
        kl = 0.0

        mu_0 = self.alpha_mu[0]
        var_0 = self.alpha_logvar[0].exp()
        kl += 0.5 * torch.sum(
            var_0 / gamma_sq + mu_0.pow(2) / gamma_sq - 1.0 - torch.log(var_0 / gamma_sq)
        )

        for t in range(1, self.num_time_steps):
            mu_t = self.alpha_mu[t]
            var_t = self.alpha_logvar[t].exp()
            mu_p = self.alpha_mu[t - 1]
            var_p = self.alpha_logvar[t - 1].exp()
            kl += 0.5 * torch.sum(
                var_t / gamma_sq
                + ((mu_t - mu_p).pow(2) + var_p) / gamma_sq
                - 1.0
                - torch.log(var_t / gamma_sq)
            )
        return kl

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        bow: torch.Tensor,
        time_indices: torch.Tensor,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        bow          : (B, V) raw word count tensor
        time_indices : (B,)  integer time-step index per document
        compute_loss : if False, skip ELBO computation (faster at inference)

        Returns
        -------
        Dict with:
            theta     : (B, K)
            word_dist : (B, V)
            alpha     : (B, K, L)
            [loss, recon_loss, kl_theta, kl_eta, kl_alpha]  — when compute_loss=True
        """
        # LSTM re-run per batch — fresh graph, no graph-reuse errors
        eta_timeline, kl_eta = self.temporal_baseline_encoder(self.avg_bow_timeline)
        eta_batch = eta_timeline[time_indices]  # (B, K)

        theta, theta_mu, theta_logvar = self.doc_encoder(bow, eta_batch)
        alpha_batch = self._sample_alpha(time_indices)          # (B, K, L)
        word_dist = self.decoder(theta, alpha_batch)            # (B, V)

        out: Dict[str, torch.Tensor] = {
            "theta": theta,
            "theta_mu": theta_mu,
            "theta_logvar": theta_logvar,
            "word_dist": word_dist,
            "alpha": alpha_batch,
        }

        if compute_loss:
            recon = self.recon_loss_fn(bow, word_dist)

            var_theta = theta_logvar.exp()
            kl_theta = 0.5 * torch.sum(
                (theta_mu - eta_batch).pow(2) + var_theta - theta_logvar - 1.0,
                dim=-1,
            ).mean()

            kl_eta_n = kl_eta / self.num_train_docs
            kl_alpha_n = self._compute_alpha_kl() / self.num_train_docs

            cfg = self.config
            loss = (
                cfg.RECON_WEIGHT * recon
                + cfg.KL_THETA_WEIGHT * kl_theta
                + cfg.KL_ETA_WEIGHT * kl_eta_n
                + cfg.KL_ALPHA_WEIGHT * kl_alpha_n
            )
            out.update({
                "loss": loss,
                "recon_loss": recon,
                "kl_theta": kl_theta,
                "kl_eta": kl_eta_n,
                "kl_alpha": kl_alpha_n,
            })

        return out

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_topics(
        self,
        time_idx: int = -1,
        top_n: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Return top-N (word, probability) pairs per topic at a given time step.

        Requires ``model.idx2word`` to be set.
        """
        if self.idx2word is None:
            raise ValueError("Set model.idx2word before calling get_topics().")

        with torch.no_grad():
            if time_idx == -1:
                time_idx = self.num_time_steps - 1
            beta = self.decoder.get_beta(self.alpha_mu[time_idx]).cpu().numpy()  # (K, V)

        return [
            [(self.idx2word[i], float(beta[k, i])) for i in beta[k].argsort()[-top_n:][::-1]]
            for k in range(self.config.NUM_TOPICS)
        ]

    @torch.no_grad()
    def get_document_topics(
        self,
        bow: torch.Tensor,
        time_indices: torch.Tensor,
    ) -> np.ndarray:
        """Return (N, K) topic proportion matrix for a batch of documents."""
        self.eval()
        device = next(self.parameters()).device
        eta_timeline, _ = self.temporal_baseline_encoder(self.avg_bow_timeline)
        theta, _, _ = self.doc_encoder(
            bow.to(device), eta_timeline[time_indices.to(device)]
        )
        return theta.cpu().numpy()
