"""
detm/evaluate.py
----------------
Topic evaluation utilities for DETM.

Public API
----------
    TopicEvaluator — coherence, diversity, perplexity

Usage
-----
    from detm.evaluate import TopicEvaluator

    evaluator = TopicEvaluator(tokens_list, vocab)

    # Coherence + diversity across multiple top-N values and multiple time steps
    metrics = evaluator.evaluate_topics(model, top_n_words=[10, 15, 20])

    # Perplexity on a held-out DataLoader
    ppl = TopicEvaluator.compute_perplexity(model, test_loader, device)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from torch.utils.data import DataLoader

from detm.model import DETM


class TopicEvaluator:
    """
    Evaluate topic quality for a trained DETM model.

    Parameters
    ----------
    tokens_list : tokenised documents (list of lists of str)
    vocabulary  : ordered vocab list (index i → word i)
    """

    def __init__(self, tokens_list: List[List[str]], vocabulary: List[str]) -> None:
        self.tokens_list = tokens_list
        self.vocabulary = vocabulary
        self.dictionary = Dictionary(tokens_list)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokens_list]

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def compute_coherence(
        self,
        topics: List[List[str]],
        coherence_type: str = "c_v",
    ) -> float:
        """
        Average topic coherence (higher is better).

        Parameters
        ----------
        topics         : list of K topics, each a list of top-N word strings
        coherence_type : ``"c_v"`` or ``"c_npmi"``
        """
        cm = CoherenceModel(
            topics=topics,
            texts=self.tokens_list,
            dictionary=self.dictionary,
            coherence=coherence_type,
        )
        return cm.get_coherence()

    def compute_topic_diversity(self, topics: List[List[str]]) -> float:
        """
        Fraction of unique words across all topic top-N lists.

        Higher → less topic redundancy.
        """
        unique_words: set = set()
        total_words = 0
        for topic in topics:
            unique_words.update(topic)
            total_words += len(topic)
        return len(unique_words) / total_words if total_words > 0 else 0.0

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------

    def evaluate_topics(
        self,
        model: DETM,
        top_n_words: Optional[List[int]] = None,
        time_sample_count: int = 5,
    ) -> Dict[str, float]:
        """
        Run coherence & diversity at multiple ``top_n`` values, averaged over
        several sampled time steps.

        Coherence is sampled at ``time_sample_count`` evenly-spaced time
        steps (including t=0 and t=T-1) and then averaged, giving a more
        robust estimate of the model's quality across the full temporal range.

        Diversity is reported at the last time step (most trained point).

        Parameters
        ----------
        model            : trained DETM (must have ``idx2word`` set)
        top_n_words      : list of N values; default ``[10, 15, 20]``
        time_sample_count: how many time steps to sample for coherence

        Returns
        -------
        results dict::

            {
                "coherence_cv_top10": 0.502,
                "coherence_cv_top10_t0": 0.481,
                ...
                "coherence_npmi_top10": 0.011,
                "diversity_top10": 0.592,
                ...
            }
        """
        if top_n_words is None:
            top_n_words = [10, 15, 20]

        T = model.num_time_steps

        # Sample time steps: always include 0 and T-1
        if time_sample_count >= T:
            sampled_t = list(range(T))
        else:
            sampled_t = sorted(set(
                [0] + [int(round(i * (T - 1) / (time_sample_count - 1)))
                        for i in range(time_sample_count)]
            ))

        results: Dict[str, float] = {}

        for n in top_n_words:
            cv_scores, npmi_scores = [], []

            for t in sampled_t:
                topics_with_probs = model.get_topics(time_idx=t, top_n=n)
                topics = [[word for word, _ in topic] for topic in topics_with_probs]
                cv_scores.append(self.compute_coherence(topics, "c_v"))
                npmi_scores.append(self.compute_coherence(topics, "c_npmi"))

            results[f"coherence_cv_top{n}"] = float(np.mean(cv_scores))
            results[f"coherence_npmi_top{n}"] = float(np.mean(npmi_scores))

            # Diversity at last time step
            last_topics_p = model.get_topics(time_idx=T - 1, top_n=n)
            last_topics = [[word for word, _ in tp] for tp in last_topics_p]
            results[f"diversity_top{n}"] = self.compute_topic_diversity(last_topics)

        return results

    # ------------------------------------------------------------------
    # Perplexity (static — no reference corpus needed)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_perplexity(
        model: DETM,
        data_loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Perplexity = exp(NLL / num_words) on a held-out set (lower is better).

        The per-batch NLL is recovered from ``recon_loss`` by reversing the
        corpus-scale normalisation applied in ``DETM.forward``:

            recon_loss = per_doc_nlls.sum() * (D / B)

        So the total corpus NLL for a batch  = recon_loss * B / D,
        and summed across all batches:
            total_NLL = Σ_batches  recon_loss_b * B_b / D

        Then perplexity = exp(total_NLL / total_words).

        Parameters
        ----------
        model       : trained DETM
        data_loader : DataLoader for evaluation split
        device      : torch.device
        """
        model.eval()
        total_nll = 0.0
        total_words = 0.0
        D = model.num_train_docs

        with torch.no_grad():
            for batch in data_loader:
                bow = batch["bow"].to(device)
                t_idx = batch["time_idx"].to(device)
                B = bow.shape[0]
                output = model(bow, t_idx, compute_loss=True)
                # Reverse corpus scaling: recon_loss = sum_nll * (D/B)  →  sum_nll = recon * B/D
                total_nll += output["recon_loss"].item() * B / D
                total_words += bow.sum().item()

        return float(np.exp(total_nll / total_words)) if total_words > 0 else float("inf")
