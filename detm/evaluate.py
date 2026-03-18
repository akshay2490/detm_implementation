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
    metrics   = evaluator.evaluate_topics(model, top_n_words=[10, 15, 20])
    ppl       = TopicEvaluator.compute_perplexity(model, test_loader, device)
"""

from __future__ import annotations

from typing import Dict, List

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
                  — the same split shown to the model during training
    vocabulary  : ordered vocab list (index i → word i)
    """

    def __init__(self, tokens_list: List[List[str]], vocabulary: List[str]):
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
        Fraction of unique words across top-N topic words (higher = less redundancy).

        Parameters
        ----------
        topics : list of K topics, each a list of word strings
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
        top_n_words: List[int] | None = None,
    ) -> Dict[str, float]:
        """
        Run coherence & diversity evaluation for multiple ``top_n`` values.

        Parameters
        ----------
        model       : trained DETM (must have ``idx2word`` set)
        top_n_words : list of N values; default ``[10, 15, 20]``

        Returns
        -------
        results dict, e.g.::

            {
                "coherence_cv_top10": 0.502,
                "coherence_npmi_top10": 0.011,
                "diversity_top10": 0.592,
                ...
            }
        """
        if top_n_words is None:
            top_n_words = [10, 15, 20]

        results: Dict[str, float] = {}
        for n in top_n_words:
            topics_with_probs = model.get_topics(top_n=n)
            topics = [[word for word, _ in topic] for topic in topics_with_probs]

            results[f"coherence_cv_top{n}"] = self.compute_coherence(topics, "c_v")
            results[f"coherence_npmi_top{n}"] = self.compute_coherence(topics, "c_npmi")
            results[f"diversity_top{n}"] = self.compute_topic_diversity(topics)

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

        Parameters
        ----------
        model       : trained DETM
        data_loader : DataLoader for evaluation split
        device      : torch.device
        """
        model.eval()
        total_nll = 0.0
        total_words = 0.0

        with torch.no_grad():
            for batch in data_loader:
                bow = batch["bow"].to(device)
                t_idx = batch["time_idx"].to(device)
                output = model(bow, t_idx, compute_loss=True)
                # recon_loss is already averaged per document; scale back to per-word
                total_nll += output["recon_loss"].item() * len(bow)
                total_words += bow.sum().item()

        return float(np.exp(total_nll / total_words))
