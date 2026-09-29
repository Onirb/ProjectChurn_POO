"""Métricas para políticas de retenção baseadas em probabilidade de churn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def top_k_metrics(y_true: pd.Series, probabilities: np.ndarray, capacity: int) -> dict[str, float | int]:
    """Avaliação de uma campanha que pode contatar exatamente ``capacity`` pessoas."""
    if capacity < 1 or capacity > len(y_true):
        raise ValueError("capacity deve estar entre 1 e o número de linhas.")
    target = np.asarray(y_true, dtype=int)
    selected = np.zeros(len(target), dtype=int)
    indices = np.argsort(-probabilities, kind="stable")[:capacity]
    selected[indices] = 1
    tp = int(((selected == 1) & (target == 1)).sum())
    fp = int(((selected == 1) & (target == 0)).sum())
    fn = int(((selected == 0) & (target == 1)).sum())
    tn = int(((selected == 0) & (target == 0)).sum())
    return {
        "capacity": capacity,
        "minimum_selected_probability": float(probabilities[indices].min()),
        "precision_at_k": float(precision_score(target, selected, zero_division=0)),
        "recall_at_k": float(recall_score(target, selected, zero_division=0)),
        "f1_at_k": float(f1_score(target, selected, zero_division=0)),
        "accuracy_at_k": float(accuracy_score(target, selected)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }
