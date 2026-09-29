import numpy as np
import pandas as pd

from src.policy_frontier import top_k_metrics


def test_top_k_metrics_selects_the_highest_risk_rows():
    target = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.8, 0.7])

    result = top_k_metrics(target, probabilities, capacity=2)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["recall_at_k"] == 0.5


def test_top_k_metrics_rejects_invalid_capacity():
    target = pd.Series([0, 1])

    try:
        top_k_metrics(target, np.array([0.1, 0.9]), capacity=0)
    except ValueError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("Era esperado ValueError para capacidade inválida.")
