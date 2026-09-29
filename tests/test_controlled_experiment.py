import numpy as np
import pandas as pd

from src.controlled_experiment import choose_threshold, metric_row, threshold_table


def test_threshold_selection_respects_precision_floor():
    target = pd.Series([1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.95, 0.80, 0.70, 0.20, 0.10, 0.05])

    threshold = choose_threshold(target, probabilities, precision_floor=0.90)
    metrics = metric_row(target, probabilities, threshold)

    assert metrics["precision"] >= 0.90
    assert metrics["recall"] == 1.0


def test_threshold_table_marks_eligible_rows():
    target = pd.Series([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])

    table = threshold_table(target, probabilities)

    assert "eligible_precision_floor" in table.columns
    assert table["eligible_precision_floor"].any()
