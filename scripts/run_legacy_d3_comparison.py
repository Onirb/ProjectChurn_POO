"""Compara d3 histórico global com d3 sem vazamento nas mesmas cinco dobras.

Uso:
    python -m scripts.run_legacy_d3_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import DataLoader
from src.legacy_benchmark import (
    benchmark_classifiers,
    historical_d3,
    leakage_free_pipeline,
    metrics_at_half,
)


RUN_NAME = "legacy_d3_reconstruction_v1"


def main() -> None:
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    data = DataLoader("data/processed/churn.csv").load()
    historical = historical_d3(data)
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    rows: list[dict[str, object]] = []

    for split_number, (train_index, test_index) in enumerate(
        splitter.split(data.loc[:, list(FEATURE_COLUMNS)], data[TARGET_COLUMN]), start=1
    ):
        X_train_raw = data.iloc[train_index].loc[:, list(FEATURE_COLUMNS)]
        X_test_raw = data.iloc[test_index].loc[:, list(FEATURE_COLUMNS)]
        X_train_historical = historical.iloc[train_index].loc[:, list(FEATURE_COLUMNS)]
        X_test_historical = historical.iloc[test_index].loc[:, list(FEATURE_COLUMNS)]
        y_train = data.iloc[train_index][TARGET_COLUMN]
        y_test = data.iloc[test_index][TARGET_COLUMN]

        for model_name, classifier in benchmark_classifiers().items():
            legacy_model = clone(classifier).fit(X_train_historical, y_train)
            legacy_probabilities = legacy_model.predict_proba(X_test_historical)[:, 1]
            rows.append(
                {
                    "preprocessing": "historical_global_d3",
                    "split": split_number,
                    "model": model_name,
                    **metrics_at_half(y_test, legacy_probabilities),
                }
            )

            safe_model = leakage_free_pipeline(clone(classifier)).fit(X_train_raw, y_train)
            safe_probabilities = safe_model.predict_proba(X_test_raw)[:, 1]
            rows.append(
                {
                    "preprocessing": "leakage_free_d3",
                    "split": split_number,
                    "model": model_name,
                    **metrics_at_half(y_test, safe_probabilities),
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["preprocessing", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            average_precision_mean=("average_precision", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            f1_std=("f1", "std"),
        )
        .sort_values(["preprocessing", "average_precision_mean"], ascending=[True, False])
    )
    detail.to_csv(run_dir / "per_split_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    with (run_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "splits": 5,
                "splitter": "StratifiedShuffleSplit(test_size=0.2, random_state=42)",
                "threshold": 0.5,
                "historical_global_d3": "numeric median, categorical mode and MinMaxScaler fit on all rows before split; categorical mode is an explicit reconstruction complement",
                "leakage_free_d3": "same transformations fit within each training fold through Pipeline",
                "models": list(benchmark_classifiers()),
                "api_artifact_changed": False,
            },
            file,
            indent=2,
        )
    print(summary.to_string(index=False))
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
