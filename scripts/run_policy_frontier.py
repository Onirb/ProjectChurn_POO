"""Compara políticas de retenção para modelos já validados localmente.

Uso:
    python -m scripts.run_policy_frontier
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.controlled_experiment import RANDOM_STATE, choose_threshold, metric_row
from src.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import DataLoader
from src.policy_frontier import top_k_metrics
from src.training_pipeline import build_churn_pipeline


RUN_NAME = "policy_frontier_v1"
PRECISION_FLOORS = (0.50, 0.70, 0.80, 0.90)
CAPACITIES = (100, 250, 500)


def candidate_models():
    """Modelos selecionados previamente, todos com pré-processamento comum."""
    return {
        "hist_gradient_boosting_tuned": HistGradientBoostingClassifier(
            min_samples_leaf=10,
            max_leaf_nodes=31,
            max_iter=450,
            learning_rate=0.03,
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "svm_recall_tuned": SVC(
            C=30.0,
            kernel="rbf",
            gamma=0.01,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    data = DataLoader("data/processed/churn.csv").load()
    X = data.loc[:, list(FEATURE_COLUMNS)]
    y = data[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    floor_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []

    for model_name, classifier in candidate_models().items():
        pipeline = build_churn_pipeline(classifier=classifier, random_state=RANDOM_STATE)
        probabilities = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        for floor in PRECISION_FLOORS:
            threshold = choose_threshold(y_train, probabilities, floor)
            floor_rows.append(
                {
                    "model": model_name,
                    "precision_floor": floor,
                    **metric_row(y_train, probabilities, threshold),
                }
            )
        for capacity in CAPACITIES:
            capacity_rows.append(
                {"model": model_name, **top_k_metrics(y_train, probabilities, capacity)}
            )

    floor_frame = pd.DataFrame(floor_rows).sort_values(["precision_floor", "recall"], ascending=[True, False])
    capacity_frame = pd.DataFrame(capacity_rows).sort_values(["capacity", "recall_at_k"], ascending=[True, False])
    floor_frame.to_csv(run_dir / "precision_floor_frontier_oof.csv", index=False)
    capacity_frame.to_csv(run_dir / "capacity_frontier_oof.csv", index=False)
    with (run_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_rows": int(len(X_train)),
                "probabilities": "out-of-fold on training partition",
                "precision_floors": PRECISION_FLOORS,
                "capacities": CAPACITIES,
                "models": list(candidate_models()),
                "holdout_used": False,
                "api_artifact_changed": False,
            },
            file,
            indent=2,
        )
    print("Precision floors")
    print(floor_frame.to_string(index=False))
    print("\nCampaign capacity")
    print(capacity_frame.to_string(index=False))
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
