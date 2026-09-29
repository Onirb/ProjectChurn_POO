"""Gera artefatos locais para a demonstração de portfólio.

Uso: python -m scripts.build_portfolio_demo_models
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import DataLoader
from src.training_pipeline import build_churn_pipeline


def main() -> None:
    data = DataLoader("data/processed/churn.csv").load()
    X = data.loc[:, list(FEATURE_COLUMNS)]
    y = data[TARGET_COLUMN]
    models = {
        "portfolio_random_forest.joblib": RandomForestClassifier(n_estimators=250, random_state=20260929, n_jobs=-1),
        "portfolio_hgb.joblib": HistGradientBoostingClassifier(
            min_samples_leaf=10, max_leaf_nodes=31, max_iter=450,
            learning_rate=0.03, l2_regularization=0.0, random_state=20260929,
        ),
    }
    output_directory = Path("model")
    output_directory.mkdir(exist_ok=True)
    for filename, classifier in models.items():
        pipeline = build_churn_pipeline(classifier=classifier, random_state=20260929)
        pipeline.fit(X, y)
        joblib.dump(pipeline, output_directory / filename)
        print(f"Saved: {output_directory / filename}")


if __name__ == "__main__":
    main()
