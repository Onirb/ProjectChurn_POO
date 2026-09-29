"""Executa o benchmark controlado de modelos tabulares para churn.

Uso:
    python -m scripts.run_controlled_experiment

Os resultados são locais em runs/controlled_tabular_v1/. O script não substitui
model/churn_pipeline.joblib, portanto não altera o comportamento da API.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.controlled_experiment import (
    ExperimentProtocol,
    candidate_classifiers,
    choose_threshold,
    make_pipeline,
    metric_row,
    oof_probabilities,
    protocol_dict,
    threshold_table,
    tune_tabular_pipeline,
)
from src.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import DataLoader


RUN_NAME = "controlled_tabular_v1"


def main() -> None:
    protocol = ExperimentProtocol()
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    data = DataLoader("data/processed/churn.csv").load()
    X = data.loc[:, list(FEATURE_COLUMNS)]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=protocol.test_size,
        random_state=protocol.random_state,
        stratify=y,
    )
    cv = StratifiedKFold(
        n_splits=protocol.cv_splits, shuffle=True, random_state=protocol.random_state
    )

    with (run_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **protocol_dict(protocol),
                "dataset_rows": int(len(data)),
                "train_rows": int(len(X_train)),
                "holdout_rows": int(len(X_test)),
                "model_selection_metric": "average_precision",
                "threshold_policy": "maximize recall subject to precision_floor",
                "api_artifact_changed": False,
            },
            file,
            indent=2,
        )

    stage_a_rows: list[dict[str, object]] = []
    candidates = candidate_classifiers(protocol.random_state)
    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier, protocol.random_state)
        probabilities = oof_probabilities(pipeline, X_train, y_train, cv)
        threshold = choose_threshold(y_train, probabilities, protocol.precision_floor)
        stage_a_rows.append(
            {
                "candidate": name,
                **metric_row(y_train, probabilities, threshold),
            }
        )

    stage_a = pd.DataFrame(stage_a_rows).sort_values(
        ["average_precision", "recall", "precision"], ascending=False
    )
    stage_a.to_csv(run_dir / "stage_a_oof_metrics.csv", index=False)

    best_name = str(stage_a.iloc[0]["candidate"])
    best_pipeline = make_pipeline(candidates[best_name], protocol.random_state)
    tuned = False
    search_details: dict[str, object] = {}
    if best_name in {"hist_gradient_boosting", "random_forest_default", "random_forest_balanced", "extra_trees", "xgboost"}:
        search = tune_tabular_pipeline(
            best_pipeline,
            best_name,
            cv,
            protocol.tuning_iterations,
            protocol.random_state,
        )
        search.fit(X_train, y_train)
        selected_pipeline = search.best_estimator_
        tuned = True
        search_details = {
            "stage_a_winner": best_name,
            "best_average_precision_cv": float(search.best_score_),
            "best_params": search.best_params_,
        }
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(10).to_csv(
            run_dir / "stage_b_tuning_top10.csv", index=False
        )
    else:
        selected_pipeline = best_pipeline
        selected_pipeline.fit(X_train, y_train)
        search_details = {"stage_a_winner": best_name, "best_params": "default candidate"}

    train_oof = oof_probabilities(clone(selected_pipeline), X_train, y_train, cv)
    chosen_threshold = choose_threshold(y_train, train_oof, protocol.precision_floor)
    threshold_table(train_oof * 0 + y_train.to_numpy(), train_oof).to_csv(
        run_dir / "threshold_candidates_oof.csv", index=False
    )

    # O estimador ajustado acima é reaproveitado apenas após seleção; o holdout
    # não é usado em escolha de família, hiperparâmetro nem limiar.
    if tuned:
        final_pipeline = selected_pipeline
    else:
        final_pipeline = clone(selected_pipeline).fit(X_train, y_train)
    holdout_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    holdout = metric_row(y_test, holdout_probabilities, chosen_threshold)
    pd.DataFrame([{"candidate": best_name, "tuned": tuned, **holdout}]).to_csv(
        run_dir / "holdout_metrics.csv", index=False
    )
    joblib.dump(final_pipeline, run_dir / "selected_pipeline.joblib")
    with (run_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **search_details,
                "tuned": tuned,
                "chosen_threshold_from_train_oof": chosen_threshold,
                "holdout_metrics": holdout,
                "promotion_status": "not_promoted_to_api",
            },
            file,
            indent=2,
            default=str,
        )

    print(f"Run saved to: {run_dir}")
    print(stage_a.to_string(index=False))
    print(pd.DataFrame([holdout]).to_string(index=False))


if __name__ == "__main__":
    main()
