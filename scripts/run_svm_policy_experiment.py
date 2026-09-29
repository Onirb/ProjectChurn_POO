"""Ajusta uma SVM para retenção e mostra seu compromisso precision--recall.

Uso:
    python -m scripts.run_svm_policy_experiment

O experimento usa apenas o conjunto de treino do protocolo controlado. O
holdout permanece intocado: esta é uma seleção por validação cruzada e uma
prévia de políticas, não uma promoção do modelo para a API.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.svm import SVC

from src.controlled_experiment import RANDOM_STATE, metric_row
from src.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import DataLoader
from src.training_pipeline import build_churn_pipeline


RUN_NAME = "svm_recall_policy_v1"
PRECISION_FLOORS = (0.50, 0.70, 0.80, 0.90)


def best_threshold_for_floor(y_true, probabilities, precision_floor: float) -> float:
    """Maior recall obtido sem violar o piso de precisão indicado."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    eligible = candidates.loc[candidates["precision"] >= precision_floor]
    if eligible.empty:
        raise RuntimeError(f"SVM não alcançou precisão mínima de {precision_floor:.0%}.")
    winner = eligible.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(winner["threshold"])


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

    # A busca replica a intenção histórica: maximizar recall no limiar padrão.
    # probability=False torna a busca viável; probabilidades calibradas são
    # geradas apenas depois, para construir políticas de limiar explícitas.
    base_classifier = SVC(probability=False, random_state=RANDOM_STATE)
    pipeline = build_churn_pipeline(classifier=base_classifier, random_state=RANDOM_STATE)
    distributions = {
        "classifier__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        "classifier__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=distributions,
        n_iter=24,
        scoring=make_scorer(recall_score),
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(24).to_csv(
        run_dir / "tuning_top24.csv", index=False
    )

    best_parameters = search.best_params_
    calibrated_classifier = SVC(
        C=best_parameters["classifier__C"],
        kernel=best_parameters["classifier__kernel"],
        gamma=best_parameters["classifier__gamma"],
        class_weight=best_parameters["classifier__class_weight"],
        probability=True,
        random_state=RANDOM_STATE,
    )
    calibrated_pipeline = build_churn_pipeline(
        classifier=calibrated_classifier, random_state=RANDOM_STATE
    )
    oof_probabilities = cross_val_predict(
        calibrated_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    policy_rows: list[dict[str, object]] = []
    for floor in PRECISION_FLOORS:
        threshold = best_threshold_for_floor(y_train, oof_probabilities, floor)
        policy_rows.append(
            {"precision_floor": floor, **metric_row(y_train, oof_probabilities, threshold)}
        )
    pd.DataFrame(policy_rows).to_csv(run_dir / "policy_frontier_oof.csv", index=False)

    final_pipeline = clone(calibrated_pipeline).fit(X_train, y_train)
    joblib.dump(final_pipeline, run_dir / "svm_recall_candidate.joblib")
    with (run_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_rows": int(len(X_train)),
                "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=20260929)",
                "selection_metric": "recall at default threshold 0.5",
                "best_cv_recall": float(search.best_score_),
                "best_params": best_parameters,
                "policy_evaluation": "out-of-fold probabilities on training partition",
                "holdout_used": False,
                "api_artifact_changed": False,
            },
            file,
            indent=2,
            default=str,
        )
    print(pd.DataFrame(policy_rows).to_string(index=False))
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
