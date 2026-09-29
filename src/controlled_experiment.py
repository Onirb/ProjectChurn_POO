"""Protocolo reproduzível para comparação controlada de modelos de churn.

O módulo não promove nenhum artefato para a API. Ele compara famílias no mesmo
pré-processamento, escolhe hiperparâmetros apenas no conjunto de treino e deixa
o holdout final intacto até a avaliação final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from .training_pipeline import build_churn_pipeline


RANDOM_STATE = 20260929
PRECISION_FLOOR = 0.90


@dataclass(frozen=True)
class ExperimentProtocol:
    """Parâmetros imutáveis que definem uma execução comparável."""

    random_state: int = RANDOM_STATE
    cv_splits: int = 5
    test_size: float = 0.20
    precision_floor: float = PRECISION_FLOOR
    tuning_iterations: int = 10


def candidate_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Famílias iniciais, todas adequadas a classificação tabular binária."""
    return {
        "logistic_balanced": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=random_state
        ),
        "mlp_baseline": MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=random_state,
        ),
        "random_forest_default": RandomForestClassifier(
            n_estimators=250, random_state=random_state, n_jobs=-1
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, random_state=random_state, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def make_pipeline(classifier: ClassifierMixin, random_state: int = RANDOM_STATE):
    return build_churn_pipeline(random_state=random_state, classifier=classifier)


def metric_row(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Calcula métricas de decisão e de ranking para um limiar explícito."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Lista os limiares observados e escolhe recall sujeito a precisão mínima."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    table["eligible_precision_floor"] = table["precision"] >= PRECISION_FLOOR
    return table.sort_values(["eligible_precision_floor", "recall", "precision"], ascending=False)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, precision_floor: float) -> float:
    """Maior recall entre limiares que alcançam o piso de precisão."""
    table = threshold_table(y_true, probabilities)
    eligible = table.loc[table["precision"] >= precision_floor]
    if eligible.empty:
        raise RuntimeError(
            f"Nenhum limiar alcançou precision >= {precision_floor:.2f} nos dados de validação."
        )
    winner = eligible.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(winner["threshold"])


def oof_probabilities(pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    """Probabilidades fora da dobra: base honesta para definir o limiar."""
    return cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]


def tune_tabular_pipeline(
    pipeline,
    family: str,
    cv: StratifiedKFold,
    tuning_iterations: int,
    random_state: int,
):
    """Pesquisa limitada de hiperparâmetros na família vencedora.

    Cada espaço é específico da família. Assim não misturamos parâmetros sem
    significado (por exemplo, ``max_leaf_nodes`` de boosting com Random Forest).
    """
    if family == "xgboost":
        distributions = {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__learning_rate": [0.02, 0.04, 0.06, 0.10],
            "classifier__max_depth": [3, 4, 6],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 2.0],
        }
    elif family == "hist_gradient_boosting":
        distributions = {
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.10],
            "classifier__max_iter": [200, 300, 450, 600],
            "classifier__max_leaf_nodes": [7, 15, 31, 63],
            "classifier__min_samples_leaf": [10, 20, 30, 50],
            "classifier__l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
        }
    elif family.startswith("random_forest"):
        class_weight_values: list[object] = [None, "balanced", "balanced_subsample"]
        distributions = {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__max_depth": [None, 8, 16, 24],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", 0.7],
            "classifier__class_weight": class_weight_values,
        }
    elif family == "extra_trees":
        class_weight_values = [None, "balanced"]
        distributions = {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__max_depth": [None, 8, 16, 24],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", 0.7],
            "classifier__class_weight": class_weight_values,
        }
    else:
        raise ValueError(f"A família '{family}' não possui espaço de tuning definido.")
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=distributions,
        n_iter=tuning_iterations,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )


def protocol_dict(protocol: ExperimentProtocol) -> dict[str, float | int]:
    return asdict(protocol)
