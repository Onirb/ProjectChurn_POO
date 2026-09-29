"""Reconstrução controlada do pré-processamento ``d3`` do notebook histórico.

O objetivo é comparar a transformação global usada na exploração original com
uma transformação equivalente aprendida em cada dobra. A variante histórica é
mantida apenas para auditoria: ela não deve alimentar a API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data_contract import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from .training_pipeline import USAGE_COLUMNS, build_churn_pipeline


def historical_d3(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reproduz a lógica d3, inclusive o ajuste global histórico.

    A imputação e o scaler são deliberadamente ajustados no dataframe completo
    para tornar explícita a diferença entre legado e protocolo correto. Não use
    esta função em produção ou para alegar uma métrica sem vazamento.
    """
    frame = dataframe.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    numeric = list(NUMERIC_COLUMNS)
    frame[numeric] = frame[numeric].fillna(frame[numeric].median())
    # O CSV atual tem 50 ausências também nos dois planos. O notebook aplicava
    # ``fillna(d3.median())``, que não resolve colunas categóricas. A moda global
    # abaixo é a menor complementação necessária para tornar o legado executável
    # e fica identificada no protocolo como aproximação, não como resultado exato.
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].fillna(frame[column].mode().iloc[0])
    frame["numbervmailmessages"] = np.log1p(frame["numbervmailmessages"])
    scaler = MinMaxScaler()
    frame[USAGE_COLUMNS] = scaler.fit_transform(frame[USAGE_COLUMNS])
    return frame


def benchmark_classifiers(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Famílias comparadas na análise original, com versões atuais reproduzíveis."""
    return {
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=6,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_balanced": SVC(
            class_weight="balanced", probability=True, random_state=random_state
        ),
        "knn": KNeighborsClassifier(),
        "decision_tree_balanced": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "gaussian_nb": GaussianNB(),
        "mlp_baseline": MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }


def metrics_at_half(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Métricas de classificação com limiar comum para comparação histórica."""
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def leakage_free_pipeline(classifier: ClassifierMixin, random_state: int = 42):
    """Usa o mesmo d3 conceitual, mas aprende as estatísticas apenas no treino."""
    return build_churn_pipeline(random_state=random_state, classifier=classifier)
