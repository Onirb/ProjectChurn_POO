"""Pipeline único de transformação e classificação para treino e inferência."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


USAGE_COLUMNS = [
    "accountlength",
    "totaldayminutes",
    "totaldaycalls",
    "totaldaycharge",
    "totaleveminutes",
    "totalevecalls",
    "totalevecharge",
    "totalnightminutes",
    "totalnightcalls",
    "totalnightcharge",
    "totalintlminutes",
    "totalintlcalls",
    "totalintlcharge",
]


def build_churn_pipeline(
    n_estimators: int = 100,
    random_state: int = 42,
    classifier: ClassifierMixin | None = None,
) -> Pipeline:
    """Cria um pipeline autocontido para treino, experimento e inferência.

    Sem um classificador explícito, mantém o Random Forest histórico usado pela
    API. Experimentos podem injetar outra família sem alterar o contrato de
    dados nem a transformação compartilhada.
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "usage",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                USAGE_COLUMNS,
            ),
            (
                "voicemail_messages",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(np.log1p, validate=False)),
                    ]
                ),
                ["numbervmailmessages"],
            ),
            (
                "customer_service_calls",
                SimpleImputer(strategy="median"),
                ["numbercustomerservicecalls"],
            ),
            (
                "plans",
                SimpleImputer(strategy="most_frequent"),
                ["internationalplan", "voicemailplan"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                classifier
                if classifier is not None
                else RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
            ),
        ]
    )
