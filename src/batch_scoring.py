"""Pontuação local de lotes para a demonstração de campanhas de retenção."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_contract import DatasetContractError, validate_feature_dataframe


class BatchScoringError(ValueError):
    """Erro de formato ou de artefato no fluxo de pontuação de lote."""


MODEL_ARTIFACTS = {
    "random_forest": "portfolio_random_forest.joblib",
    "hist_gradient_boosting_tuned": "portfolio_hgb.joblib",
}
IDENTIFIER_NORMALIZED_NAMES = {"customerid", "id"}


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _identifier_column(dataframe: pd.DataFrame) -> str:
    matches = [str(column) for column in dataframe.columns if _normalized_name(column) in IDENTIFIER_NORMALIZED_NAMES]
    if not matches:
        raise BatchScoringError("O lote precisa de uma coluna 'customer_id' ou 'id'.")
    if len(matches) > 1:
        raise BatchScoringError("Informe apenas uma coluna de identificador: 'customer_id' ou 'id'.")
    identifier = matches[0]
    values = dataframe[identifier]
    if values.isna().any() or values.astype("string").str.strip().eq("").any():
        raise BatchScoringError("Cada cliente precisa ter um identificador preenchido.")
    if values.duplicated().any():
        raise BatchScoringError("Os identificadores dos clientes devem ser únicos no lote.")
    return identifier


def load_demo_pipeline(candidate_model: str, model_directory: Path | None = None):
    """Carrega somente os artefatos locais próprios para a demonstração."""
    artifact_name = MODEL_ARTIFACTS.get(candidate_model)
    if artifact_name is None:
        raise BatchScoringError(f"Modelo de demonstração não reconhecido: {candidate_model}.")
    directory = model_directory or Path("model")
    artifact = directory / artifact_name
    if not artifact.exists():
        raise BatchScoringError(
            "Artefato de demonstração ausente. Execute 'python -m scripts.build_portfolio_demo_models'."
        )
    return joblib.load(artifact)


def score_customer_batch(
    dataframe: pd.DataFrame,
    campaign_plan: dict[str, Any],
    pipeline: Any | None = None,
    model_directory: Path | None = None,
) -> pd.DataFrame:
    """Ordena um lote e marca a quantidade indicada pelo plano comercial."""
    if dataframe.empty:
        raise BatchScoringError("O lote não contém clientes.")
    identifier_column = _identifier_column(dataframe)
    try:
        features = validate_feature_dataframe(dataframe)
    except DatasetContractError as error:
        raise BatchScoringError(str(error)) from error

    technical = campaign_plan["technical_policy"]
    active_pipeline = pipeline or load_demo_pipeline(technical["candidate_model"], model_directory=model_directory)
    probabilities = np.asarray(active_pipeline.predict_proba(features))[:, 1]
    scored = pd.DataFrame(
        {
            "customer_id": dataframe[identifier_column].astype("string"),
            "churn_probability": probabilities.round(6),
        }
    ).sort_values("churn_probability", ascending=False, kind="stable")
    scored["campaign_rank"] = range(1, len(scored) + 1)

    threshold = technical.get("threshold")
    if threshold is not None:
        eligible = scored["churn_probability"] >= float(threshold)
        selected_limit = min(int(campaign_plan["suggested_contacts"]), int(eligible.sum()))
        selected_indices = scored.loc[eligible].head(selected_limit).index
    else:
        selected_limit = min(int(campaign_plan["suggested_contacts"]), len(scored))
        selected_indices = scored.head(selected_limit).index

    scored["recommended_for_campaign"] = False
    scored.loc[selected_indices, "recommended_for_campaign"] = True
    scored["campaign_reason"] = np.where(
        scored["recommended_for_campaign"],
        "Selecionado para a estratégia comercial escolhida.",
        "Fora da capacidade ou da faixa da estratégia escolhida.",
    )
    return scored.reset_index(drop=True)
