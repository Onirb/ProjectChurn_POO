"""Contrato de dados para o benchmark de churn telecom.

O projeto aceita variantes conhecidas do dataset (Kaggle e legado), mas o
modelo sempre recebe as mesmas 17 variáveis, com nomes canônicos.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd


TARGET_COLUMN: Final = "churn"
FEATURE_COLUMNS: Final = (
    "accountlength",
    "internationalplan",
    "voicemailplan",
    "numbervmailmessages",
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
    "numbercustomerservicecalls",
)
NUMERIC_COLUMNS: Final = tuple(
    column for column in FEATURE_COLUMNS if column not in {"internationalplan", "voicemailplan"}
)
CATEGORICAL_COLUMNS: Final = ("internationalplan", "voicemailplan")


class DatasetContractError(ValueError):
    """O arquivo não satisfaz o contrato mínimo do projeto."""


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_ALIASES: Final = {
    "accountlength": "accountlength",
    "internationalplan": "internationalplan",
    "voicemailplan": "voicemailplan",
    "numbervmailmessages": "numbervmailmessages",
    "totaldayminutes": "totaldayminutes",
    "totaldaycalls": "totaldaycalls",
    "totaldaycharge": "totaldaycharge",
    "totaleveminutes": "totaleveminutes",
    "totalevecalls": "totalevecalls",
    "totalevecharge": "totalevecharge",
    "totalnightminutes": "totalnightminutes",
    "totalnightcalls": "totalnightcalls",
    "totalnightcharge": "totalnightcharge",
    "totalintlminutes": "totalintlminutes",
    "totalintlcalls": "totalintlcalls",
    "totalintlcharge": "totalintlcharge",
    "numbercustomerservicecalls": "numbercustomerservicecalls",
    "customerservicecalls": "numbercustomerservicecalls",
    "churn": "churn",
}
_OPTIONAL_SOURCE_COLUMNS: Final = {"state", "areacode", "phonenumber", "id", "customerid"}
_BINARY_VALUES: Final = {
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
}


def canonicalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Converte nomes com espaços/sublinhados para o esquema canônico."""
    frame = dataframe.copy()
    rename_map: dict[object, str] = {}
    unknown_columns: list[str] = []

    for original in frame.columns:
        normalized = _normalized_name(original)
        if normalized in _ALIASES:
            rename_map[original] = _ALIASES[normalized]
        elif normalized in _OPTIONAL_SOURCE_COLUMNS:
            rename_map[original] = normalized
        else:
            unknown_columns.append(str(original))

    if unknown_columns:
        raise DatasetContractError(
            "Colunas não reconhecidas pelo contrato: " + ", ".join(sorted(unknown_columns))
        )

    frame = frame.rename(columns=rename_map)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetContractError(f"Colunas duplicadas após normalização: {duplicates}")

    return frame.drop(columns=list(_OPTIONAL_SOURCE_COLUMNS), errors="ignore")


def _binary_series(series: pd.Series, column: str, *, allow_missing: bool) -> pd.Series:
    missing = series.isna()
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(_BINARY_VALUES)
    invalid = normalized[~missing & mapped.isna()].unique().tolist()
    if invalid:
        raise DatasetContractError(f"Valores inválidos em '{column}': {invalid}")
    if not allow_missing and mapped.isna().any():
        raise DatasetContractError(f"Valores ausentes em '{column}'.")
    if allow_missing:
        return mapped.astype("float64")
    return mapped.astype("int64")


def validate_feature_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Valida uma entrada de inferência sem exigir a coluna alvo."""
    frame = canonicalize_columns(dataframe)
    required = set(FEATURE_COLUMNS)
    missing = required - set(frame.columns)
    extras = set(frame.columns) - required

    if missing:
        raise DatasetContractError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))
    if extras:
        raise DatasetContractError("Colunas extras após normalização: " + ", ".join(sorted(extras)))
    if frame.empty:
        raise DatasetContractError("O dataset não contém linhas.")

    for column in NUMERIC_COLUMNS:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original[original.notna() & converted.isna()].unique().tolist()
        if invalid:
            raise DatasetContractError(f"Valores não numéricos em '{column}': {invalid}")
        frame[column] = converted

    for column in CATEGORICAL_COLUMNS:
        frame[column] = _binary_series(frame[column], column, allow_missing=True)

    return frame.loc[:, list(FEATURE_COLUMNS)]


def validate_churn_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma cópia canônica, completa e pronta para divisão treino/teste."""
    frame = canonicalize_columns(dataframe)
    if TARGET_COLUMN not in frame.columns:
        raise DatasetContractError("Colunas obrigatórias ausentes: churn")
    target = _binary_series(frame[TARGET_COLUMN], TARGET_COLUMN, allow_missing=False)
    features = validate_feature_dataframe(frame.drop(columns=[TARGET_COLUMN]))

    if target.nunique() != 2:
        raise DatasetContractError("O alvo 'churn' deve conter as duas classes (0 e 1).")

    return pd.concat([features, target.rename(TARGET_COLUMN)], axis=1)


def dataset_metadata(dataframe: pd.DataFrame) -> dict[str, object]:
    """Metadados simples para registrar a versão local da base processada."""
    validated = validate_churn_dataframe(dataframe)
    return {
        "rows": len(validated),
        "columns": list(validated.columns),
        "churn_rate": round(float(validated[TARGET_COLUMN].mean()), 6),
        "class_counts": {
            str(label): int(count)
            for label, count in validated[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "missing_values": {
            column: int(count)
            for column, count in validated.isna().sum().items()
            if count
        },
    }
