import pandas as pd
import pytest

from src.data_contract import DatasetContractError, FEATURE_COLUMNS, validate_churn_dataframe


def make_raw_frame() -> pd.DataFrame:
    rows = []
    for churn, plan in [("False", "no"), ("True", "yes")]:
        row = {"state": "KS", "area code": 408, "churn": churn}
        for index, column in enumerate(FEATURE_COLUMNS):
            if column in {"internationalplan", "voicemailplan"}:
                row[column] = plan
            else:
                row[column] = index + 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_normalizes_kaggle_style_columns_and_removes_context_columns():
    raw = make_raw_frame().rename(columns={"accountlength": "account length"})
    validated = validate_churn_dataframe(raw)

    assert validated.columns.tolist() == [*FEATURE_COLUMNS, "churn"]
    assert validated["internationalplan"].tolist() == [0, 1]
    assert validated["churn"].tolist() == [0, 1]


def test_rejects_missing_required_column():
    raw = make_raw_frame().drop(columns=["totaldayminutes"])

    with pytest.raises(DatasetContractError, match="ausentes"):
        validate_churn_dataframe(raw)


def test_preserves_missing_feature_values_for_the_training_pipeline():
    raw = make_raw_frame()
    raw.loc[0, "accountlength"] = None
    raw.loc[0, "internationalplan"] = None

    validated = validate_churn_dataframe(raw)

    assert validated["accountlength"].isna().sum() == 1
    assert validated["internationalplan"].isna().sum() == 1


def test_accepts_binary_values_reloaded_from_csv_as_floats():
    raw = make_raw_frame()
    raw["internationalplan"] = [0.0, 1.0]
    raw["voicemailplan"] = [1.0, 0.0]

    validated = validate_churn_dataframe(raw)

    assert validated["internationalplan"].tolist() == [0, 1]
    assert validated["voicemailplan"].tolist() == [1, 0]
