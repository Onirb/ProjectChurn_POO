import pandas as pd

from src.data_contract import FEATURE_COLUMNS
from src.training_pipeline import build_churn_pipeline


def make_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    labels = []
    for index in range(20):
        row = {column: float(index + 1) for column in FEATURE_COLUMNS}
        row["internationalplan"] = index % 2
        row["voicemailplan"] = (index + 1) % 2
        rows.append(row)
        labels.append(index % 2)
    rows[0]["accountlength"] = None
    rows[1]["internationalplan"] = None
    return pd.DataFrame(rows), pd.Series(labels)


def test_pipeline_handles_missing_values_and_predicts_from_raw_features():
    features, target = make_training_frame()
    pipeline = build_churn_pipeline(n_estimators=10, random_state=42)
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)

    assert probabilities.shape == (20, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
