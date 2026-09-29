import pandas as pd

from src.data_contract import FEATURE_COLUMNS
from src.legacy_benchmark import historical_d3
from src.training_pipeline import USAGE_COLUMNS


def test_historical_d3_imputes_and_scales_usage():
    rows = []
    for index in range(4):
        row = {column: float(index + 1) for column in FEATURE_COLUMNS}
        row["internationalplan"] = index % 2
        row["voicemailplan"] = (index + 1) % 2
        row["churn"] = index % 2
        rows.append(row)
    data = pd.DataFrame(rows)
    data.loc[0, "totaldayminutes"] = None
    data.loc[1, "numbervmailmessages"] = None

    transformed = historical_d3(data)

    assert transformed["totaldayminutes"].isna().sum() == 0
    assert transformed["numbervmailmessages"].isna().sum() == 0
    assert transformed[USAGE_COLUMNS].min().min() >= 0
    assert transformed[USAGE_COLUMNS].max().max() <= 1
