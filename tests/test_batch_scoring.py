import pandas as pd

from src.batch_scoring import score_customer_batch


class StubPipeline:
    def predict_proba(self, frame):
        assert len(frame) == 3
        return [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]


def make_batch():
    rows = []
    for index in range(3):
        rows.append(
            {
                "customer_id": f"C-{index}",
                "accountlength": 100 + index,
                "internationalplan": "yes" if index else "no",
                "voicemailplan": "no",
                "numbervmailmessages": 0,
                "totaldayminutes": 200 + index,
                "totaldaycalls": 100,
                "totaldaycharge": 30,
                "totaleveminutes": 180,
                "totalevecalls": 90,
                "totalevecharge": 15,
                "totalnightminutes": 220,
                "totalnightcalls": 100,
                "totalnightcharge": 10,
                "totalintlminutes": 10,
                "totalintlcalls": 3,
                "totalintlcharge": 2.7,
                "numbercustomerservicecalls": 2,
            }
        )
    return pd.DataFrame(rows)


def test_scores_and_ranks_batch_by_probability():
    plan = {"suggested_contacts": 2, "technical_policy": {"candidate_model": "random_forest", "threshold": None}}

    result = score_customer_batch(make_batch(), plan, pipeline=StubPipeline())

    assert result["customer_id"].tolist() == ["C-2", "C-1", "C-0"]
    assert result["recommended_for_campaign"].sum() == 2


def test_strict_plan_can_select_fewer_than_capacity():
    plan = {"suggested_contacts": 3, "technical_policy": {"candidate_model": "random_forest", "threshold": 0.7}}

    result = score_customer_batch(make_batch(), plan, pipeline=StubPipeline())

    assert result["recommended_for_campaign"].sum() == 1
