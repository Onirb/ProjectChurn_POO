from fastapi.testclient import TestClient

from src.api import app as api_module


class StubPipeline:
    def predict_proba(self, frame):
        assert len(frame) == 1
        return [[0.8, 0.2]]

    def predict(self, frame):
        assert len(frame) == 1
        return [0]


PAYLOAD = {
    "accountlength": 120,
    "internationalplan": "no",
    "voicemailplan": "yes",
    "numbervmailmessages": 20,
    "totaldayminutes": 250,
    "totaldaycalls": 100,
    "totaldaycharge": 40,
    "totaleveminutes": 180,
    "totalevecalls": 90,
    "totalevecharge": 15,
    "totalnightminutes": 200,
    "totalnightcalls": 80,
    "totalnightcharge": 9,
    "totalintlminutes": 10,
    "totalintlcalls": 3,
    "totalintlcharge": 2.5,
    "numbercustomerservicecalls": 2,
}


def test_predict_uses_validated_raw_features(monkeypatch):
    monkeypatch.setattr(api_module, "model", StubPipeline())
    client = TestClient(api_module.app)

    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.2
    assert response.json()["churn_prediction"] == 0


def test_predict_returns_service_unavailable_without_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)
    client = TestClient(api_module.app)

    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 503


def test_campaign_objectives_are_business_facing():
    client = TestClient(api_module.app)

    response = client.get("/campaign/objectives")

    assert response.status_code == 200
    assert len(response.json()["objectives"]) == 4
    assert len(response.json()["capacity_choices"]) == 3
    assert response.json()["objectives"][0]["id"] == "oferta_premium"


def test_campaign_plan_respects_capacity_and_does_not_require_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)
    client = TestClient(api_module.app)

    response = client.post(
        "/campaign/plan",
        json={"target_customers": 700, "objective": "oferta_premium"},
    )

    assert response.status_code == 200
    plan = response.json()
    assert plan["suggested_contacts"] == 501
    assert plan["can_fill_target"] is False
    assert plan["technical_policy"]["model_is_not_promoted_to_api"] is True


def test_campaign_plan_can_fill_capacity_with_standard_campaign():
    client = TestClient(api_module.app)

    response = client.post(
        "/campaign/plan",
        json={
            "target_customers": 700,
            "objective": "oferta_premium",
            "capacity_handling": "completar_com_campanha_padrao",
        },
    )

    assert response.status_code == 200
    plan = response.json()
    assert plan["suggested_contacts"] == 700
    assert plan["can_fill_target"] is True
    assert plan["fits_original_profile"] is False
    assert plan["strategy"] == "Campanha padrão de retenção"
    assert plan["technical_policy"]["selection_mode"] == "top_k_por_capacidade"


def test_campaign_score_uses_batch_scoring_service(monkeypatch):
    def fake_score(_, __):
        import pandas as pd

        return pd.DataFrame(
            {
                "customer_id": ["demo-1"],
                "churn_probability": [0.8],
                "campaign_rank": [1],
                "recommended_for_campaign": [True],
                "campaign_reason": ["Selecionado"],
            }
        )

    monkeypatch.setattr(api_module, "score_customer_batch", fake_score)
    client = TestClient(api_module.app)
    response = client.post(
        "/campaign/score",
        json={
            "target_customers": 1,
            "objective": "campanha_padrao",
            "customers": [{"customer_id": "demo-1"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_customers"] == 1


def test_health_check_does_not_require_model(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False
