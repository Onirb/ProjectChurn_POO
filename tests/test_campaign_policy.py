from src.campaign_policy import (
    CampaignObjective,
    CapacityHandling,
    build_campaign_plan,
    list_capacity_choices,
    list_objectives,
)


def test_objective_cards_do_not_expose_metrics_to_business_user():
    cards = list_objectives()

    assert len(cards) == 4
    assert all("precision" not in card for card in cards)
    assert all("recall" not in card for card in cards)


def test_premium_plan_can_leave_capacity_unfilled_to_preserve_strategy():
    plan = build_campaign_plan(CampaignObjective.PREMIUM_OFFER, 700)

    assert plan["suggested_contacts"] == 501
    assert plan["can_fill_target"] is False


def test_standard_completion_fills_the_requested_capacity():
    plan = build_campaign_plan(
        CampaignObjective.PREMIUM_OFFER,
        700,
        CapacityHandling.COMPLETE_WITH_STANDARD,
    )

    assert plan["suggested_contacts"] == 700
    assert plan["technical_policy"]["candidate_model"] == "random_forest"
    assert plan["technical_policy"]["threshold"] is None


def test_capacity_choices_explain_three_business_paths():
    choices = list_capacity_choices()

    assert len(choices) == 3
    assert choices[0]["id"] == "manter_perfil_da_oferta"
