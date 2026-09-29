"""Tradução de escolhas comerciais em políticas técnicas já validadas.

Este módulo não reclassifica clientes. Ele orienta a campanha antes do envio de
um lote para pontuação e mantém a decisão compreensível para pessoas de negócio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CampaignObjective(StrEnum):
    """Tipos de ação apresentados na interface, sem expor métricas de ML."""

    PREMIUM_OFFER = "oferta_premium"
    HUMAN_CONTACT = "contato_humano"
    STANDARD_CAMPAIGN = "campanha_padrao"
    BROAD_OUTREACH = "comunicacao_ampla"


class CapacityHandling(StrEnum):
    """Como proceder quando a meta excede a faixa mais seletiva."""

    KEEP_OFFER_PROFILE = "manter_perfil_da_oferta"
    COMPLETE_WITH_STANDARD = "completar_com_campanha_padrao"
    FILL_WITH_BROADER_LIST = "preencher_com_lista_ampla"


@dataclass(frozen=True)
class CampaignPolicy:
    objective: CampaignObjective
    title: str
    action_description: str
    commercial_reason: str
    model_name: str
    threshold: float
    validated_qualified_pool: int


# Contagens obtidas em 4.000 clientes de validação OOF, no relatório
# runs/policy_frontier_v1. Elas representam referências de planejamento, não
# previsão de volume para uma base futura de tamanho ou perfil diferente.
POLICIES = {
    CampaignObjective.PREMIUM_OFFER: CampaignPolicy(
        objective=CampaignObjective.PREMIUM_OFFER,
        title="Oferta premium ou desconto relevante",
        action_description="Use quando cada contato ou benefício tem custo alto.",
        commercial_reason="Prioriza uma lista curta de pessoas com sinais muito fortes de cancelamento.",
        model_name="random_forest",
        threshold=0.404,
        validated_qualified_pool=501,
    ),
    CampaignObjective.HUMAN_CONTACT: CampaignPolicy(
        objective=CampaignObjective.HUMAN_CONTACT,
        title="Contato humano proativo",
        action_description="Use para ligações, atendimento especializado ou follow-up manual.",
        commercial_reason="Equilibra qualidade da lista e aproveitamento da equipe de atendimento.",
        model_name="random_forest",
        threshold=0.292,
        validated_qualified_pool=586,
    ),
    CampaignObjective.STANDARD_CAMPAIGN: CampaignPolicy(
        objective=CampaignObjective.STANDARD_CAMPAIGN,
        title="Campanha padrão de retenção",
        action_description="Use para abordagem regular com custo moderado por cliente.",
        commercial_reason="Amplia o alcance sem transformar a lista em uma comunicação genérica.",
        model_name="random_forest",
        threshold=0.216,
        validated_qualified_pool=688,
    ),
    CampaignObjective.BROAD_OUTREACH: CampaignPolicy(
        objective=CampaignObjective.BROAD_OUTREACH,
        title="Comunicação ampla e de baixo custo",
        action_description="Use para e-mail, push ou outra ação escalável e barata.",
        commercial_reason="Busca alcançar a maior parte dos clientes com risco, aceitando uma lista mais ampla.",
        model_name="hist_gradient_boosting_tuned",
        threshold=0.01445,
        validated_qualified_pool=978,
    ),
}


def list_objectives() -> list[dict[str, str]]:
    """Cards prontos para a futura interface comercial."""
    return [
        {
            "id": policy.objective.value,
            "title": policy.title,
            "description": policy.action_description,
            "reason": policy.commercial_reason,
        }
        for policy in POLICIES.values()
    ]


def list_capacity_choices() -> list[dict[str, str]]:
    """Escolhas mostradas somente quando a meta supera a lista mais seletiva."""
    return [
        {
            "id": CapacityHandling.KEEP_OFFER_PROFILE.value,
            "title": "Manter o perfil da oferta",
            "description": "Use menos contatos quando necessário para preservar uma lista mais seletiva.",
        },
        {
            "id": CapacityHandling.COMPLETE_WITH_STANDARD.value,
            "title": "Completar com campanha padrão",
            "description": "Preencha a meta com uma ação de custo ou benefício moderado.",
        },
        {
            "id": CapacityHandling.FILL_WITH_BROADER_LIST.value,
            "title": "Usar toda a capacidade",
            "description": "Preencha a meta com uma lista mais ampla, adequada a ações de baixo custo.",
        },
    ]


def build_campaign_plan(
    objective: CampaignObjective,
    target_customers: int,
    capacity_handling: CapacityHandling = CapacityHandling.KEEP_OFFER_PROFILE,
) -> dict[str, object]:
    """Retorna um plano comercial sem exigir que o usuário conheça métricas."""
    policy = POLICIES[objective]
    fits_original_profile = target_customers <= policy.validated_qualified_pool
    applied_handling = capacity_handling
    selected_policy = policy
    selection_mode = "lista_qualificada"

    if fits_original_profile:
        suggested_contacts = target_customers
        note = "A meta cabe na faixa validada para esta estratégia."
    elif capacity_handling == CapacityHandling.KEEP_OFFER_PROFILE:
        suggested_contacts = policy.validated_qualified_pool
        note = (
            "A meta excede a faixa seletiva desta oferta. O plano mantém apenas os contatos "
            "que preservam o perfil da ação."
        )
    elif capacity_handling == CapacityHandling.COMPLETE_WITH_STANDARD:
        suggested_contacts = target_customers
        selected_policy = POLICIES[CampaignObjective.STANDARD_CAMPAIGN]
        selection_mode = "top_k_por_capacidade"
        note = (
            "A meta será preenchida com uma campanha padrão. A oferta ou abordagem deve ter "
            "custo moderado, pois a lista será mais ampla que o perfil premium."
        )
    else:
        suggested_contacts = target_customers
        selected_policy = POLICIES[CampaignObjective.BROAD_OUTREACH]
        selection_mode = "top_k_por_capacidade"
        note = (
            "A meta será preenchida com uma lista ampla. Esta opção é indicada somente para "
            "ações escaláveis e de baixo custo, como e-mail ou push."
        )

    return {
        "objective": policy.objective.value,
        "strategy": selected_policy.title,
        "action_description": selected_policy.action_description,
        "commercial_reason": selected_policy.commercial_reason,
        "target_customers": target_customers,
        "suggested_contacts": suggested_contacts,
        "can_fill_target": suggested_contacts == target_customers,
        "fits_original_profile": fits_original_profile,
        "capacity_handling": applied_handling.value,
        "planning_note": note,
        "status": "plano_validado_localmente",
        "scope_note": "Estimativa baseada na validação local; a lista real exige pontuar o lote de clientes.",
        "technical_policy": {
            "candidate_model": selected_policy.model_name,
            "threshold": selected_policy.threshold if selection_mode == "lista_qualificada" else None,
            "selection_mode": selection_mode,
            "model_is_not_promoted_to_api": True,
        },
    }
