"""Painel local de demonstração do ProjectChurn_POO.

Execute com: streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.batch_scoring import BatchScoringError, score_customer_batch
from src.campaign_policy import (
    CapacityHandling,
    CampaignObjective,
    build_campaign_plan,
    list_capacity_choices,
    list_objectives,
)


st.set_page_config(page_title="Churn Campaign Studio", page_icon="📉", layout="wide")

OBJECTIVES = {item["id"]: item for item in list_objectives()}
CAPACITY_CHOICES = {item["id"]: item for item in list_capacity_choices()}
DEMO_BATCH = Path("data/demo/customer_batch_demo.csv")


def objective_label(identifier: str) -> str:
    return OBJECTIVES[identifier]["title"]


def handling_label(identifier: str) -> str:
    return CAPACITY_CHOICES[identifier]["title"]


st.title("Churn Campaign Studio")
st.caption("Demonstração local para planejar e priorizar uma campanha de retenção.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("1. Escolha a ação")
    selected_objective = st.radio(
        "Que tipo de ação será usada?",
        options=list(OBJECTIVES),
        format_func=objective_label,
        horizontal=False,
    )
    objective = OBJECTIVES[selected_objective]
    st.info(f"**{objective['title']}** — {objective['description']}")

with right:
    st.subheader("2. Informe a capacidade")
    target_customers = int(
        st.number_input(
            "Quantos clientes sua campanha consegue atender?",
            min_value=1,
            max_value=1000,
            value=250,
            step=25,
        )
    )

initial_plan = build_campaign_plan(CampaignObjective(selected_objective), target_customers)
selected_handling = CapacityHandling.KEEP_OFFER_PROFILE.value
if not initial_plan["fits_original_profile"]:
    st.warning("A meta ultrapassa a faixa mais seletiva dessa ação.")
    selected_handling = st.radio(
        "Como você prefere preencher a capacidade?",
        options=list(CAPACITY_CHOICES),
        format_func=handling_label,
        horizontal=False,
    )
    st.caption(CAPACITY_CHOICES[selected_handling]["description"])

plan = build_campaign_plan(
    CampaignObjective(selected_objective),
    target_customers,
    CapacityHandling(selected_handling),
)

st.subheader("3. Plano sugerido")
plan_columns = st.columns(3)
plan_columns[0].metric("Meta da campanha", plan["target_customers"])
plan_columns[1].metric("Contatos sugeridos", plan["suggested_contacts"])
plan_columns[2].metric("Estratégia", plan["strategy"])
st.write(plan["planning_note"])

st.divider()
st.subheader("4. Envie ou teste um lote")
uploaded = st.file_uploader("CSV de clientes", type=["csv"])
use_demo = st.checkbox("Usar lote sintético incluído no projeto", value=uploaded is None)

batch: pd.DataFrame | None = None
if uploaded is not None:
    batch = pd.read_csv(uploaded)
elif use_demo and DEMO_BATCH.exists():
    batch = pd.read_csv(DEMO_BATCH)

if batch is None:
    st.info("Envie um CSV com `customer_id` (ou `id`) e as 17 variáveis do contrato.")
else:
    st.caption(f"Lote carregado: {len(batch)} clientes. Os dados são usados apenas nesta sessão local.")
    with st.expander("Visualizar colunas do lote"):
        st.dataframe(batch.head(), use_container_width=True)

    if st.button("Priorizar clientes", type="primary"):
        try:
            ranking = score_customer_batch(batch, plan)
        except BatchScoringError as error:
            st.error(str(error))
        else:
            selected = ranking.loc[ranking["recommended_for_campaign"]].copy()
            c1, c2 = st.columns(2)
            c1.metric("Clientes analisados", len(ranking))
            c2.metric("Clientes priorizados", len(selected))
            st.success("Lista priorizada gerada localmente. Nenhum dado foi armazenado ou enviado.")
            st.dataframe(ranking, use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar lista priorizada em CSV",
                data=ranking.to_csv(index=False).encode("utf-8"),
                file_name="clientes_priorizados.csv",
                mime="text/csv",
            )

with st.expander("Sobre esta demonstração"):
    st.markdown(
        """
        Este painel é um artefato de portfólio local. Ele separa a decisão
        comercial da implementação estatística: a pessoa escolhe uma ação e
        uma capacidade, enquanto o sistema usa a política validada em testes.

        Não use dados reais de clientes nesta demonstração. Os artefatos de
        modelo são locais e a tela não envia ações automaticamente.
        """
    )
