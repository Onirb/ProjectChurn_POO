from ..logger_config import logger  # importa o logger configurado
import time
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from typing import Any

from ..data_contract import DatasetContractError, validate_feature_dataframe
from ..campaign_policy import (
    CampaignObjective,
    CapacityHandling,
    build_campaign_plan,
    list_capacity_choices,
    list_objectives,
)
from ..batch_scoring import BatchScoringError, score_customer_batch

# === Inicialização ===
app = FastAPI(
    title="Project Churn API",
    description="API de previsão de churn e planejamento comercial de retenção.",
    version="1.1.0"
)

from statistics import mean

# === Métricas internas ===
metrics = {
    "requests_total": 0,
    "requests_duration": []
}

# === Middleware para logging de requisições ===
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round(time.time() - start_time, 3)
    client_host = request.client.host

    # Atualiza métricas
    metrics["requests_total"] += 1
    metrics["requests_duration"].append(duration)

    logger.info(
        f"🌐 {request.method} {request.url.path} "
        f"de {client_host} | Status: {response.status_code} | Tempo: {duration}s"
    )
    return response




# === Middleware para capturar e logar exceções ===
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"🔥 Exceção não tratada: {str(e)}\n{error_trace}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erro interno do servidor. A equipe técnica foi notificada."},
        )

# === Carregar modelo salvo ===
model_path = Path(__file__).resolve().parents[2] / "model" / "churn_pipeline.joblib"
model = joblib.load(model_path) if model_path.exists() else None

# === Schema de entrada ===
class CustomerData(BaseModel):
    accountlength: float
    internationalplan: str
    voicemailplan: str
    numbervmailmessages: float
    totaldayminutes: float
    totaldaycalls: float
    totaldaycharge: float
    totaleveminutes: float
    totalevecalls: float
    totalevecharge: float
    totalnightminutes: float
    totalnightcalls: float
    totalnightcharge: float
    totalintlminutes: float
    totalintlcalls: float
    totalintlcharge: float
    numbercustomerservicecalls: float


class CampaignPlanRequest(BaseModel):
    target_customers: int = Field(
        ge=1,
        le=1000,
        description="Quantidade máxima de clientes que a campanha consegue atender.",
    )
    objective: CampaignObjective
    capacity_handling: CapacityHandling = CapacityHandling.KEEP_OFFER_PROFILE


class BatchScoreRequest(CampaignPlanRequest):
    customers: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=1000,
        description="Registros com customer_id ou id e as 17 variáveis do contrato.",
    )

# === Endpoint principal ===
@app.post("/predict")
def predict(data: CustomerData, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não foi treinado neste ambiente.")

    try:
        frame = validate_feature_dataframe(pd.DataFrame([data.model_dump()]))
        proba = model.predict_proba(frame)[0][1]
        pred = model.predict(frame)[0]
        result = {
            "churn_probability": round(float(proba), 3),
            "churn_prediction": int(pred),
            "interpretation": "Churn" if pred == 1 else "Não Churn"
        }
        # 🔹 Loga a predição com IP do cliente
        logger.info(f"🤖 Predição feita para {request.client.host}: {result}")
        return result

    except DatasetContractError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"❌ Erro na predição: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Falha ao gerar a predição.") from e


@app.get("/campaign/objectives")
def get_campaign_objectives():
    """Opções comerciais para preencher os cards da interface."""
    return {"objectives": list_objectives(), "capacity_choices": list_capacity_choices()}


@app.post("/campaign/plan")
def create_campaign_plan(data: CampaignPlanRequest):
    """Traduz meta e tipo de ação em um plano de campanha rastreável."""
    plan = build_campaign_plan(data.objective, data.target_customers, data.capacity_handling)
    logger.info(
        "Plano de campanha solicitado: objetivo=%s meta=%s contatos_sugeridos=%s",
        data.objective.value,
        data.target_customers,
        plan["suggested_contacts"],
    )
    return plan


@app.post("/campaign/score")
def score_campaign_batch(data: BatchScoreRequest):
    """Pontua localmente um lote e devolve a lista priorizada para a campanha."""
    plan = build_campaign_plan(data.objective, data.target_customers, data.capacity_handling)
    try:
        ranking = score_customer_batch(pd.DataFrame(data.customers), plan)
    except BatchScoringError as error:
        message = str(error)
        status_code = 503 if "Artefato de demonstração ausente" in message else 422
        raise HTTPException(status_code=status_code, detail=message) from error
    selected = ranking.loc[ranking["recommended_for_campaign"]]
    logger.info(
        "Lote de campanha pontuado: registros=%s selecionados=%s objetivo=%s",
        len(ranking),
        len(selected),
        data.objective.value,
    )
    return {
        "plan": plan,
        "received_customers": len(ranking),
        "selected_customers": len(selected),
        "ranking": ranking.to_dict(orient="records"),
    }

@app.get("/metrics")
def get_metrics():
    total = metrics["requests_total"]
    durations = metrics["requests_duration"]
    avg_time = round(mean(durations), 3) if durations else 0

    logger.info(f"📈 Métricas consultadas: {total} requisições, tempo médio {avg_time}s")

    return {
        "total_requests": total,
        "average_response_time": avg_time,
        "uptime_status": "🟢 API operacional"
    }


@app.get("/health")
def health_check():
    """Verificação simples usada por Docker e pela demonstração local."""
    return {"status": "ok", "service": "project-churn-api", "model_loaded": model is not None}
#@app.get("/force-error")
#def force_error():
#    logger.info("🧨 Testando exceção forçada...")
#    raise ValueError("Erro de teste intencional para validar o middleware.")

# === Endpoint de teste ===
@app.get("/")
def root():
    return {"message": "API de previsão de churn ativa! 🚀"}
