from ..logger_config import logger  # importa o logger configurado
import time
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# === Inicialização ===
app = FastAPI(
    title="Project Churn API",
    description="API para predição de churn de clientes baseada em RandomForest.",
    version="1.0.0"
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
model_path = "model/churn_model.joblib"
model = joblib.load(model_path)

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

# === Endpoint principal ===
@app.post("/predict")
def predict(data: CustomerData, request: Request):
    df = pd.DataFrame([data.dict()])
    df.replace({"yes": 1, "no": 0, "Yes": 1, "No": 0}, inplace=True)

    try:
        proba = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]
        result = {
            "churn_probability": round(float(proba), 3),
            "churn_prediction": int(pred),
            "interpretation": "Churn" if pred == 1 else "Não Churn"
        }
        # 🔹 Loga a predição com IP do cliente
        logger.info(f"🤖 Predição feita para {request.client.host}: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Erro na predição: {str(e)}", exc_info=True)
        return {"erro": str(e)}

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
#@app.get("/force-error")
#def force_error():
#    logger.info("🧨 Testando exceção forçada...")
#    raise ValueError("Erro de teste intencional para validar o middleware.")

# === Endpoint de teste ===
@app.get("/")
def root():
    return {"message": "API de previsão de churn ativa! 🚀"}
