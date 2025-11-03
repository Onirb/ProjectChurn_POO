# === Etapa base ===
FROM python:3.10-slim

# === Diretório de trabalho dentro do container ===
WORKDIR /app

# === Copiar dependências ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Copiar o restante do projeto ===
COPY . .

# === Variáveis de ambiente ===
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"

# === Expor a porta padrão da API ===
EXPOSE 8000

# === Comando de inicialização ===
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
