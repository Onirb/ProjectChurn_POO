# === Etapa base ===
FROM python:3.12-slim

# === Diretório de trabalho dentro do container ===
WORKDIR /app

# === Dependências mínimas para servir a API ===
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# === Copiar o restante do projeto ===
COPY . .

# === Variáveis de ambiente ===
ENV PYTHONUNBUFFERED=1

# === Expor a porta padrão da API ===
EXPOSE 8000

# A imagem só é considerada saudável quando a API responde localmente.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

# === Comando de inicialização ===
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
