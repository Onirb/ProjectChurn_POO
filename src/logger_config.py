import logging
import os

# === Diretório de logs ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# === Configuração base ===
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Cria um logger padrão reutilizável
logger = logging.getLogger("ProjectChurn")
