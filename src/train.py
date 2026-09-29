import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import DataLoader
from .data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from .models.churn_model import ChurnModel
from .evaluation import Evaluator
from .logger_config import logger  # 🔹 importa o logger


def main():
    mlflow.set_experiment("ProjectChurn_POO")

    try:
        with mlflow.start_run():
            logger.info("📥 Carregando dados...")
            dl = DataLoader("data/processed/churn.csv")
            df = dl.load()

            logger.info("🧹 Pré-processando dados...")
            X = df.loc[:, list(FEATURE_COLUMNS)]
            y = df[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info("🤖 Treinando modelo RandomForest...")
            model = ChurnModel(n_estimators=100)
            model.fit(X_train, y_train)

            logger.info("📊 Avaliando modelo...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = Evaluator.evaluate(y_test, y_pred, y_proba)
            logger.info(f"📊 Avaliação concluída: {metrics}")

            logger.info("💾 Salvando modelo e métricas...")
            os.makedirs("model", exist_ok=True)
            model.save("model/churn_pipeline.joblib")
            Evaluator.save_metrics(metrics)

            logger.info("📈 Registrando métricas no MLflow...")
            mlflow.log_params({"n_estimators": 100})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not pd.isna(v):
                    mlflow.log_metric(k, v)
            mlflow.log_artifacts("model")

            logger.info("✅ Execução concluída com sucesso.")

    except Exception as e:
        logger.error(f"❌ Erro durante execução: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
