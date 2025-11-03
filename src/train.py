import os
import mlflow
import pandas as pd
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .models.churn_model import ChurnModel
from .evaluation import Evaluator
from .logger_config import logger  # 🔹 importa o logger


def main():
    mlflow.set_experiment("ProjectChurn_POO")

    try:
        with mlflow.start_run():
            logger.info("📥 Carregando dados...")
            dl = DataLoader("data/churn.csv")
            df = dl.load()

            logger.info("🧹 Pré-processando dados...")
            pre = Preprocessor(target_col="churn")
            X_train, X_test, y_train, y_test = pre.preprocess(df)

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
            model.save("model/churn_model.joblib")
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

if __name__ == "__main__":
    main()
