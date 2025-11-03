from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import json

class Evaluator:
    """
    Classe para cálculo e salvamento das métricas de avaliação.
    """
    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        """Calcula e retorna todas as métricas relevantes."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    @staticmethod
    def save_metrics(metrics, path="model/metrics.json"):
        """Salva métricas em formato JSON."""
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
