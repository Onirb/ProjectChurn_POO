import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

class ChurnModel:
    """
    Classe que encapsula o modelo de Churn (RandomForest).
    """
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.trained = False
        self.metrics = {}

    def fit(self, X_train, y_train):
        """Treina o modelo."""
        self.model.fit(X_train, y_train)
        self.trained = True

    def predict(self, X_test):
        """Retorna as predições (0/1)."""
        if not self.trained:
            raise ValueError("Modelo não treinado. Execute .fit() primeiro.")
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Retorna probabilidades de churn."""
        if not self.trained:
            raise ValueError("Modelo não treinado. Execute .fit() primeiro.")
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate_auc(self, X_test, y_test):
        """Calcula ROC-AUC do modelo."""
        preds = self.predict_proba(X_test)
        auc = roc_auc_score(y_test, preds)
        self.metrics["roc_auc"] = auc
        return auc

    def save(self, path="model/churn_model.joblib"):
        """Salva o modelo treinado."""
        joblib.dump(self.model, path)

    def load(self, path="model/churn_model.joblib"):
        """Carrega modelo salvo."""
        self.model = joblib.load(path)
        self.trained = True
