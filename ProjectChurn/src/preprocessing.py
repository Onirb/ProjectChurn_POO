

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Classe responsável por tratar valores faltantes, codificar variáveis
    e dividir o dataset em treino e teste.
    """
    def __init__(self, target_col='churn', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.numeric_cols = None
        self.categorical_cols = None

    def encode_categorical(self, df):
        """Codifica colunas categóricas (Yes/No -> 1/0) e get_dummies."""
        df = df.copy()
        # Corrige variações de maiúsculas/minúsculas
        df[self.target_col] = (
            df[self.target_col]
            .str.lower()
            .replace({'yes': 1, 'no': 0})
            .infer_objects(copy=False)
        )
        df = df.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0}).infer_objects(copy=False)

        self.categorical_cols = df.select_dtypes(include='object').columns
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        return df

    def split(self, df):
        """Divide o dataset em treino e teste."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self.numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        return X_train, X_test, y_train, y_test

    def preprocess(self, df):
        """Pipeline completo de pré-processamento com prevenção de vazamento de dados."""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        df = self.encode_categorical(df)
        X_train, X_test, y_train, y_test = self.split(df)

        # Escala só após o split
        X_train[self.numeric_cols] = self.scaler.fit_transform(X_train[self.numeric_cols])
        X_test[self.numeric_cols] = self.scaler.transform(X_test[self.numeric_cols])

        return X_train, X_test, y_train, y_test
