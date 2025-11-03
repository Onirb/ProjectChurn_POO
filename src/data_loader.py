import pandas as pd

class DataLoader:
    """
    Classe responsável por carregar e inspecionar os dados.
    """
    def __init__(self, file_path, drop_cols=None):
        self.file_path = file_path
        self.drop_cols = drop_cols or []
        self.data = None

    def load(self):
        """Carrega o dataset em um DataFrame."""
        df = pd.read_csv(self.file_path)
        if self.drop_cols:
            df = df.drop(columns=self.drop_cols, errors="ignore")
        self.data = df
        return df

    def info(self):
        """Retorna informações básicas do dataset."""
        if self.data is not None:
            return self.data.info()
        raise ValueError("Nenhum dado carregado. Execute .load() primeiro.")
