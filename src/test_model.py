import pandas as pd
import joblib

# === 1. Carregar modelo salvo ===
model_path = "model/churn_model.joblib"
model = joblib.load(model_path)
print("✅ Modelo carregado com sucesso!")

# === 2. Carregar dataset e selecionar amostra ===
data_path = "data/churn.csv"
df = pd.read_csv(data_path)

# Seleciona uma linha aleatória
sample = df.sample(1, random_state=42)

# Guarda o valor real (ground truth)
real_value = sample["churn"].values[0]

# Remove a coluna alvo
X = sample.drop(columns=["churn"])

# Converte "Yes"/"No" para 1/0 se ainda existirem
X = X.replace({"yes": 1, "no": 0, "Yes": 1, "No": 0}).infer_objects(copy=False)


# === 3. Fazer predição ===
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0][1]

print("\n📊 Predição do modelo:")
print(f"  Probabilidade de churn: {proba:.2f}")
print(f"  Classe prevista: {'Churn' if pred == 1 else 'Não Churn'}")
print(f"  Valor real: {'Churn' if real_value == 1 else 'Não Churn'}")
