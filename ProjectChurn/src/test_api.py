import requests
import json

# URL da API local (container Docker)
API_URL = "http://127.0.0.1:8000/predict"

# Dados de exemplo no mesmo formato da API
sample = {
    "accountlength": 120,
    "internationalplan": "no",
    "voicemailplan": "yes",
    "numbervmailmessages": 20,
    "totaldayminutes": 250,
    "totaldaycalls": 100,
    "totaldaycharge": 40,
    "totaleveminutes": 180,
    "totalevecalls": 90,
    "totalevecharge": 15,
    "totalnightminutes": 200,
    "totalnightcalls": 80,
    "totalnightcharge": 9,
    "totalintlminutes": 10,
    "totalintlcalls": 3,
    "totalintlcharge": 2.5,
    "numbercustomerservicecalls": 2
}

# Faz a requisição POST
response = requests.post(API_URL, json=sample)

# Exibe os resultados
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=4, ensure_ascii=False))
else:
    print(f"Erro {response.status_code}: {response.text}")
