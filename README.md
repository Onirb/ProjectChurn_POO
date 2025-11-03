# ProjectChurn_POO 🚀  
**Customer Churn Prediction API** built with **Python 3.13**, **FastAPI**, and **Scikit-Learn**, featuring a fully object-oriented pipeline, structured logging, MLflow tracking, and Dockerized deployment.

![Python](https://img.shields.io/badge/Python-3.13.5-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.3.3-orange)
![NumPy](https://img.shields.io/badge/numpy-2.3.4-lightblue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.120.4-brightgreen)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.38.0-purple)
![MLflow](https://img.shields.io/badge/MLflow-3.5.1-lightgrey)
![Docker](https://img.shields.io/badge/Containerized-Yes-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

### 🧠 Overview
`ProjectChurn_POO` is a **production-ready churn prediction system**, redesigned from the original *ProjectChurn* to adopt:
- A **clean OOP architecture** for scalability and maintainability  
- **MLflow integration** for experiment tracking and reproducibility  
- **FastAPI service** exposing a `/predict` endpoint for real-time inference  
- **Docker containerization**, ensuring cross-environment deployment  

This project demonstrates best practices in **MLOps**, **software engineering for ML**, and **data-driven business strategy** for customer retention.

---

### ⚙️ Environment Summary
| Tool | Version |
|------|----------|
| Python | 3.13.5 |
| Pandas | 2.3.3 |
| NumPy | 2.3.4 |
| Scikit-Learn | 1.7.2 |
| Joblib | 1.5.2 |
| FastAPI | 0.120.4 |
| Uvicorn | 0.38.0 |
| MLflow | 3.5.1 |

---

💡 *Ideal for*:  
Portfolio projects, enterprise churn analysis, or as a reference for OOP-based ML pipeline design.

---

## 🎯 Business Objective
The goal of this project is to **reduce customer churn** in a telecommunications company by identifying clients at high risk of cancellation.  
The predictive model enables the business to proactively engage with customers (via retention campaigns, personalized offers, or improved customer service) before churn occurs.

---

## 🧩 Methodology
1. **Data ingestion and preprocessing** handled by modular classes (`src/data_loader.py`, `src/preprocessing.py`).
2. **Categorical encoding** (Yes/No → 1/0) and **numeric scaling** for model stability.
3. **Random Forest classifier** implemented in a fully OOP structure (`src/models/churn_model.py`).
4. **Evaluation metrics:** accuracy, precision, recall, F1-score, and ROC-AUC (`src/evaluation.py`).
5. **Experiment tracking** with MLflow and **structured logging** for model monitoring.
6. **FastAPI** serves real-time predictions via REST API.
7. **Docker containerization** ensures reproducible and scalable deployment.

---

## 📈 Model Performance
| Metric       | Value     | Interpretation                                                |
|---------------|-----------|---------------------------------------------------------------|
| Accuracy      | **0.955** | The model correctly predicts ~95.5% of all cases              |
| Precision     | **0.929** | Among predicted churns, ~92.9% were actual churns             |
| Recall        | **0.738** | The model identifies ~73.8% of customers who actually churned |
| F1-score      | **0.822** | Balanced trade-off between precision and recall               |
| ROC-AUC       | **0.901** | Excellent discrimination between churn vs non-churn           |

---

## ⚙️ Project Architecture
```
ProjectChurn_POO/
├── data/ # Raw input data
├── model/ # Trained model artifacts (.joblib)
├── logs/ # Structured log files
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── evaluation.py
│ ├── train.py
│ ├── api/
│ │ └── app.py
│ └── models/
│ ├── base_model.py
│ └── churn_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```


---


---

## 🔧 Tech Stack
- **Python **  
- **Pandas, NumPy, Scikit-learn** — data handling and model training  
- **MLflow** — experiment tracking  
- **FastAPI** — REST API for model inference  
- **Docker** — containerized deployment  
- **Logging & Metrics** — production-grade observability  

---

## 🏃 How to Run

### ▶️ Local execution
```bash
pip install -r requirements.txt
python -m src.train
uvicorn src.api.app:app --reload
```


### 🐳 Using Docker
```bash
docker build -t projectchurn-api .
docker run -p 8000:8000 projectchurn-api
```
---
## 📋 API Endpoints
| Endpoint   | Method | Description                                                 |
| ---------- | ------ | ----------------------------------------------------------- |
| `/`        | GET    | Health check — confirms the API is running                  |
| `/predict` | POST   | Returns churn probability and class prediction              |
| `/metrics` | GET    | Shows API uptime, total requests, and average response time |

### Example request (/predict)
```
{
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
```
### Example response
```
{
  "churn_probability": 0.42,
  "churn_prediction": 0,
  "interpretation": "No Churn"
}
```
## 🧠 Observability Features

Structured logs persisted under logs/app.log

Automatic error handling with middleware-based exception logging

Internal metrics endpoint (/metrics) for uptime monitoring

Integrated MLflow experiment tracking for reproducibility

## 🧠 Observability Features

Structured logs persisted under logs/app.log

Automatic error handling with middleware-based exception logging

Internal metrics endpoint (/metrics) for uptime monitoring

Integrated MLflow experiment tracking for reproducibility

## 🚀 Next Steps

Add model retraining pipeline triggered by new data (data drift handling).

Integrate a visualization dashboard (Streamlit / Power BI).

Publish Docker image to Docker Hub.

Deploy the API to a cloud environment (Render, AWS, or Railway).

## 📄 License

MIT License — Free to use for educational, research, and portfolio purposes.

## ✨ Author

**[Bruno Cerqueira](https://www.linkedin.com/in/brunobcerqueira/)**  
🧠 MSc in Astronomy & Astrophysics · Data Scientist · Deep Learning  · TensorFlow 


[![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/Onirb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/brunobcerqueira/)
