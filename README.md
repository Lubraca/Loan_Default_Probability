# Credit Default Risk Prediction API

Production-ready machine learning inference API for predicting the probability of loan default.

This project exposes a trained LightGBM credit risk model through a FastAPI service, designed with a strict separation between training and inference, reproducible preprocessing, and stateless deployment. The service is fully containerized with Docker and suitable for cloud-native environments.

The focus of this repository is **model serving and production architecture**, including artifact management, feature consistency, and API reliability.

The research repository containing exploratory analysis, feature engineering experiments, and modeling decisions is available here:
https://github.com/Lubraca/Supervised_ML_Models/tree/main/Classification/Loan_Default_Prediction/BR-Macro-Enhanced_Credit_Default

This repository demonstrates production ML engineering practices rather than model performance optimization.
---

## Live Deployment

The API is publicly available and deployed on **Render**.

**Base URL**  
https://credit-risk-api-ho5h.onrender.com

---

## Overview

The API receives raw loan application data and returns the probability of default (`TARGET = 1`).  
It reproduces the same preprocessing pipeline used during model training, including:

- Feature engineering  
- Target encoding for categorical variables  
- Missing value imputation  
- Feature alignment with the final trained model  

The service is optimized for **inference only** and does not include training logic.

---

## Tech Stack

- Python  
- FastAPI  
- LightGBM  
- scikit-learn  
- Docker  
- Uvicorn  
- Pydantic v2  

---

## Project Structure

```text
.
├── src/
│   ├── main.py        # FastAPI application
│   ├── predict.py     # Inference pipeline (PredictionHandler)
│   └── schemas.py     # Pydantic input/output schemas
│
├── models/            # Model artifacts
│   ├── final_lgbm_model.pkl
│   ├── final_target_encoder.pkl
│   ├── final_imputation_map.json
│   └── FINAL_MODEL_FEATURES.json
│
├── Dockerfile
├── requirements.txt   # Production dependencies
├── docker-compose.yml # Local development
├── .gitignore
└── README.md
```

---

## API Endpoints

### Health Check

```
GET /health
```

**Full URL**
```
https://credit-risk-api-ho5h.onrender.com/health
```

Example response:
```json
{
  "status": "ok",
  "model_ready": true
}
```

---

### Predict Default Probability

```
POST /predict
```

**Full URL**
```
https://credit-risk-api-ho5h.onrender.com/predict
```

---

### Example Request (cURL)

```bash
curl -X POST "https://credit-risk-api-ho5h.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "SK_ID_CURR": 100001,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "M",
  "AMT_INCOME_TOTAL": 120000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 25000,
  "DAYS_BIRTH": -16000,
  "DAYS_EMPLOYED": -2000,
  "DAYS_ID_PUBLISH": -3000,
  "DAYS_REGISTRATION": -5000,
  "DAYS_LAST_PHONE_CHANGE": -800,
  "EXT_SOURCE_1": 0.45,
  "EXT_SOURCE_2": 0.51,
  "EXT_SOURCE_3": 0.48,
  "ORGANIZATION_TYPE": "Business Entity Type 3"
}
JSON
```

---

### Example Response

```json
{
  "SK_ID_CURR": 100001,
  "probability_of_default": 0.08454134770124873
}
```

**Note:** Derived features (e.g. ratios) are computed server-side when sufficient raw inputs are provided.

---

## Running Locally with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

The API will be available at:
```
http://localhost:8000
```

---

## Running with Docker Compose (Development)

```bash
docker compose up --build
```

---

## Deployment

This service is designed to be deployed as a Docker-based Web Service on platforms such as:

- Render  
- Fly.io  
- Railway  
- AWS ECS / Fargate  

The application listens on the port provided by the environment (`$PORT`), making it cloud-compatible by default.

---

## Notes

- This repository contains **only inference logic**.  
- Training, feature exploration, and experimentation live in a separate research repository ().  
- Model artifacts are versioned to guarantee reproducibility.  
- No sensitive data or credentials are included.  

---

## License

This project is intended for educational and portfolio purposes.
