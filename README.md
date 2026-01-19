# Credit Default Risk Prediction API

Production-ready machine learning inference API for predicting the probability of loan default.

This project exposes a trained LightGBM credit risk model through a FastAPI service, fully containerized with Docker and deployed as a stateless API suitable for cloud environments.

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

---

## API Endpoints

### Health Check

GET /health

Returns service and model readiness.

Example response:
{
  "status": "ok",
  "model_ready": true
}

---

### Predict Default Probability

POST /predict

Returns the probability of loan default for a single application.

Example payload:
{
  "SK_ID_CURR": 100001,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "M",
  "AMT_INCOME_TOTAL": 120000,
  "AMT_CREDIT": 450000,
  "DAYS_BIRTH": -16000,
  "DAYS_EMPLOYED": -2000,
  "EXT_SOURCE_1": 0.45,
  "EXT_SOURCE_2": 0.51,
  "EXT_SOURCE_3": 0.48,
  "ORGANIZATION_TYPE": "Business Entity Type 3"
}

Example response:
{
  "SK_ID_CURR": 100001,
  "probability_of_default": 0.1834
}

---

## Running Locally with Docker

Build the image:

docker build -t credit-risk-api .

Run the container:

docker run -p 8000:8000 credit-risk-api

The API will be available at:
http://localhost:8000

---

## Running with Docker Compose (Development)

docker compose up --build

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

- This repository contains only inference logic.
- Training, feature exploration, and experimentation live in a separate research repository.
- Model artifacts are versioned to guarantee reproducibility.
- No sensitive data or credentials are included.

---

## License

This project is intended for educational and portfolio purposes.
