# 🌊 Flood Risk Prediction

Regression project that predicts `FloodProbability` from 20 environmental and
socio-economic factors (50,000 samples).

## Run locally

    pip install -r requirements.txt
    streamlit run app.py
    uvicorn api.main:app --reload

## API

- `GET /health`
- `GET /model-info`
- `POST /predict`

## Structure

- `notebooks/` training notebook
- `src/` transformers, config, predictor
- `api/` FastAPI app
- `models/` trained model, preprocessor, model card
