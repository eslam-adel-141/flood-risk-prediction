from fastapi import FastAPI
from api.routes import health, predict

app = FastAPI(title="Flood Risk Prediction API", version="1.0.0")

app.include_router(health.router)
app.include_router(predict.router)
