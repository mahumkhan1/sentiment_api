import logging 
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from contextlib import asynccontextmanager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")



# pydantic schemas
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"

class PredictRequest(BaseModel):
    text: str = Field(..., example="I love this product!")  

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    score: float
    label: int

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., example=["I love this!", "This is terrible."])

# model loading
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading model...")
    try:
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("model/model.pkl not found — run train.py first")
    yield
    logger.info("Shutting down")


# initialize app
app = FastAPI(
    title="Sentiment Analyzer API",
    description="Classify text as positive or negative",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# helper functions
def predict_one(text) -> dict:
    proba = model.predict_proba([text])[0]
    label = 1 if proba[1] > 0.5 else 0
    sentiment = "positive" if label == 1 else "negative"
    score = float(max(proba))

    return {"sentiment": sentiment, "score": score, "label": label}


@app.get("/")
def root():
    return {"message": "Sentiment API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = predict_one(request.text)
    logger.info(f"predict: [{result['sentiment']}] score={result['score']:.2f})")
    return PredictResponse(text=request.text, **result)

@app.post("/predict/batch")
def batch_predict(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = [predict_one(t) for t in request.texts]
    return {
        "results": [{"text": t, **r} for t, r in zip(request.texts, results)],
        "count": len(results)
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}