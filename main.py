from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
import pickle
import pandas as pd
import os
import random

# Initialize FastAPI with docs enabled
app = FastAPI(
    title="ChurnML Predictor", 
    version="1.0.0",
    docs_url="/docs", 
    redoc_url="/redoc"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the imblearn pipeline
MODEL_PATH = os.path.join("model", "model_pipeline.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. Running in Mock Mode.")

# --- Pydantic Schemas ---
class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Age: int = Field(..., ge=18, le=100)
    Balance: float = Field(..., ge=0)
    EstimatedSalary: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    IsActiveMember: int = Field(..., ge=0, le=1)
    Geography: str = Field(..., pattern="^(France|Germany|Spain)$")
    Gender: str = Field(..., pattern="^(Male|Female)$")

class PredictionResponse(BaseModel):
    churn: bool
    churn_probability: float
    risk_level: str
    message: str
    insights: List[str]

# --- EDA Logic (Based on your analysis) ---
def generate_insights(customer: CustomerFeatures) -> List[str]:
    insights = []
    # Product Count Findings
    if customer.NumOfProducts >= 3:
        insights.append("Critical Risk: Analysis shows a near 100% exit rate for 4-product users.")
    elif customer.NumOfProducts == 2:
        insights.append("Loyalty Segment: 2-product users are the most stable 'Sweet Spot'.")
    
    # Geographic Findings
    if customer.Geography == "Germany":
        insights.append("Regional Alert: German customers show higher balances and 2x the churn risk.")
    
    # Demographic Findings
    if customer.Age > 45:
        insights.append("Demographic Driver: Customer is in the peak churn age bracket (45+).")
        
    return insights

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.dict()])
    
    if model:
        try:
            prob = float(model.predict_proba(input_df)[0][1])
        except:
            prob = 0.5
    else:
        prob = round(random.uniform(0.1, 0.85), 4)

    churn = prob >= 0.5
    risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"
    
    return {
        "churn": churn,
        "churn_probability": round(prob, 4),
        "risk_level": risk,
        "message": "Target for retention" if churn else "Maintain engagement",
        "insights": generate_insights(customer)
    }