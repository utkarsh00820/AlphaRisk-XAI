from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import engine, get_db, Base
import models
from ml_engine import train_and_predict

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AlphaRisk XAI Complete API")

@app.get("/")
def health_check():
    return {"status": "Online and Ready!"}

@app.post("/predict/{ticker}")
def predict_stock(ticker: str, db: Session = Depends(get_db)):
    try:
        # Run full ML pipeline (Download real data, train, and predict)
        results = train_and_predict(ticker)
        
        # Log to database
        new_log = models.PredictionLog(
            ticker=results["ticker"],
            volatility_risk=results["prediction"]["volatility_risk"],
            direction=results["prediction"]["direction"],
            close_price=results["latest_close"]
        )
        db.add(new_log)
        db.commit()
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
