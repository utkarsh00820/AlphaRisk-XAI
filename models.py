from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    volatility_risk = Column(String)
    direction = Column(String)
    close_price = Column(Float)
