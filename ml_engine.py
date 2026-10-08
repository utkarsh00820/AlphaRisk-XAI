import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

def fetch_and_engineer_features(ticker: str):
    # Fetch 2 years of data to have enough for training
    df = yf.download(ticker, period="2y", progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    
    # RSI (14 days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Volatility (20-day rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Target: Next day's return direction (1 if Up, 0 if Down)
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    return df

def train_and_predict(ticker: str):
    df = fetch_and_engineer_features(ticker)
    
    features = ['Close', 'Volume', 'RSI_14', 'MACD', 'Volatility']
    X = df[features]
    y = df['Target']
    
    # Train XGBoost on all data except the very last day
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    # The last day is what we want to predict for tomorrow
    X_predict = X.iloc[[-1]]
    latest_close = float(X_predict['Close'].iloc[0])
    current_volatility = float(X_predict['Volatility'].iloc[0])
    
    model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Predict
    prob_up = float(model.predict_proba(X_predict)[0][1])
    direction = "Upward" if prob_up > 0.5 else "Downward"
    
    # Risk Assessment
    volatility_risk = "High" if current_volatility > 0.3 else ("Medium" if current_volatility > 0.15 else "Low")
    
    # Explainability (SHAP)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_predict)
    
    # Format SHAP values for JSON response
    if isinstance(shap_values, list):
        sv = shap_values[1][0] # Get values for class 1
    else:
        sv = shap_values[0]
        
    shap_dict = {feature: float(val) for feature, val in zip(features, sv)}
    
    return {
        "ticker": ticker.upper(),
        "latest_close": round(latest_close, 2),
        "prediction": {
            "volatility_risk": volatility_risk,
            "direction": direction,
            "probability_up": round(prob_up, 2)
        },
        "shap_values": shap_dict
    }
