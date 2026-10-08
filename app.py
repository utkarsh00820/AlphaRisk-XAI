import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AlphaRisk XAI", layout="wide")

st.title("📈 AlphaRisk XAI: Complete Real-Time Predictor")
st.markdown("This system downloads live stock data from Yahoo Finance, trains an **XGBoost** Machine Learning model on the fly, and uses **SHAP** Explainable AI to justify its predictions.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "AAPL")

if st.button("Run Complete AI Pipeline"):
    with st.spinner(f"Downloading live data & training AI for {ticker.upper()}..."):
        try:
            # Talk to our FastAPI engine
            response = requests.post(f"http://127.0.0.1:8000/predict/{ticker}")
            
            if response.status_code == 200:
                data = response.json()
                
                st.success(f"Analysis Complete for {data['ticker']}! Current Price: ${data['latest_close']}")
                
                # --- Metrics ---
                col1, col2, col3 = st.columns(3)
                
                direction = data["prediction"]["direction"]
                prob = data["prediction"]["probability_up"]
                risk = data["prediction"]["volatility_risk"]
                
                if direction == "Upward":
                    col1.metric("Predicted Direction", direction, f"{int(prob*100)}% Confidence")
                else:
                    col1.metric("Predicted Direction", direction, f"-{int((1-prob)*100)}% Confidence")
                    
                col2.metric("Volatility Risk", risk)
                col3.metric("Latest Close Price", f"${data['latest_close']}")
                
                # --- Explainable AI (SHAP) Chart ---
                st.subheader("🧠 Explainable AI: Why did the model make this choice?")
                st.write("Green bars push the price prediction UP. Red bars push it DOWN.")
                
                shap_data = data["shap_values"]
                # Convert to dataframe for plotly chart
                shap_df = pd.DataFrame(list(shap_data.items()), columns=['Feature', 'Impact'])
                shap_df = shap_df.sort_values('Impact')
                
                shap_df['Color'] = shap_df['Impact'].apply(lambda x: 'green' if x > 0 else 'red')
                
                fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                             color='Color', color_discrete_map={'green': '#00CC96', 'red': '#EF553B'})
                fig.update_layout(showlegend=False, xaxis_title="SHAP Value (Impact on Prediction)")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Error from Backend: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Please ensure you are running: `python -m uvicorn main:app --reload` in your other terminal.")
