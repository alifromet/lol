import streamlit as st
import pandas as pd
from ta import add_all_ta_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

# Set Streamlit page
st.set_page_config(page_title="Stock Direction Predictor", layout="wide")

# Title
st.title("📈 Stock Market Direction Predictor")

# Sidebar inputs
api_key = st.sidebar.text_input("🔑 Alpha Vantage API Key", type="password")
symbol = st.sidebar.text_input("📊 Stock Symbol (e.g. AAPL, TSLA)", value="AAPL")
submit = st.sidebar.button("Run Prediction")

# When button clicked
if submit:
    try:
        st.info(f"Fetching data for **{symbol}**...")

        # Fetch stock data
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')

        # Format and clean data
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High',
            '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
        }, inplace=True)
        data = data[::-1].copy()
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data.dropna(inplace=True)

        # Select features
        features = [
            'momentum_rsi', 'trend_macd', 'trend_macd_signal',
            'trend_sma_fast', 'trend_sma_slow',
            'momentum_stoch', 'volume_adi', 'volatility_bbh'
        ]
        X = data[features]
        y = data['Target']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]

        # Format prediction result
        predictions = []
        for prob in y_probs:
            label = "Up" if prob >= 0.5 else "Down"
            chance = prob if prob >= 0.5 else 1 - prob
            predictions.append(f"{label} with {round(chance * 100)}% chance")

        # Show results
        st.subheader(f"📋 Predictions for {symbol}")
        results_df = X_test.copy()
        results_df['Actual'] = y_test.values
        results_df['Predicted'] = predictions
        results_df['Probability_Up'] = y_probs
        st.dataframe(results_df.tail(100).reset_index(drop=True))

        # Show accuracy
        y_pred = (y_probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc * 100:.2f}%")

        # Plot probability distribution
        st.subheader("📊 Probability Distribution (Up)")
        fig, ax = plt.subplots()
        ax.hist(y_probs, bins=20, color='skyblue')
        ax.set_xlabel("Probability of Up")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Confidence Distribution for {symbol}")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
