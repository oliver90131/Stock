# This is a sample Python script.
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from itertools import product
import random
import io
import datetime

st.title("📈 Stock LSTM Forecast & Backtest (Auto-update Latest Data)")

# 1️⃣ User inputs
ticker_input = st.text_input("Enter stock ticker (e.g., 2330.TW):", value="2330.TW")
date_input = st.date_input("Prediction start date (YYYY-MM-DD)", value=datetime.date.today())

if st.button("Start Prediction"):

    try:
        ticker = ticker_input.strip()
        input_date = pd.to_datetime(date_input)

        # -------------------------------
        # 2️⃣ Auto-download latest 2-year data
        # -------------------------------
        data = yf.download(ticker, period="2y", interval="1d")[['Close']].dropna()
        data.index = pd.to_datetime(data.index)
        if data.empty:
            st.error(f"No data found for {ticker}")
            st.stop()

        # 如果輸入日期比最新交易日還晚，改成最新交易日
        latest_date = data.index[-1]
        if input_date > latest_date:
            input_date = latest_date
            st.warning(f"Input date is after last trading day. Using latest trading day: {latest_date.date()}")

        # -------------------------------
        # 3️⃣ Technical indicators
        # -------------------------------
        def RSI(series, period=14):
            delta = series.diff()
            gain = delta.where(delta>0, 0)
            loss = -delta.where(delta<0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def MACD(series, fast=12, slow=26, signal=9):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line

        def Bollinger(series, period=20):
            sma = series.rolling(period).mean()
            std = series.rolling(period).std()
            upper = sma + 2*std
            lower = sma - 2*std
            return upper, lower

        data['RSI'] = RSI(data['Close'])
        data['MACD'], data['MACD_signal'] = MACD(data['Close'])
        data['BB_upper'], data['BB_lower'] = Bollinger(data['Close'])
        data.dropna(inplace=True)

        # -------------------------------
        # 4️⃣ Weighted normalization
        # -------------------------------
        weights = 1 + 0.5 * np.arange(len(data)) / len(data)
        data_weighted = data * weights[:, None]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_weighted)

        # -------------------------------
        # 5️⃣ LSTM dataset
        # -------------------------------
        time_step = 60
        def create_dataset(dataset, time_step=time_step):
            X, Y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i-time_step:i])
                Y.append(dataset[i,0])
            return np.array(X), np.array(Y)

        X, y = create_dataset(scaled_data)

        # -------------------------------
        # 6️⃣ LSTM model auto neuron selection
        # -------------------------------
        lstm1_options = [100, 120, 150]
        lstm2_options = [50, 80, 100]
        best_val_loss = float("inf")
        best_model = None
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        for n1, n2 in product(lstm1_options, lstm2_options):
            model = Sequential()
            model.add(LSTM(n1, return_sequences=True, input_shape=(X.shape[1], X.shape[2]),
                           kernel_regularizer=regularizers.l2(0.001)))
            model.add(Dropout(0.3))
            model.add(LSTM(n2, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))
            model.add(Dropout(0.3))
            model.add(Dense(30, kernel_regularizer=regularizers.l2(0.001)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X, y, batch_size=32, epochs=60, validation_split=0.1,
                                callbacks=[early_stop], verbose=0)
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        st.success(f"✅ Best model trained! Validation loss: {best_val_loss:.6f}")

        # -------------------------------
        # 7️⃣ Predict next 30 days
        # -------------------------------
        start_idx = data.index.get_loc(input_date) - time_step
        last_60_days = scaled_data[start_idx:data.index.get_loc(input_date)+1]
        X_test = np.array([last_60_days])
        future_days = 30
        predicted_prices = []
        current_input = X_test.copy()
        for _ in range(future_days):
            pred = best_model.predict(current_input)[0][0]
            temp = np.tile(pred, (scaled_data.shape[1],))
            pred_price = float(scaler.inverse_transform([temp])[0][0])
            pred_price *= (1 + random.uniform(-0.03, 0.03))
            predicted_prices.append(pred_price)
            current_input = np.append(current_input[:,1:,:], [[temp]], axis=1)
        predicted_prices = np.array(predicted_prices).flatten()
        future_dates = pd.date_range(input_date + pd.Timedelta(days=1), periods=future_days, freq='B')
        df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": predicted_prices})

        # -------------------------------
        # 8️⃣ Backtest historical data
        # -------------------------------
        X_test_seq, y_test_seq = create_dataset(scaled_data)
        y_test_prices = data['Close'].values[time_step:]
        y_pred_scaled = best_model.predict(X_test_seq)
        y_pred_prices = []
        for pred_scaled in y_pred_scaled:
            temp = np.tile(pred_scaled, (scaled_data.shape[1],))
            y_pred_prices.append(float(scaler.inverse_transform([temp])[0][0]))
        y_pred_prices = np.array(y_pred_prices).flatten()

        # -------------------------------
        # 9️⃣ Simulated portfolio
        # -------------------------------
        initial_cash = 100000
        cash = initial_cash
        shares = 0
        portfolio_value = []
        for i in range(len(y_pred_prices)):
            if i < len(y_pred_prices)-1:
                if y_pred_prices[i+1] > y_pred_prices[i] and shares == 0:
                    shares = cash / y_pred_prices[i]
                    cash = 0
                elif y_pred_prices[i+1] <= y_pred_prices[i] and shares > 0:
                    cash = shares * y_pred_prices[i]
                    shares = 0
            total = cash + (shares * y_pred_prices[i] if shares > 0 else 0)
            portfolio_value.append(total)
        portfolio_value = np.array(portfolio_value).flatten()

        # -------------------------------
        # 10️⃣ CSV download buttons
        # -------------------------------
        csv_future = df_future.to_csv(index=False).encode('utf-8')
        csv_backtest = pd.DataFrame({
            "Date": data.index[-len(y_test_prices):],
            "Actual_Close": y_test_prices.flatten(),
            "Predicted_Close": y_pred_prices.flatten(),
            "Portfolio_Value": portfolio_value
        }).to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download 30-Day Forecast CSV",
            data=csv_future,
            file_name=f"{ticker}_predicted_30_days.csv",
            mime="text/csv"
        )

        st.download_button(
            label="📥 Download Backtest CSV",
            data=csv_backtest,
            file_name=f"{ticker}_backtest.csv",
            mime="text/csv"
        )

        # -------------------------------
        # 11️⃣ Visualization
        # -------------------------------
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax2 = ax1.twinx()
        ax1.plot(data.index[-60:], data['Close'].tail(60), label="Historical Close", color='blue', linewidth=2)
        ax1.plot(data.index[-len(y_pred_prices):], y_pred_prices, label="Model Prediction", color='green', linewidth=2)
        ax1.plot(future_dates, predicted_prices, marker='o', label="Next 30 Days Forecast", color='red', linewidth=2)
        ax2.plot(data.index[-len(y_pred_prices):], portfolio_value, color='orange', label="Simulated Portfolio Value", linewidth=2)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Stock Price (TWD)")
        ax2.set_ylabel("Portfolio Value (TWD)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")