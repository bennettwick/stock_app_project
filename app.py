# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

# 1. Accept a comma-separated list of tickers
ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value="AAPL, MSFT").upper().strip()
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

# 2. Validate that there are between 2 and 5 tickers
if not (2 <= len(tickers) <= 5):
    st.sidebar.warning("Please enter between 2 and 5 valid ticker symbols.")
    st.stop()

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

if start_date is None or end_date is None:
    st.sidebar.warning("Please select both start and end dates.")
    st.stop()

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# 3. Enforce a minimum date range of 1 year (365 days)
if (end_date - start_date).days < 365:
    st.sidebar.error("The date range must be at least 1 year (365 days).")
    st.stop()

# -- Data download ----------------------------------------
# 4. Download tickers + S&P 500, wrapped in try/except
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker_list: list, start: date, end: date) -> pd.DataFrame:
    """Download daily adjusted close data from Yahoo Finance."""
    # Append the S&P 500 benchmark
    download_list = ticker_list + ["^GSPC"]
    
    try:
        # Download data (returns a MultiIndex DataFrame for multiple tickers)
        df_raw = yf.download(download_list, start=start, end=end, progress=False)
        
        if df_raw.empty:
            return pd.DataFrame()
            
        # Extract the 'Adj Close' level to account for dividends and splits
        if "Adj Close" in df_raw.columns.levels[0]:
            df_prices = df_raw["Adj Close"]
        elif "Close" in df_raw.columns.levels[0]:
            df_prices = df_raw["Close"]
        else:
            df_prices = df_raw
            
        return df_prices

    except Exception as e:
        # Raise to be caught in the main logic block
        raise Exception(f"API Error: {e}")

# -- Main logic -------------------------------------------
try:
    df = load_data(tickers, start_date, end_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

if df.empty:
    st.error("No data found for the provided tickers. Check the symbols and try again.")
    st.stop()

# Check if any user-requested tickers completely failed to download
missing_tickers = [t for t in tickers + ["^GSPC"] if t not in df.columns]
if missing_tickers:
    st.error(f"Failed to download data for: {', '.join(missing_tickers)}. Please ensure all tickers are valid.")
    st.stop()

# 5. Handle partial data by dropping NaNs and issuing a warning
initial_rows = len(df)
df = df.dropna()

if len(df) < initial_rows:
    st.warning(
        "Some tickers had missing data for the selected date range. "
        "The dataset has been truncated to the overlapping dates to ensure accurate comparison."
    )

if df.empty:
    st.error("After aligning the dates, no overlapping data remains. Try adjusting your date range or tickers.")
    st.stop()

# -- Temporary testing outputs ----------------------------
st.success("Data successfully loaded and validated! The foundation is ready.")
st.write("Preview of aligned Adjusted Closing Prices:")
st.dataframe(df.tail())

st.subheader("Price Chart Preview")
st.line_chart(df)