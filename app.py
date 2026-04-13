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
import numpy as np
import scipy.stats as stats

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
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker_list: list, start: date, end: date) -> pd.DataFrame:
    """Download daily adjusted close data from Yahoo Finance."""
    download_list = ticker_list + ["^GSPC"]
    
    try:
        df_raw = yf.download(download_list, start=start, end=end, progress=False)
        
        if df_raw.empty:
            return pd.DataFrame()
            
        if "Adj Close" in df_raw.columns.levels[0]:
            df_prices = df_raw["Adj Close"]
        elif "Close" in df_raw.columns.levels[0]:
            df_prices = df_raw["Close"]
        else:
            df_prices = df_raw
            
        return df_prices

    except Exception as e:
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

missing_tickers = [t for t in tickers + ["^GSPC"] if t not in df.columns]
if missing_tickers:
    st.error(f"Failed to download data for: {', '.join(missing_tickers)}. Please ensure all tickers are valid.")
    st.stop()

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

# Determine the actual list of user tickers successfully downloaded
valid_user_tickers = [t for t in tickers if t in df.columns]

# -- Calculations -----------------------------------------
# Compute daily simple returns
returns = df.pct_change().dropna()

# -- Tabs Setup -------------------------------------------
tab1, tab2, tab3 = st.tabs(["Price & Returns", "Risk & Distribution", "Correlation & Portfolio"])

# =========================================================
# TAB 1: Price & Returns
# =========================================================
with tab1:
    st.header("Price & Return Analysis")
    
    # 1. Price Chart (Excluding S&P 500)
    st.subheader("Adjusted Closing Prices")
    
    # Add the required multi-select widget
    selected_price_tickers = st.multiselect(
        "Select stocks to display:", 
        options=valid_user_tickers, 
        default=valid_user_tickers
    )

    fig_price = go.Figure()
    # Loop through the WIDGET selection instead of all tickers
    for col in selected_price_tickers:
        fig_price.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.5)))
    
    fig_price.update_layout(
        yaxis_title="Price (USD)", 
        xaxis_title="Date",
        template="plotly_white", 
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # 2. Summary Statistics Table
    st.subheader("Summary Statistics")
    stats_list = []
    
    for col in df.columns:
        col_returns = returns[col]
        ann_mean = col_returns.mean() * 252
        ann_vol = col_returns.std() * np.sqrt(252)
        skewness = stats.skew(col_returns)
        kurt = stats.kurtosis(col_returns)
        min_ret = col_returns.min()
        max_ret = col_returns.max()
        
        stats_list.append({
            "Asset": col,
            "Ann. Mean Return": f"{ann_mean:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Skewness": f"{skewness:.4f}",
            "Kurtosis": f"{kurt:.4f}",
            "Min Daily Return": f"{min_ret:.2%}",
            "Max Daily Return": f"{max_ret:.2%}"
        })
        
    stats_df = pd.DataFrame(stats_list).set_index("Asset")
    st.dataframe(stats_df, use_container_width=True)

    # 3. Cumulative Wealth Index
    st.subheader("Cumulative Wealth Index ($10,000 Investment)")
    
    # Create a copy of returns to calculate the Equal-Weight portfolio
    wealth_returns = returns.copy()
    wealth_returns["Equal-Weight Portfolio"] = wealth_returns[valid_user_tickers].mean(axis=1)
    
    # Calculate cumulative wealth
    wealth_index = 10000 * (1 + wealth_returns).cumprod()
    
    # We prepend a starting row of $10,000 at the day before the first return
    # to make the chart start cleanly at $10,000 for all assets.
    start_dt = wealth_index.index[0] - timedelta(days=1)
    start_row = pd.DataFrame([[10000] * len(wealth_index.columns)], columns=wealth_index.columns, index=[start_dt])
    wealth_index = pd.concat([start_row, wealth_index])

    fig_wealth = go.Figure()
    
    # Plot individual stocks
    for col in valid_user_tickers:
        fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index[col], mode="lines", name=col, line=dict(width=1.5, dash='dot')))
        
    # Plot S&P 500
    fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index["^GSPC"], mode="lines", name="S&P 500", line=dict(width=2, color='gray')))
    
    # Plot Equal-Weight Portfolio
    fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index["Equal-Weight Portfolio"], mode="lines", name="Equal-Weight Portfolio", line=dict(width=3, color='black')))

    fig_wealth.update_layout(
        yaxis_title="Portfolio Value (USD)", 
        xaxis_title="Date",
        template="plotly_white", 
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

# =========================================================
# TAB 2: Risk & Distribution
# =========================================================
with tab2:
    st.header("Risk & Distribution Analysis")
    
    # 1. Rolling Volatility
    st.subheader("Rolling Annualized Volatility")
    
    vol_window = st.selectbox(
        "Select Rolling Window (Trading Days):", 
        options=[21, 63, 252], 
        format_func=lambda x: "21 Days (1 Month)" if x == 21 else ("63 Days (3 Months)" if x == 63 else "252 Days (1 Year)"),
        index=0
    )
    
    # Check if we have enough trading days to calculate the selected window
    if len(returns) <= vol_window:
        st.warning(
            f"⚠️ Not enough data to calculate a {vol_window}-day rolling volatility. "
            f"Your selected date range only contains {len(returns)} trading days. "
            "Please select a longer date range in the sidebar."
        )
    else:
        # Calculate rolling standard deviation and annualize
        rolling_vol = returns[valid_user_tickers].rolling(window=vol_window).std() * np.sqrt(252)
        
        fig_rolling_vol = go.Figure()
        for col in valid_user_tickers:
            fig_rolling_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col, line=dict(width=1.5)))
            
        fig_rolling_vol.update_layout(
            yaxis_title="Annualized Volatility", 
            xaxis_title="Date",
            template="plotly_white", 
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_rolling_vol, use_container_width=True)

    st.markdown("---")

    # 2. Stock Selector for Distribution Analysis
    st.subheader("Distribution Analysis")
    selected_dist_stock = st.selectbox("Select a stock for distribution analysis:", options=valid_user_tickers)
    dist_returns = returns[selected_dist_stock]

    # 4. Normality Test (Jarque-Bera)
    jb_stat, jb_pval = stats.jarque_bera(dist_returns)
    st.write(f"**Jarque-Bera Test for {selected_dist_stock}:**")
    # Changed .4f to .4e to display the p-value in scientific/exponential notation
    st.write(f"Test Statistic: {jb_stat:.4f} | p-value: {jb_pval:.4e}")
    
    if jb_pval < 0.05:
        st.warning(f"Result: **Rejects normality** (p < 0.05). The returns are not normally distributed.")
    else:
        st.success(f"Result: **Fails to reject normality** (p >= 0.05). The returns are approximately normal.")

    # 3. Distribution vs. Q-Q Plot Toggle
    plot_type = st.radio("Select Plot Type:", options=["Histogram with Normal Curve", "Q-Q Plot"], horizontal=True)
    
    if plot_type == "Histogram with Normal Curve":
        # Fit normal distribution
        mu, std = stats.norm.fit(dist_returns)
        
        fig_dist = go.Figure()
        # Histogram of returns
        fig_dist.add_trace(go.Histogram(
            x=dist_returns, 
            histnorm='probability density', 
            name='Daily Returns', 
            opacity=0.7,
            marker_color='royalblue'
        ))
        
        # Normal Curve overlay
        xmin, xmax = dist_returns.min(), dist_returns.max()
        x_range = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x_range, mu, std)
        fig_dist.add_trace(go.Scatter(
            x=x_range, 
            y=p, 
            mode='lines', 
            name='Fitted Normal Curve', 
            line=dict(color='darkorange', width=3)
        ))
        
        fig_dist.update_layout(
            title=f"Return Distribution for {selected_dist_stock}",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        # Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(dist_returns, dist="norm", fit=True)
        
        fig_qq = go.Figure()
        # Scatter of quantiles
        fig_qq.add_trace(go.Scatter(
            x=osm, 
            y=osr, 
            mode='markers', 
            name='Data Quantiles',
            marker=dict(color='royalblue', size=6)
        ))
        
        # Line of best fit representing theoretical normal distribution
        x_line = np.array([np.min(osm), np.max(osm)])
        y_line = slope * x_line + intercept
        fig_qq.add_trace(go.Scatter(
            x=x_line, 
            y=y_line, 
            mode='lines', 
            name='Theoretical Normal', 
            line=dict(color='darkorange', width=2)
        ))
        
        fig_qq.update_layout(
            title=f"Q-Q Plot for {selected_dist_stock}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("---")

    # 5. Box Plot Comparison
    st.subheader("Return Distributions Comparison")
    fig_box = go.Figure()
    
    for col in valid_user_tickers:
        fig_box.add_trace(go.Box(
            y=returns[col], 
            name=col,
            boxpoints='outliers'
        ))
        
    fig_box.update_layout(
        yaxis_title="Daily Return",
        template="plotly_white",
        height=450,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 3: Correlation & Portfolio (Placeholder)
# =========================================================
with tab3:
    st.info("Correlation & Portfolio features will go here.")