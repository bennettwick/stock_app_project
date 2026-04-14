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
# Deduplicate while preserving order to protect matrix math in Tab 3
tickers = list(dict.fromkeys([t.strip() for t in ticker_input.split(",") if t.strip()]))

# 2. Validate that there are between 2 and 5 tickers
if not (2 <= len(tickers) <= 5):
    st.sidebar.warning("Please enter between 2 and 5 valid, unique ticker symbols.")
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

# -- About & Methodology Section --------------------------
with st.sidebar.expander("ℹ️ About & Methodology"):
    st.write("""
    **Project Overview:**
    This application facilitates multi-stock comparison and risk analysis using modern portfolio theory principles.
    
    **Key Assumptions & Methodology:**
    * **Data Source:** Real-time financial data is retrieved from **Yahoo Finance** using the `yfinance` library.
    * **Daily Simple Returns:** Calculated as $P_t / P_{t-1} - 1$ using Adjusted Close prices to account for splits and dividends.
    * **Annualization:** All metrics are annualized assuming **252 trading days**. Volatility is scaled by $\sqrt{252}$.
    * **Wealth Index:** Simulates the cumulative growth of a **$10,000** initial investment.
    * **Portfolio Modeling:** Uses the standard variance-covariance method for two-asset portfolios.
    * **Distribution Analysis:** Uses the **Jarque-Bera test** and **Maximum Likelihood Estimation (MLE)** to compare returns against a theoretical normal distribution.
    * **Data Handling:** Missing data is handled via listwise deletion to ensure all assets are analyzed over an identical, overlapping time period.
    """)

# -- Data download ----------------------------------------
@st.cache_data(show_spinner="Fetching data from Yahoo Finance...", ttl=3600)
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
with st.spinner("Calculating financial metrics..."):
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
    for col in selected_price_tickers:
        fig_price.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.5)))
    
    fig_price.update_layout(
        title="Daily Adjusted Closing Prices",
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
    
    with st.spinner("Rendering wealth index..."):
        wealth_returns = returns.copy()
        wealth_returns["Equal-Weight Portfolio"] = wealth_returns[valid_user_tickers].mean(axis=1)
        
        wealth_index = 10000 * (1 + wealth_returns).cumprod()
        
        start_dt = wealth_index.index[0] - timedelta(days=1)
        start_row = pd.DataFrame([[10000] * len(wealth_index.columns)], columns=wealth_index.columns, index=[start_dt])
        wealth_index = pd.concat([start_row, wealth_index])

        fig_wealth = go.Figure()
        
        for col in valid_user_tickers:
            fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index[col], mode="lines", name=col, line=dict(width=1.5, dash='dot')))
            
        fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index["^GSPC"], mode="lines", name="S&P 500", line=dict(width=2, color='gray')))
        fig_wealth.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index["Equal-Weight Portfolio"], mode="lines", name="Equal-Weight Portfolio", line=dict(width=3, color='black')))

        fig_wealth.update_layout(
            title="Cumulative Wealth ($10,000 Initial Investment)",
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
        options=[21, 63, 126, 252], 
        format_func=lambda x: {
            21: "21 Days (1 Month)", 
            63: "63 Days (3 Months)", 
            126: "126 Days (6 Months)", 
            252: "252 Days (1 Year)"
        }.get(x),
        index=0
    )
    
    if len(returns) <= vol_window:
        st.warning(
            f"⚠️ Not enough data to calculate a {vol_window}-day rolling volatility. "
            f"Your selected date range only contains {len(returns)} trading days. "
            "Please select a longer date range in the sidebar."
        )
    else:
        with st.spinner("Calculating rolling volatility..."):
            rolling_vol = returns[valid_user_tickers].rolling(window=vol_window).std() * np.sqrt(252)
            
            fig_rolling_vol = go.Figure()
            for col in valid_user_tickers:
                fig_rolling_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col, line=dict(width=1.5)))
                
            fig_rolling_vol.update_layout(
                title=f"{vol_window}-Day Rolling Annualized Volatility",
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
    st.write(f"Test Statistic: {jb_stat:.4f} | p-value: {jb_pval:.4e}")
    
    if jb_pval < 0.05:
        st.warning(f"Result: **Rejects normality** (p < 0.05). The returns are not normally distributed.")
    else:
        st.success(f"Result: **Fails to reject normality** (p >= 0.05). The returns are approximately normal.")

    # 3. Distribution vs. Q-Q Plot Toggle
    plot_type = st.radio("Select Plot Type:", options=["Histogram with Normal Curve", "Q-Q Plot"], horizontal=True)
    
    if plot_type == "Histogram with Normal Curve":
        mu, std = stats.norm.fit(dist_returns)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=dist_returns, 
            histnorm='probability density', 
            name='Daily Returns', 
            opacity=0.7,
            marker_color='royalblue'
        ))
        
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
        (osm, osr), (slope, intercept, r) = stats.probplot(dist_returns, dist="norm", fit=True)
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, 
            y=osr, 
            mode='markers', 
            name='Data Quantiles',
            marker=dict(color='royalblue', size=6)
        ))
        
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
        title="Distribution of Daily Returns by Stock",
        yaxis_title="Daily Return",
        xaxis_title="Stock",
        template="plotly_white",
        height=450,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 3: Correlation & Portfolio
# =========================================================
with tab3:
    st.header("Correlation & Diversification Analysis")
    
    # 1. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    user_returns = returns[valid_user_tickers]
    corr_matrix = user_returns.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    
    fig_corr.update_layout(
        title="Pairwise Correlation of Daily Returns",
        xaxis_title="Stock",
        yaxis_title="Stock",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Scatter Plot & Rolling Correlation
    st.subheader("Pairwise Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        stock_a = st.selectbox("Select Stock A:", options=valid_user_tickers, index=0)
    with col2:
        default_b_idx = 1 if len(valid_user_tickers) > 1 else 0
        stock_b = st.selectbox("Select Stock B:", options=valid_user_tickers, index=default_b_idx)
        
    col3, col4 = st.columns(2)
    
    with col3:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=returns[stock_a],
            y=returns[stock_b],
            mode='markers',
            marker=dict(opacity=0.6, color='teal')
        ))
        
        fig_scatter.update_layout(
            title=f"Daily Returns: {stock_a} vs. {stock_b}",
            xaxis_title=f"{stock_a} Daily Return",
            yaxis_title=f"{stock_b} Daily Return",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col4:
        roll_corr_window = st.selectbox(
            "Rolling Correlation Window (Trading Days):", 
            options=[21, 63, 126, 252], 
            format_func=lambda x: {
                21: "21 Days (1 Month)", 
                63: "63 Days (3 Months)", 
                126: "126 Days (6 Months)", 
                252: "252 Days (1 Year)"
            }.get(x),
            index=1
        )
        
        if len(returns) > roll_corr_window:
            rolling_corr = returns[stock_a].rolling(window=roll_corr_window).corr(returns[stock_b])
            
            fig_roll_corr = go.Figure()
            fig_roll_corr.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                line=dict(color='purple', width=2)
            ))
            
            fig_roll_corr.update_layout(
                title=f"{roll_corr_window}-Day Rolling Correlation",
                xaxis_title="Date",
                yaxis_title="Correlation",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_roll_corr, use_container_width=True)
        else:
            st.warning("Not enough data to calculate rolling correlation for the selected window.")
            
    st.markdown("---")
    
    # 3. Two-Asset Portfolio Explorer
    st.subheader("Two-Asset Portfolio Explorer")
    
    with st.spinner("Calculating portfolio diversification models..."):
        ann_ret_a = returns[stock_a].mean() * 252
        ann_ret_b = returns[stock_b].mean() * 252
        
        cov_matrix = returns[[stock_a, stock_b]].cov() * 252
        var_a = cov_matrix.loc[stock_a, stock_a]
        var_b = cov_matrix.loc[stock_b, stock_b]
        cov_ab = cov_matrix.loc[stock_a, stock_b]
        
        weight_a_pct = st.slider(f"Weight of {stock_a} (%)", min_value=0, max_value=100, value=50, step=1)
        w_a = weight_a_pct / 100.0
        w_b = 1.0 - w_a
        
        port_ret = (w_a * ann_ret_a) + (w_b * ann_ret_b)
        port_var = (w_a**2 * var_a) + (w_b**2 * var_b) + (2 * w_a * w_b * cov_ab)
        port_vol = np.sqrt(port_var)
        
        col_metric1, col_metric2 = st.columns(2)
        col_metric1.metric("Portfolio Annualized Return", f"{port_ret:.2%}")
        col_metric2.metric("Portfolio Annualized Volatility", f"{port_vol:.2%}")
        
        weights_a_array = np.linspace(0, 1, 101)
        weights_b_array = 1.0 - weights_a_array
        
        port_vols = np.sqrt(
            (weights_a_array**2 * var_a) + 
            (weights_b_array**2 * var_b) + 
            (2 * weights_a_array * weights_b_array * cov_ab)
        )
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=weights_a_array * 100,
            y=port_vols,
            mode='lines',
            name="Portfolio Volatility Curve",
            line=dict(color='darkblue', width=2)
        ))
        
        fig_curve.add_trace(go.Scatter(
            x=[weight_a_pct],
            y=[port_vol],
            mode='markers',
            name="Current Portfolio",
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig_curve.update_layout(
            title="Diversification Effect: Portfolio Volatility vs. Asset Weight",
            xaxis_title=f"Weight of {stock_a} (%)",
            yaxis_title="Portfolio Annualized Volatility",
            template="plotly_white",
            height=450,
            hovermode="x"
        )
    st.plotly_chart(fig_curve, use_container_width=True)
    
    st.info(
        """
        💡 **Diversification Concept:** This curve demonstrates that by combining two stocks, you can produce a portfolio 
        with lower volatility than either stock individually. This effect occurs because the assets are not perfectly correlated. 
        As long as the correlation is less than 1.0, the curve "dips" or bows downward, 
        showing that diversification reduces overall risk.
        """
    )