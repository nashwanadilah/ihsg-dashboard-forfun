import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date

st.set_page_config(
    page_title="IHSG Sectoral Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Header ───────────────────────────────────────
st.title("📊 IHSG Sectoral Analysis Dashboard")
st.markdown("**Nashwa Nadilah** | Actuarial Science, Universitas Indonesia")
st.markdown("---")

# ── Sidebar (Panel Kontrol) ───────────────────────
st.sidebar.header("⚙️ Settings")

TICKERS = {
    "IHSG":      "^JKSE",
    "Financial (BBCA)":  "BBCA.JK",
    "Consumer (UNVR)":   "UNVR.JK",
    "Mining (ADRO)":     "ADRO.JK",
    "Telco (TLKM)":      "TLKM.JK",
    "Property (BSDE)":   "BSDE.JK",
}

# Widget pilih saham
selected = st.sidebar.multiselect(
    "Pilih Saham:",
    options=list(TICKERS.keys()),
    default=["IHSG", "Financial (BBCA)", "Mining (ADRO)"]
)

# Widget pilih periode
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date   = st.sidebar.date_input("End Date", date(2025, 12, 31))

# ── Load Data ─────────────────────────────────────
@st.cache_data  # supaya data tidak re-download setiap kali ada perubahan
def load_data(tickers_dict, start, end):
    raw = {}
    for name, ticker in tickers_dict.items():
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                raw[name] = df["Close"][ticker]
            else:
                raw[name] = df["Close"]
    return pd.DataFrame(raw).ffill()

if len(selected) == 0:
    st.warning("Pilih minimal 1 saham di panel kiri!")
    st.stop()

with st.spinner("Mengunduh data..."):
    prices = load_data(
        {k: TICKERS[k] for k in selected},
        start_date, end_date
    )

st.success(f"Data loaded: {prices.index[0].date()} → {prices.index[-1].date()}")
st.markdown("---")

# ── Tab Layout ────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance",
    "📉 Volatility",
    "🔗 Correlation",
    "📊 Summary"
])

COLORS = ["#1B4F72","#2E86C1","#28B463","#E67E22","#8E44AD","#C0392B"]

# ════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Normalized Price Performance (Base = 100)")
    
    normalized = prices.div(prices.iloc[0]) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(normalized.columns):
        ax.plot(normalized.index, normalized[col],
                label=col, color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylabel("Normalized Price (Base 100)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Yearly return table
    st.subheader("Yearly Return (%)")
    yearly = {}
    for year in range(start_date.year, end_date.year + 1):
        yr_data = prices[prices.index.year == year]
        if len(yr_data) >= 2:
            yearly[year] = ((yr_data.iloc[-1] / yr_data.iloc[0]) - 1) * 100
    
    returns_df = pd.DataFrame(yearly).T.round(2)
    st.dataframe(returns_df.style.format("{:.2f}%")
                 .background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 — VOLATILITY
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Annualized Volatility (%)")
    
    daily_returns = prices.pct_change().dropna()
    ann_vol = (daily_returns.std() * np.sqrt(252) * 100).sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(ann_vol.index, ann_vol.values,
                      color=COLORS[:len(ann_vol)], alpha=0.85)
        for bar, val in zip(bars, ann_vol.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Volatility (%)")
        ax.set_title("Annualized Volatility Ranking")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Rolling volatility
        roll_vol = daily_returns.rolling(30).std() * np.sqrt(252) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, col in enumerate(roll_vol.columns):
            ax.plot(roll_vol.index, roll_vol[col],
                    label=col, color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.set_ylabel("Volatility (%)")
        ax.set_title("Rolling 30-Day Volatility")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Risk-Return Scatter
    st.subheader("Risk-Return Profile")
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, stock in enumerate(ann_vol.index):
        ax.scatter(ann_vol[stock], total_ret[stock],
                   color=COLORS[i % len(COLORS)], s=150, zorder=5)
        ax.annotate(stock, xy=(ann_vol[stock], total_ret[stock]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Annualized Volatility / Risk (%)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Risk vs Return")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════
# TAB 3 — CORRELATION
# ════════════════════════════════════════════════
with tab3:
    st.subheader("Correlation Matrix")
    
    if len(selected) < 2:
        st.warning("Pilih minimal 2 saham untuk melihat korelasi!")
    else:
        corr = daily_returns.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, annot_kws={"size": 11}, ax=ax)
        ax.set_title("Correlation Matrix — Daily Returns")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabel korelasi ranking
        st.subheader("Correlation Ranking (Lowest = Best for Diversification)")
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append({
                    "Stock A": cols[i],
                    "Stock B": cols[j],
                    "Correlation": round(corr.loc[cols[i], cols[j]], 3)
                })
        pairs_df = pd.DataFrame(pairs).sort_values("Correlation")
        st.dataframe(pairs_df, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — SUMMARY
# ════════════════════════════════════════════════
with tab4:
    st.subheader("Portfolio Summary")
    
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    daily_ret = prices.pct_change().dropna()
    ann_vol   = daily_ret.std() * np.sqrt(252) * 100
    
    summary = pd.DataFrame({
        "Cumulative Return (%)": total_ret.round(2),
        "Annualized Volatility (%)": ann_vol.round(2),
        "Best Year": [
            f"{daily_ret[col].resample('Y').apply(lambda x: (1+x).prod()-1).idxmax().year}"
            for col in prices.columns
        ],
    })
    
    st.dataframe(
        summary.style.format({
            "Cumulative Return (%)": "{:.2f}%",
            "Annualized Volatility (%)": "{:.2f}%",
        }).background_gradient(subset=["Cumulative Return (%)"], cmap="RdYlGn"),
        use_container_width=True
    )
    
    # Metric cards
    st.markdown("---")
    cols = st.columns(len(selected))
    for i, stock in enumerate(prices.columns):
        ret = total_ret[stock]
        vol = ann_vol[stock]
        with cols[i]:
            st.metric(
                label=stock,
                value=f"{ret:.1f}%",
                delta=f"Vol: {vol:.1f}%"
            )

# ── Footer ────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Nashwa Nadilah** | Actuarial Science, Universitas Indonesia | "
    "[GitHub](https://github.com/nashwanadilah/IHSG-Sectoral-Analysis)"
)