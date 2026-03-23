import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import date

st.set_page_config(
    page_title="IHSG Sectoral Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 1px solid #e94560;
    }
    
    /* Header utama */
    .main-header {
        background: linear-gradient(90deg, #e94560, #f5a623, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Subtitle */
    .sub-header {
        color: #a0a0b0;
        text-align: center;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a0a0b0;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560, #f5a623) !important;
        color: white !important;
    }

    /* Divider */
    hr {
        border-color: #e94560;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "IHSG":                  "#00d2ff",
    "Financial (BBCA)":      "#e94560",
    "Consumer (UNVR)":       "#f5a623",
    "Mining (ADRO)":         "#00ff88",
    "Telco (TLKM)":          "#bf5fff",
    "Property (BSDE)":       "#ff6b6b",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ── Header ───────────────────────────────────────
st.markdown('<p class="main-header">📊 IHSG Sectoral Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Nashwa Nadilah · Actuarial Science, Universitas Indonesia · <a href="https://github.com/nashwanadilah/IHSG-Sectoral-Analysis" style="color:#e94560">GitHub</a></p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")

TICKERS = {
    "IHSG":                  "^JKSE",
    "Financial (BBCA)":      "BBCA.JK",
    "Consumer (UNVR)":       "UNVR.JK",
    "Mining (ADRO)":         "ADRO.JK",
    "Telco (TLKM)":          "TLKM.JK",
    "Property (BSDE)":       "BSDE.JK",
}

selected = st.sidebar.multiselect(
    "📈 Pilih Saham:",
    options=list(TICKERS.keys()),
    default=["IHSG", "Financial (BBCA)", "Mining (ADRO)", "Consumer (UNVR)"]
)

start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date   = st.sidebar.date_input("End Date",   date(2025, 12, 31))

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.markdown("""
Proyek analisis performa sektoral pasar modal Indonesia 
menggunakan data historis 2020–2025.

Mencakup analisis:
- Trend & Return
- Volatilitas
- Korelasi
""")

# ── Load Data ─────────────────────────────────────
@st.cache_data
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

with st.spinner("Mengunduh data pasar..."):
    prices = load_data(
        {k: TICKERS[k] for k in selected},
        start_date, end_date
    )

# ── Metric Cards ──────────────────────────────────
total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
daily_ret = prices.pct_change().dropna()
ann_vol   = daily_ret.std() * np.sqrt(252) * 100

cols = st.columns(len(selected))
for i, stock in enumerate(prices.columns):
    ret = total_ret[stock]
    vol = ann_vol[stock]
    with cols[i]:
        st.metric(
            label=f"**{stock}**",
            value=f"{ret:.1f}%",
            delta=f"Vol: {vol:.1f}%",
        )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance",
    "Volatility",
    "Correlation",
    "Summary"
])

# ════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Normalized Price Performance (Base = 100)")

    normalized = prices.div(prices.iloc[0]) * 100

    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            name=col,
            line=dict(color=COLORS.get(col, "#ffffff"), width=2.5),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>"
        ))

    # COVID highlight
    fig.add_vrect(
        x0="2020-02-24", x1="2020-03-24",
        fillcolor="red", opacity=0.1,
        annotation_text="COVID Crash", annotation_position="top left",
        annotation_font_color="red"
    )
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Yearly Return ──
    st.subheader("Yearly Return (%)")

    yearly = {}
    for year in range(start_date.year, end_date.year + 1):
        yr_data = prices[prices.index.year == year]
        if len(yr_data) >= 2:
            yearly[year] = ((yr_data.iloc[-1] / yr_data.iloc[0]) - 1) * 100

    returns_df = pd.DataFrame(yearly).T.round(2)

    fig2 = go.Figure()
    for i, col in enumerate(returns_df.columns):
        fig2.add_trace(go.Bar(
            name=col,
            x=returns_df.index.astype(str),
            y=returns_df[col],
            marker_color=COLORS.get(col, "#ffffff"),
            opacity=0.85,
            hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>"
        ))

    fig2.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.5)
    fig2.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        height=400,
        xaxis_title="Year",
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 — VOLATILITY
# ════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Volatility Ranking")
        ann_vol_sorted = ann_vol.sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=ann_vol_sorted.values,
            y=ann_vol_sorted.index,
            orientation="h",
            marker=dict(
                color=ann_vol_sorted.values,
                colorscale="RdYlGn_r",
                showscale=True
            ),
            text=[f"{v:.1f}%" for v in ann_vol_sorted.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Volatility: %{x:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            xaxis_title="Annualized Volatility (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.2)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk-Return Profile")
        fig = go.Figure()
        for stock in ann_vol.index:
            fig.add_trace(go.Scatter(
                x=[ann_vol[stock]],
                y=[total_ret[stock]],
                mode="markers+text",
                name=stock,
                text=[stock],
                textposition="top center",
                marker=dict(
                    size=18,
                    color=COLORS.get(stock, "#ffffff"),
                    line=dict(width=2, color="white")
                ),
                hovertemplate=f"<b>{stock}</b><br>Risk: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>"
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=ann_vol.mean(), line_dash="dash",
                      line_color="gray", opacity=0.5)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            xaxis_title="Annualized Volatility / Risk (%)",
            yaxis_title="Cumulative Return (%)",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.2)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling Volatility
    st.subheader("Rolling 30-Day Volatility")
    roll_vol = daily_ret.rolling(30).std() * np.sqrt(252) * 100

    fig = go.Figure()
    for col in roll_vol.columns:
        fig.add_trace(go.Scatter(
            x=roll_vol.index,
            y=roll_vol[col],
            name=col,
            line=dict(color=COLORS.get(col, "#ffffff"), width=1.8),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Vol: %{{y:.1f}}%<extra></extra>"
        ))

    fig.add_vrect(
        x0="2020-02-24", x1="2020-03-24",
        fillcolor="red", opacity=0.1,
        annotation_text="COVID Crash",
        annotation_font_color="red"
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 — CORRELATION
# ════════════════════════════════════════════════
with tab3:
    if len(selected) < 2:
        st.warning("Pilih minimal 2 saham untuk melihat korelasi!")
    else:
        st.subheader("Correlation Matrix")
        corr = daily_ret.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            aspect="auto",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation ranking
        st.subheader("Correlation Ranking")
        pairs = []
        cols_list = corr.columns.tolist()
        for i in range(len(cols_list)):
            for j in range(i+1, len(cols_list)):
                pairs.append({
                    "Stock A": cols_list[i],
                    "Stock B": cols_list[j],
                    "Correlation": round(corr.loc[cols_list[i], cols_list[j]], 3),
                    "Interpretation": (
                        "Low — Good for diversification" if abs(corr.loc[cols_list[i], cols_list[j]]) < 0.4
                        else "Medium" if abs(corr.loc[cols_list[i], cols_list[j]]) < 0.7
                        else "High — Similar movement"
                    )
                })
        pairs_df = pd.DataFrame(pairs).sort_values("Correlation")
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════
# TAB 4 — SUMMARY
# ════════════════════════════════════════════════
with tab4:
    st.subheader("Portfolio Summary")

    summary = pd.DataFrame({
        "Cumulative Return (%)": total_ret.round(2),
        "Annualized Volatility (%)": ann_vol.round(2),
        "Risk-Return Ratio": (total_ret / ann_vol).round(2),
    }).sort_values("Cumulative Return (%)", ascending=False)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Stock</b>"] + [f"<b>{c}</b>" for c in summary.columns],
            fill_color="#e94560",
            align="center",
            font=dict(color="white", size=13),
            height=35
        ),
        cells=dict(
            values=[summary.index] + [summary[c] for c in summary.columns],
            fill_color=[["#1a1a2e", "#16213e"] * len(summary)],
            align="center",
            font=dict(color="white", size=12),
            height=30
        )
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative return bar
    st.subheader("Cumulative Return Ranking")
    sorted_ret = total_ret.sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=sorted_ret.values,
        y=sorted_ret.index,
        orientation="h",
        marker=dict(
            color=sorted_ret.values,
            colorscale="RdYlGn",
            showscale=True
        ),
        text=[f"{v:.1f}%" for v in sorted_ret.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Return: %{x:.1f}%<extra></extra>"
    ))
    fig.add_vline(x=0, line_color="white", line_width=0.8, opacity=0.5)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        xaxis_title="Cumulative Return (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ by <b>Nashwa Nadilah</b> · "
    "Actuarial Science, Universitas Indonesia · "
    "<a href='https://github.com/nashwanadilah/IHSG-Sectoral-Analysis' style='color:#e94560'>GitHub</a>"
    "</small></center>",
    unsafe_allow_html=True)
