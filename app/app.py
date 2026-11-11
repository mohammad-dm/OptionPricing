import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from optlib.data import fetch_stock_data, rolling_mu_sigma

from optlib.model import Stock, black_scholes, compute_greeks, binomial_american, monte_carlo

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }
    .metric-val   { font-size: 1.4rem; font-weight: 600; }
    .call-color   { color: #3fb950; }
    .put-color    { color: #f85149; }
    .neutral-color{ color: #79c0ff; }
    div[data-testid="stMetric"] { background:#161b22; border:1px solid #21262d;
                                   border-radius:10px; padding:0.75rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    POPULAR = {
        "SPY – S&P 500 ETF": "SPY",
        "AAPL – Apple":       "AAPL",
        "MSFT – Microsoft":   "MSFT",
        "TSLA – Tesla":       "TSLA",
        "NVDA – NVIDIA":      "NVDA",
        "AMZN – Amazon":      "AMZN",
        "GOOGL – Alphabet":   "GOOGL",
        "META – Meta":        "META",
        "BRK-B – Berkshire":  "BRK-B",
        "GLD – Gold ETF":     "GLD",
        "Custom…":            "__custom__",
    }
    choice = st.selectbox("Stock", list(POPULAR.keys()))
    if POPULAR[choice] == "__custom__":
        ticker = st.text_input("Enter ticker", "QQQ").upper().strip()
    else:
        ticker = POPULAR[choice]

    st.markdown("---")
    st.subheader("Option parameters")
    K_pct   = st.slider("Strike (% of spot)", 80, 120, 100, step=1,
                         help="Strike price as % of current spot")
    T_days  = st.slider("Time to expiry (days)", 7, 730, 90, step=1)
    r_pct   = st.slider("Risk-free rate (%)", 0.0, 10.0, 5.0, step=0.1)
    use_hist = st.checkbox("Use estimated σ from history", value=True)
    sigma_override = None
    if not use_hist:
        sigma_override = st.slider("Implied volatility (%)", 5, 100, 20, step=1) / 100

    st.markdown("---")
    st.subheader("Monte Carlo")
    n_sims  = st.slider("Simulations", 100, 2000, 500, step=100)
    n_steps = st.slider("Steps (days)", 30, 365, 252, step=1)
    n_binom = st.slider("Binomial tree steps", 50, 500, 200, step=50)

    st.markdown("---")
    st.subheader("Rolling μ/σ history")
    roll_years = st.slider("Lookback (years)", 2, 15, 10, step=1)

    run_btn = st.button("▶  Run Analysis", width="stretch", type="primary")


# ── Main area ──────────────────────────────────────────────────────────────────
st.title(f"📊 Options Analytics — {ticker}")

if not run_btn:
    st.info("Configure settings in the sidebar, then click **▶ Run Analysis**.")
    st.stop()

# ── Fetch data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**…"):
    try:
        S, mu_hat, sigma_hat, price_series = fetch_stock_data(ticker)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

sigma = sigma_override if sigma_override is not None else sigma_hat
K = S * K_pct / 100
T = T_days / 365
r = r_pct / 100

stock = Stock(S)

# ── Compute prices & greeks ────────────────────────────────────────────────────
eu_call = black_scholes(S, K, T, r, sigma, "call")
eu_put  = black_scholes(S, K, T, r, sigma, "put")
am_call = binomial_american(S, K, T, r, sigma, n_binom, "call")
am_put  = binomial_american(S, K, T, r, sigma, n_binom, "put")
greeks  = compute_greeks(S, K, T, r, sigma)

# ── Top metrics ────────────────────────────────────────────────────────────────
st.subheader("Market snapshot")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot price",   f"${S:.2f}")
c2.metric("Strike (K)",   f"${K:.2f}")
c3.metric("Expiry",       f"{T_days}d")
c4.metric("σ (annual)",   f"{sigma*100:.1f}%")
c5.metric("μ̂ (annual)",  f"{mu_hat*100:.1f}%")

st.markdown("---")

# ── Option prices ──────────────────────────────────────────────────────────────
st.subheader("Option prices")
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("🇪🇺 European Call", f"${eu_call:.4f}")
pc2.metric("🇪🇺 European Put",  f"${eu_put:.4f}",  delta=f"Put-Call parity Δ: {eu_call-eu_put-(S-K*np.exp(-r*T)):.4f}")
pc3.metric("🇺🇸 American Call", f"${am_call:.4f}", delta=f"Early exercise premium: ${am_call-eu_call:.4f}")
pc4.metric("🇺🇸 American Put",  f"${am_put:.4f}",  delta=f"Early exercise premium: ${am_put-eu_put:.4f}")

st.markdown("---")

# ── Greeks dashboard ───────────────────────────────────────────────────────────
st.subheader("Greeks")
gcols = st.columns(5)
greek_meta = {
    "delta": ("Δ Delta",  "Sensitivity to spot price",       ""),
    "gamma": ("Γ Gamma",  "Rate of change of delta",         ""),
    "theta": ("Θ Theta",  "Time decay (per calendar day)",   "/day"),
    "vega":  ("ν Vega",   "Sensitivity to 1% vol change",    "/1%σ"),
    "rho":   ("ρ Rho",    "Sensitivity to 1% rate change",   "/1%r"),
}
for i, (key, (label, tip, unit)) in enumerate(greek_meta.items()):
    with gcols[i]:
        g = greeks[key]
        st.metric(f"{label} (Call)", f"{g['call']:.4f}{unit}", help=tip)
        st.metric(f"{label} (Put)",  f"{g['put']:.4f}{unit}")

st.markdown("---")

# ── Greeks vs Spot price ───────────────────────────────────────────────────────
st.subheader("Greeks vs spot price")
spot_range = np.linspace(S * 0.6, S * 1.4, 200)
greek_series = {g: {"call": [], "put": []} for g in greek_meta}

for s in spot_range:
    g = compute_greeks(s, K, T, r, sigma)
    for key in greek_meta:
        greek_series[key]["call"].append(g[key]["call"])
        greek_series[key]["put"].append(g[key]["put"])

colors_call = ["#3fb950", "#79c0ff", "#f0883e", "#d2a8ff", "#58a6ff"]
colors_put  = ["#f85149", "#58a6ff", "#ffa657", "#bc8cff", "#388bfd"]

fig_greeks = make_subplots(
    rows=3, cols=2,
    subplot_titles=[m[0] for m in greek_meta.values()] + [""],
    vertical_spacing=0.10,
    horizontal_spacing=0.08,
)

positions = [(1,1),(1,2),(2,1),(2,2),(3,1)]
for idx, (key, (label, _, _)) in enumerate(greek_meta.items()):
    row, col = positions[idx]
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=greek_series[key]["call"],
        name=f"{label} Call", line=dict(color=colors_call[idx], width=2),
        legendgroup=key, showlegend=True), row=row, col=col)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=greek_series[key]["put"],
        name=f"{label} Put", line=dict(color=colors_put[idx], width=2, dash="dash"),
        legendgroup=key, showlegend=True), row=row, col=col)
    fig_greeks.add_vline(x=S, line_width=1, line_dash="dot",
        line_color="rgba(255,255,255,0.4)", row=row, col=col)
    fig_greeks.add_vline(x=K, line_width=1, line_dash="dot",
        line_color="rgba(255,200,0,0.5)", row=row, col=col)

fig_greeks.update_layout(
    height=760, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    legend=dict(orientation="h", y=-0.08),
    margin=dict(l=40, r=20, t=60, b=60),
)
fig_greeks.update_xaxes(title_text="Spot price ($)", showgrid=True,
    gridcolor="rgba(255,255,255,0.07)")
fig_greeks.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
st.plotly_chart(fig_greeks, width="stretch")

st.markdown("---")

# ── Option price vs Spot ───────────────────────────────────────────────────────
st.subheader("Option price vs spot price (European & American)")

eu_calls, eu_puts, am_calls, am_puts = [], [], [], []
for s in spot_range:
    eu_calls.append(black_scholes(s, K, T, r, sigma, "call"))
    eu_puts.append(black_scholes(s, K, T, r, sigma, "put"))
    am_calls.append(binomial_american(s, K, T, r, sigma, 100, "call"))
    am_puts.append(binomial_american(s, K, T, r, sigma, 100, "put"))

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=spot_range, y=eu_calls, name="EU Call",
    line=dict(color="#3fb950", width=2.5)))
fig_price.add_trace(go.Scatter(x=spot_range, y=eu_puts, name="EU Put",
    line=dict(color="#f85149", width=2.5)))
fig_price.add_trace(go.Scatter(x=spot_range, y=am_calls, name="AM Call",
    line=dict(color="#3fb950", width=2, dash="dash")))
fig_price.add_trace(go.Scatter(x=spot_range, y=am_puts, name="AM Put",
    line=dict(color="#f85149", width=2, dash="dash")))
fig_price.add_vline(x=S, line_color="rgba(255,255,255,0.5)",
    line_dash="dot", annotation_text="Spot", annotation_position="top right")
fig_price.add_vline(x=K, line_color="rgba(255,200,0,0.5)",
    line_dash="dot", annotation_text="Strike", annotation_position="top left")
fig_price.update_layout(
    height=420, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Spot price ($)", yaxis_title="Option price ($)",
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_price, width="stretch")

st.markdown("---")

# ── Option price vs Volatility ─────────────────────────────────────────────────
st.subheader("Option price vs implied volatility")
vol_range = np.linspace(0.05, 1.0, 200)
vc = [black_scholes(S, K, T, r, v, "call") for v in vol_range]
vp = [black_scholes(S, K, T, r, v, "put")  for v in vol_range]

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=vol_range*100, y=vc, name="EU Call",
    line=dict(color="#3fb950", width=2.5)))
fig_vol.add_trace(go.Scatter(x=vol_range*100, y=vp, name="EU Put",
    line=dict(color="#f85149", width=2.5)))
fig_vol.add_vline(x=sigma*100, line_color="rgba(255,255,255,0.5)",
    line_dash="dot", annotation_text=f"σ={sigma*100:.1f}%", annotation_position="top right")
fig_vol.update_layout(
    height=380, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Volatility (%)", yaxis_title="Option price ($)",
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_vol, width="stretch")

st.markdown("---")

# ── Option price vs Time to expiry ────────────────────────────────────────────
st.subheader("Option price vs time to expiry")
T_range = np.linspace(1/365, 2, 200)
tc = [black_scholes(S, K, t, r, sigma, "call") for t in T_range]
tp = [black_scholes(S, K, t, r, sigma, "put")  for t in T_range]

fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=T_range*365, y=tc, name="EU Call",
    line=dict(color="#3fb950", width=2.5)))
fig_time.add_trace(go.Scatter(x=T_range*365, y=tp, name="EU Put",
    line=dict(color="#f85149", width=2.5)))
fig_time.add_vline(x=T_days, line_color="rgba(255,255,255,0.5)",
    line_dash="dot", annotation_text=f"T={T_days}d", annotation_position="top right")
fig_time.update_layout(
    height=380, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Days to expiry", yaxis_title="Option price ($)",
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_time, width="stretch")

st.markdown("---")

# ── Monte Carlo simulation ─────────────────────────────────────────────────────
st.subheader("Monte Carlo price paths")
with st.spinner("Running Monte Carlo simulation…"):
    df = monte_carlo(stock, sigma=sigma, r=mu_hat, delta_t=1/365,
                     n_steps=n_steps, n_simulations=n_sims)

t_axis = np.arange(len(df)) / 365

fig_mc = go.Figure()
show_n = min(n_sims, 300)
for col in df.columns[:show_n]:
    fig_mc.add_trace(go.Scatter(
        x=t_axis, y=df[col],
        mode="lines",
        line=dict(width=0.5, color="rgba(55,138,221,0.15)"),
        showlegend=False,
    ))

final_prices = df.iloc[-1]
fig_mc.add_trace(go.Scatter(
    x=t_axis, y=df.mean(axis=1),
    mode="lines", name="Mean path",
    line=dict(color="#f0883e", width=2.5),
))
pct5  = df.quantile(0.05, axis=1)
pct95 = df.quantile(0.95, axis=1)
fig_mc.add_trace(go.Scatter(
    x=np.concatenate([t_axis, t_axis[::-1]]),
    y=np.concatenate([pct95, pct5[::-1]]),
    fill="toself", fillcolor="rgba(120,200,255,0.07)",
    line=dict(color="rgba(0,0,0,0)"),
    name="5th–95th percentile",
))
fig_mc.add_hline(y=S, line_color="rgba(255,255,255,0.4)",
    line_dash="dot", annotation_text="S₀")
fig_mc.update_layout(
    height=450, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Years", yaxis_title="Price ($)",
    legend=dict(orientation="h", y=1.08),
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig_mc, width="stretch")

# Distribution of final prices
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=80,
    marker_color="rgba(55,138,221,0.6)", marker_line_color="#378ADD",
    marker_line_width=0.5, name="Final price"))
fig_hist.add_vline(x=float(final_prices.mean()), line_color="#f0883e",
    line_width=2, annotation_text=f"Mean ${final_prices.mean():.2f}",
    annotation_position="top right")
fig_hist.add_vline(x=float(final_prices.median()), line_color="#3fb950",
    line_width=2, line_dash="dash",
    annotation_text=f"Median ${final_prices.median():.2f}",
    annotation_position="top left")
fig_hist.add_vline(x=K, line_color="rgba(255,200,0,0.8)",
    line_width=1.5, line_dash="dot",
    annotation_text=f"K=${K:.2f}", annotation_position="bottom right")
fig_hist.update_layout(
    height=360, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Final price ($)", yaxis_title="Count",
    title="Distribution of simulated final prices",
    margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_hist, width="stretch")

# MC-estimated option prices
mc_call = np.exp(-r * T) * np.mean(np.maximum(final_prices - K, 0))
mc_put  = np.exp(-r * T) * np.mean(np.maximum(K - final_prices, 0))
m1, m2 = st.columns(2)
m1.metric("MC European Call (estimate)", f"${mc_call:.4f}",
          delta=f"vs BS: ${mc_call - eu_call:+.4f}")
m2.metric("MC European Put (estimate)",  f"${mc_put:.4f}",
          delta=f"vs BS: ${mc_put - eu_put:+.4f}")

st.markdown("---")

# ── Historical μ / σ rolling ──────────────────────────────────────────────────
st.subheader(f"Rolling μ̂ & σ̂ history ({roll_years} years, monthly windows)")

today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
with st.spinner("Computing rolling estimates…"):
    periods, mu_roll, sig_roll = rolling_mu_sigma(ticker, today_str, roll_years)

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=periods, y=[m * 100 for m in mu_roll],
    name="μ̂ (%)", line=dict(color="#79c0ff", width=2), mode="lines+markers",
    marker=dict(size=3)))
fig_roll.add_trace(go.Scatter(x=periods, y=[s * 100 for s in sig_roll],
    name="σ̂ (%)", line=dict(color="#f0883e", width=2), mode="lines+markers",
    marker=dict(size=3)))
fig_roll.update_layout(
    height=380, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Window start", yaxis_title="Annualised estimate (%)",
    xaxis=dict(tickangle=45, nticks=20),
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=40, r=20, t=30, b=70),
)
st.plotly_chart(fig_roll, width="stretch")

# ── Historical price chart ─────────────────────────────────────────────────────
st.subheader("Historical price")
fig_hist_price = go.Figure()
fig_hist_price.add_trace(go.Scatter(
    x=price_series.index, y=price_series.values,
    fill="tozeroy", fillcolor="rgba(55,138,221,0.1)",
    line=dict(color="#378ADD", width=1.8),
    name=ticker,
))
fig_hist_price.update_layout(
    height=320, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Date", yaxis_title="Price ($)",
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_hist_price, width="stretch")

st.caption("Black-Scholes (European) · CRR Binomial Tree (American) · GBM Monte Carlo · Data via yfinance")
