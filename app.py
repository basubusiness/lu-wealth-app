import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------
# DATA
# ---------------------------------------------------

default_market_data = {
    "Global Equity": {
        "World Equity": [7.5, 16],
        "US (S&P 500)": [7.8, 17],
        "Europe 600": [7.0, 16],
        "Emerging Markets": [8.0, 22],
        "World Small Cap": [8.5, 20]
    },
    "Diversifiers": {
        "Global REITs": [6.5, 18],
        "Gold": [4.0, 15]
    },
    "Defensive": {
        "Euro Gov Bonds": [3.0, 6],
        "Corp Bonds": [3.5, 7],
        "Cash": [2.0, 1]
    }
}

all_assets = [a for cat in default_market_data.values() for a in cat]

# ---------------------------------------------------
# CORRELATION MATRIX DEFAULT
# ---------------------------------------------------

if "corr_matrix" not in st.session_state:

    corr = pd.DataFrame(0.25, index=all_assets, columns=all_assets)

    for a in all_assets:
        corr.loc[a, a] = 1

    equities = [
        "World Equity",
        "US (S&P 500)",
        "Europe 600",
        "Emerging Markets",
        "World Small Cap"
    ]

    for a in equities:
        for b in equities:
            if a != b:
                corr.loc[a, b] = 0.85

    corr.loc["Gold", equities] = 0.05
    corr.loc[equities, "Gold"] = 0.05

    corr.loc["Global REITs", equities] = 0.7
    corr.loc[equities, "Global REITs"] = 0.7

    corr.loc["Euro Gov Bonds", equities] = 0.2
    corr.loc[equities, "Euro Gov Bonds"] = 0.2

    st.session_state.corr_matrix = corr

# ---------------------------------------------------
# MATRIX CLEANER
# ---------------------------------------------------

def clean_corr_matrix(corr):

    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1)

    eigvals, eigvecs = np.linalg.eigh(corr)

    eigvals[eigvals < 0] = 0

    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return pd.DataFrame(corr_psd, index=corr.index, columns=corr.columns)

# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def monte_carlo_paths(mu, sigma, years, start, contrib, step_up, sims=2000):

    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start

    for s in range(sims):

        value = start

        for t in range(1, years + 1):

            r = np.random.normal(mu, sigma)

            yearly_contribution = contrib * 12 * ((1 + step_up) ** (t - 1))

            value = value * (1 + r) + yearly_contribution

            paths[s, t] = value

    return paths

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Portfolio Inputs")

risk_profile = st.sidebar.select_slider(
    "Risk Profile",
    ["Conservative", "Balanced", "Aggressive"],
    value="Balanced"
)

risk_caps = {
    "Conservative": 0.25,
    "Balanced": 0.45,
    "Aggressive": 0.8
}

crisis_mode = st.sidebar.checkbox("Stress Crisis Correlations")

# ---------------------------------------------------
# ASSET INPUT
# ---------------------------------------------------

selected_names = []
returns = []
vols = []

for category, assets in default_market_data.items():

    with st.sidebar.expander(category, expanded=True):

        for asset, params in assets.items():

            c1, c2, c3 = st.columns([2,1,1])

            with c1:
                active = st.checkbox(asset, value=True)

            with c2:
                r = st.number_input("Return", value=float(params[0]), key=f"ret_{asset}")

            with c3:
                v = st.number_input("Vol", value=float(params[1]), key=f"vol_{asset}")

            if active:
                selected_names.append(asset)
                returns.append(r / 100)
                vols.append(v / 100)

# ---------------------------------------------------
# MAIN INPUTS
# ---------------------------------------------------

st.title("🇱🇺 Wealth Architect")

col1, col2 = st.columns(2)

with col1:

    target_return = st.number_input("Target Return %", 1.0, 15.0, 6.5) / 100
    initial_capital = st.number_input("Initial Capital €", value=100000)
    step_up = st.slider("Contribution Growth %", 0, 10, 3) / 100

with col2:

    monthly_savings = st.number_input("Monthly Savings €", value=3000)
    horizon = st.slider("Years", 1, 40, 20)
    inflation = st.number_input("Inflation %", value=2.0) / 100

# ---------------------------------------------------
# CALCULATE BUTTON
# ---------------------------------------------------

if st.button("Calculate Strategy"):

    if len(selected_names) == 0:
        st.error("Select at least one asset.")
        st.stop()

    rets = np.array(returns)
    vols = np.array(vols)

    # Return shrinkage
    avg_ret = np.mean(rets)
    shrink = 0.4
    shrunk_returns = shrink * rets + (1 - shrink) * avg_ret

    if target_return > max(shrunk_returns):
        st.error("Target return is higher than any asset return.")
        st.stop()

    corr = st.session_state.corr_matrix.loc[selected_names, selected_names]

    if crisis_mode:

        equity_mask = corr.index.str.contains("Equity|S&P|Europe|Emerging|Small")

        corr.loc[equity_mask, equity_mask] = 0.95

    corr = clean_corr_matrix(corr)

    cov = np.diag(vols) @ corr.values @ np.diag(vols)

    # ---------------------------------------------------
    # OPTIMIZATION
    # ---------------------------------------------------

    n = len(rets)

    def portfolio_vol(w):
        return np.sqrt(w.T @ cov @ w)

    constraints = [
        {'type':'eq','fun':lambda w: np.sum(w) - 1},
        {'type':'ineq','fun':lambda w: w @ shrunk_returns - target_return}
    ]

    bounds = [(0, risk_caps[risk_profile])] * n

    result = minimize(portfolio_vol, np.ones(n)/n, bounds=bounds, constraints=constraints)

    if not result.success:
        st.error("Optimizer could not find a feasible portfolio.")
        st.stop()

    weights = result.x

    port_return = weights @ shrunk_returns
    port_vol = np.sqrt(weights.T @ cov @ weights)

    # ---------------------------------------------------
    # MONTE CARLO
    # ---------------------------------------------------

    paths = monte_carlo_paths(
        port_return,
        port_vol,
        horizon,
        initial_capital,
        monthly_savings,
        step_up
    )

    median = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    st.session_state.results = {
        "weights": weights,
        "names": selected_names,
        "return": port_return,
        "vol": port_vol,
        "years": np.arange(horizon + 1),
        "median": median,
        "p10": p10,
        "p90": p90
    }

# ---------------------------------------------------
# OUTPUT
# ---------------------------------------------------

if st.session_state.results:

    r = st.session_state.results

    tab1, tab2 = st.tabs(["Portfolio", "Monte Carlo"])

    with tab1:

        st.metric("Expected Return", f"{r['return']*100:.2f}%")
        st.metric("Volatility", f"{r['vol']*100:.2f}%")

        df = pd.DataFrame({
            "Asset": r["names"],
            "Weight": r["weights"]
        })

        st.table(df.style.format({"Weight":"{:.1%}"}))

    with tab2:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=r["years"],
            y=r["median"],
            name="Median"
        ))

        fig.add_trace(go.Scatter(
            x=r["years"],
            y=r["p10"],
            name="10% Worst Case",
            line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=r["years"],
            y=r["p90"],
            name="10% Best Case",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig, use_container_width=True)