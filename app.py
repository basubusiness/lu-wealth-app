import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

default_market_data = {
    "Global Core": {
        "World Equity": [7.5, 16],
        "All-World": [7.3, 16.5],
        "World Small Cap": [8.5, 20]
    },
    "Regional Equity": {
        "US (S&P 500)": [7.8, 17],
        "Europe 600": [7.0, 16],
        "Emerging Mkts": [8.0, 22]
    },
    "Alts & Diversifiers": {
        "Global REITs": [6.5, 18],
        "Gold": [4.0, 15]
    },
    "Bonds & Cash": {
        "Euro Gov Bonds": [3.0, 6],
        "Corp Bonds": [3.5, 7],
        "Cash/MM": [2.0, 1]
    }
}

all_assets = [a for cat in default_market_data.values() for a in cat]

# ---------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------

if "corr_matrix" not in st.session_state:

    corr = pd.DataFrame(0.25, index=all_assets, columns=all_assets)

    for a in all_assets:
        corr.loc[a, a] = 1

    equities = [
        "World Equity",
        "All-World",
        "World Small Cap",
        "US (S&P 500)",
        "Europe 600",
        "Emerging Mkts"
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

# ---------------------------------------------------------
# CORR MATRIX CLEANER
# ---------------------------------------------------------

def clean_corr_matrix(corr):

    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 0] = 0

    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return pd.DataFrame(corr_psd, index=corr.index, columns=corr.columns)

# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

def monte_carlo_paths(mu, sigma, years, start, contrib, step_up, sims=2000):

    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start

    for s in range(sims):

        val = start

        for t in range(1, years + 1):

            r = np.random.normal(mu, sigma)

            yc = contrib * 12 * ((1 + step_up) ** (t - 1))

            val = val * (1 + r) + yc

            paths[s, t] = val

    return paths

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("📂 Market Hub")

risk_profile = st.sidebar.select_slider(
    "Risk Profile",
    ["Conservative", "Balanced", "Aggressive"],
    "Balanced"
)

risk_caps = {
    "Conservative": 0.25,
    "Balanced": 0.45,
    "Aggressive": 0.8
}

crisis_mode = st.sidebar.checkbox("Stress Crisis Correlations")

# ---------------------------------------------------------
# ASSET INPUT
# ---------------------------------------------------------

selected_names = []
rets = []
vols = []

for cat, assets in default_market_data.items():

    with st.sidebar.expander(cat, expanded=(cat == "Global Core")):

        for asset, params in assets.items():

            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                active = st.checkbox(asset, value=True)

            with c2:
                r = st.number_input(
                    "R",
                    value=float(params[0]),
                    key=f"ret_{asset}",
                    label_visibility="collapsed"
                )

            with c3:
                v = st.number_input(
                    "V",
                    value=float(params[1]),
                    key=f"vol_{asset}",
                    label_visibility="collapsed"
                )

            if active:
                selected_names.append(asset)
                rets.append(r / 100)
                vols.append(v / 100)

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

st.title("🇱🇺 Wealth Architect")

c1, c2 = st.columns(2)

with c1:

    target_return = st.number_input("Target Return %", 1.0, 15.0, 6.5) / 100
    initial_capital = st.number_input("Initial Capital (€)", value=100000)
    step_up = st.slider("Annual Contribution Increase %", 0, 10, 3) / 100

with c2:

    monthly_savings = st.number_input("Monthly Savings (€)", value=3000)
    horizon = st.slider("Horizon (Years)", 1, 40, 20)
    inflation = st.number_input("Inflation %", value=2.0) / 100

# ---------------------------------------------------------
# CALCULATE
# ---------------------------------------------------------

if st.button("🚀 Calculate Strategy"):

    rets = np.array(rets)
    vols = np.array(vols)

    avg_ret = np.mean(rets)
    shrink = 0.4

    shrunk_rets = shrink * rets + (1 - shrink) * avg_ret

    if target_return > max(shrunk_rets):
        st.error("Target return higher than available assets.")
        st.stop()

    corr = st.session_state.corr_matrix.loc[selected_names, selected_names]

    if crisis_mode:

        eq_mask = corr.index.str.contains("Equity|S&P|Europe|Emerging|World")

        corr.loc[eq_mask, eq_mask] = 0.95

    corr = clean_corr_matrix(corr)

    cov = np.diag(vols) @ corr.values @ np.diag(vols)

    n = len(rets)

    def obj(w):
        return np.sqrt(w.T @ cov @ w)

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: w @ shrunk_rets - target_return}
    ]

    bounds = [(0, risk_caps[risk_profile])] * n

    res = minimize(obj, np.ones(n) / n, bounds=bounds, constraints=cons)

    weights = res.x

    port_return = weights @ shrunk_rets
    port_vol = np.sqrt(weights.T @ cov @ weights)

    risk_contrib = (weights * (cov @ weights)) / (port_vol ** 2)

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
        "risk_contrib": risk_contrib,
        "years": np.arange(horizon + 1),
        "median": median,
        "p10": p10,
        "p90": p90,
        "paths": paths
    }

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

if st.session_state.results:

    r = st.session_state.results

    tabs = st.tabs([
        "📊 Portfolio",
        "📈 Wealth Path",
        "⚠️ Risk",
        "🔄 Rebalance",
        "🧠 Advanced Risk",
        "⚙️ Engine Room"
    ])

    # Portfolio
    with tabs[0]:

        st.metric("Expected Return", f"{r['return']*100:.2f}%")
        st.metric("Volatility", f"{r['vol']*100:.2f}%")

        df = pd.DataFrame({
            "Asset": r["names"],
            "Weight": r["weights"]
        })

        st.table(df.style.format({"Weight": "{:.1%}"}))

    # Wealth Path
    with tabs[1]:

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=r["years"], y=r["median"], name="Median"))
        fig.add_trace(go.Scatter(x=r["years"], y=r["p10"], name="Pessimistic", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=r["years"], y=r["p90"], name="Optimistic", line=dict(dash="dash")))

        st.plotly_chart(fig, use_container_width=True)

        final_median = r["median"][-1]
        final_p10 = r["p10"][-1]

        total_contrib = monthly_savings * 12 * horizon

        growth_component = final_median - (initial_capital + total_contrib)

        st.subheader("Insights")

        st.info(f"""
Expected portfolio value after {horizon} years: **€{final_median:,.0f}**

Pessimistic scenario (10th percentile): **€{final_p10:,.0f}**
""")

        st.write(f"""
Total invested: **€{initial_capital + total_contrib:,.0f}**

Estimated market growth contribution: **€{growth_component:,.0f}**
""")

    # Risk
    with tabs[2]:

        mu = r["return"]
        sigma = r["vol"]

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mu, sigma), fill="tozeroy"))

        st.plotly_chart(fig, use_container_width=True)

        st.write(f"""
Typical yearly returns may fall roughly between  
**{(mu-sigma)*100:.1f}% and {(mu+sigma)*100:.1f}%**
""")

    # Rebalance
    with tabs[3]:

        total = 0
        actual_vals = []

        for n, w in zip(r["names"], r["weights"]):

            val = st.number_input(
                f"Current {n} (€)",
                value=float(w * initial_capital),
                key=f"reb_{n}"
            )

            actual_vals.append(val)
            total += val

        if st.button("Generate Rebalance Plan"):

            reb_df = pd.DataFrame({
                "Asset": r["names"],
                "Actual": actual_vals,
                "Target %": r["weights"]
            })

            reb_df["Action"] = (reb_df["Target %"] * total) - reb_df["Actual"]

            st.table(reb_df.style.format({
                "Actual": "€{:,.0f}",
                "Target %": "{:.1%}",
                "Action": "€{:,.0f}"
            }))

    # Advanced Risk
    with tabs[4]:

        rc_df = pd.DataFrame({
            "Asset": r["names"],
            "Risk Contribution": r["risk_contrib"]
        })

        fig = go.Figure(go.Waterfall(
            x=rc_df["Asset"],
            y=rc_df["Risk Contribution"] * 100
        ))

        st.plotly_chart(fig, use_container_width=True)

    # Engine Room
    with tabs[5]:

        st.write("Editable Correlation Matrix")

        st.data_editor(
            st.session_state.corr_matrix.loc[r["names"], r["names"]],
            use_container_width=True
        )