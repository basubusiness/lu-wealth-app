import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# ---------------------------------------------------
# CARD STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.card {
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    background-color:#fafafa;
    text-align:center;
}
.card-value {
    font-size:28px;
    font-weight:600;
    color:#1f2937;
}
.card-label {
    font-size:14px;
    color:#6b7280;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# ASSET UNIVERSE (Original Data + Logic)
# ---------------------------------------------------
ASSETS = {
    "World Equity":{"return":0.075,"vol":0.16,"cat":"Equity"},
    "US Equity":{"return":0.078,"vol":0.17,"cat":"Equity"},
    "Emerging Markets":{"return":0.08,"vol":0.22,"cat":"Equity"},
    "Global REIT":{"return":0.065,"vol":0.18,"cat":"Real"},
    "Euro Gov Bonds":{"return":0.03,"vol":0.06,"cat":"Bond"},
    "Corp Bonds":{"return":0.035,"vol":0.07,"cat":"Bond"},
    "Gold":{"return":0.065,"vol":0.17,"cat":"Real"},
    "Cash":{"return":0.02,"vol":0.01,"cat":"Cash"}
}

CORR_RULES = {
    ("Equity","Equity"):0.85,
    ("Equity","Bond"):0.2,
    ("Equity","Real"):0.15,
    ("Bond","Bond"):0.6,
    ("Bond","Real"):0.1,
    ("Real","Real"):0.5,
    ("Cash","Equity"):0.05,
    ("Cash","Bond"):0.2,
    ("Cash","Real"):0.05,
    ("Cash","Cash"):1.0
}

ASSUMPTIONS = {
    "optimizer":{
        "return_shrinkage":0.4,
        "div_penalty":0.02,
        "max_asset_weight":0.4
    },
    "simulation":{
        "paths":2000
    }
}

# ---------------------------------------------------
# HELPER FUNCTIONS (Original)
# ---------------------------------------------------
def build_corr(selected):
    mat = pd.DataFrame(index=selected, columns=selected)
    for a in selected:
        for b in selected:
            ca, cb = ASSETS[a]["cat"], ASSETS[b]["cat"]
            if a == b:
                mat.loc[a,b] = 1.0
            else:
                key = (ca, cb) if (ca, cb) in CORR_RULES else (cb, ca)
                mat.loc[a,b] = CORR_RULES.get(key, 0.3) # Default 0.3 if missing
    return mat.astype(float)

def objective(w, cov):
    vol = np.sqrt(w.T @ cov @ w)
    concentration = np.sum(w**2)
    return vol + 0.02 * concentration

def optimize_portfolio(names, target):
    original_returns = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink * original_returns + (1 - shrink) * np.mean(original_returns)

    if "corr_override" in st.session_state and st.session_state["corr_override"].shape[0] == len(names):
        corr = np.array(st.session_state["corr_override"])
    else:
        corr = build_corr(names).values

    cov = np.diag(vols) @ corr @ np.diag(vols)
    max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]

    equity_idx = [i for i, a in enumerate(names) if ASSETS[a]["cat"] == "Equity"]
    bond_idx = [i for i, a in enumerate(names) if ASSETS[a]["cat"] == "Bond"]
    real_idx = [i for i, a in enumerate(names) if ASSETS[a]["cat"] == "Real"]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    ret_constraint = {"type": "ineq", "fun": lambda w: w @ rets - target}
    constraints.append(ret_constraint)

    if equity_idx:
        constraints += [{"type": "ineq", "fun": lambda w: np.sum(w[equity_idx]) - 0.4},
                        {"type": "ineq", "fun": lambda w: 0.8 - np.sum(w[equity_idx])}]
    if bond_idx:
        constraints += [{"type": "ineq", "fun": lambda w: np.sum(w[bond_idx]) - 0.1},
                        {"type": "ineq", "fun": lambda w: 0.5 - np.sum(w[bond_idx])}]
    if real_idx:
        constraints += [{"type": "ineq", "fun": lambda w: np.sum(w[real_idx]) - 0.05}]

    res = minimize(objective, np.ones(len(names)) / len(names), args=(cov,),
                   bounds=[(0, max_w)] * len(names), constraints=constraints, method="SLSQP")

    if not res.success:
        st.warning("⚠️ Target return could not be achieved. Showing lowest-risk portfolio.")
        fallback_constraints = [c for c in constraints if c != ret_constraint]
        res = minimize(objective, np.ones(len(names)) / len(names), args=(cov,),
                       bounds=[(0, max_w)] * len(names), constraints=fallback_constraints, method="SLSQP")

    w = res.x
    w[w < 0.002] = 0
    if np.sum(w) > 0: w = w / np.sum(w)
    return w, w @ rets, np.sqrt(w.T @ cov @ w)

def simulate(mu, sigma, years, start, monthly, growth):
    sims = ASSUMPTIONS["simulation"]["paths"]
    df = 5 
    scale_factor = np.sqrt((df - 2) / df)
    mu_log = mu - (0.5 * sigma**2)
    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start
    for t in range(1, years + 1):
        shocks = mu_log + np.random.standard_t(df, (sims,)) * sigma * scale_factor
        contrib = monthly * 12 * ((1 + growth)**(t - 1))
        paths[:, t] = paths[:, t-1] * np.exp(shocks) + contrib
    return paths

# ---------------------------------------------------
# UI SETUP
# ---------------------------------------------------
st.title("LU Wealth Architect")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: initial = st.number_input("Initial Capital", 10000, 5000000, 100000)
with c2: monthly = st.number_input("Monthly Saving", 0, 20000, 3000)
with c3: years = st.slider("Horizon (years)", 1, 40, 20)
with c4: target_pct = st.number_input("Target Return %", 3.0, 10.0, 6.5)
with c5: growth_pct = st.slider("Saving Growth %", 0, 10, 3)

target, growth = target_pct / 100, growth_pct / 100
confidence = 0.90

# --- IMPROVED GROUPED CHOOSER ---
st.subheader("Asset Selection")
selected_assets = []
# Group assets by category automatically
categories = sorted(list(set(data["cat"] for data in ASSETS.values())))
cols = st.columns(len(categories))

for i, cat in enumerate(categories):
    with cols[i]:
        st.markdown(f"**{cat}**")
        cat_assets = [name for name, data in ASSETS.items() if data["cat"] == cat]
        for asset in cat_assets:
            # Default check some assets to avoid empty state
            is_default = asset in ["World Equity", "Euro Gov Bonds", "Gold"]
            if st.checkbox(asset, value=is_default, key=f"check_{asset}"):
                selected_assets.append(asset)

# ---------------------------------------------------
# RUN & TABS
# ---------------------------------------------------
if st.button("Build Plan") and selected_assets:
    w, port_r, port_v = optimize_portfolio(selected_assets, target)
    paths = simulate(port_r, port_v, years, initial, monthly, growth)
    
    worst, median, best = np.percentile(paths, [10, 50, 90], axis=0)
    years_axis = list(range(len(median)))
    
    total_invested = initial + sum([monthly * 12 * ((1 + growth)**i) for i in range(years)])
    growth_value = median[-1] - total_invested
    monthly_income = median[-1] * port_r / 12
    tipping = next((i for i, g in enumerate(median * (port_r / 12)) if g >= monthly), None)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Projection", "Risk", "Rebalance", "Engine"])

    with tab1:
        plan = pd.DataFrame({"Asset": selected_assets, "Weight": w, "Invest Now": w * initial, "Monthly": w * monthly})
        plan = plan[plan["Weight"] > 0.01]
        st.dataframe(plan.style.format({"Weight": "{:.1%}", "Invest Now": "€{:,.0f}", "Monthly": "€{:,.0f}"}), use_container_width=True)
        
        m_cols = st.columns(5)
        metrics = [(f"€{median[-1]:,.0f}", "Final Value"), (f"{port_r*100:.1f}%", "Return"), (f"€{growth_value:,.0f}", "Growth"), (f"Year {tipping}" if tipping else "N/A", "Tipping Point"), (f"€{monthly_income:,.0f}", "Income")]
        for i, (val, label) in enumerate(metrics):
            m_cols[i].markdown(f'<div class="card"><div class="card-value">{val}</div><div class="card-label">{label}</div></div>', unsafe_allow_html=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_axis, y=best, name="Best Case", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=years_axis, y=median, name="Expected", line=dict(color="blue", width=3)))
        fig.add_trace(go.Scatter(x=years_axis, y=worst, name="Worst Case", line=dict(color="red")))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(go.Figure(go.Histogram(x=paths[:, -1], nbinsx=40)), use_container_width=True)

    with tab4:
        st.subheader("Rebalancing")
        current = {a: st.number_input(f"Current {a}", value=0.0) for a in selected_assets}
        total_curr = sum(current.values()) if sum(current.values()) > 0 else initial
        rebal_df = pd.DataFrame({"Asset": selected_assets, "Target €": [w[i]*total_curr for i,a in enumerate(selected_assets)], "Current €": list(current.values())})
        rebal_df["Buy/Sell"] = rebal_df["Target €"] - rebal_df["Current €"]
        st.dataframe(rebal_df)

    with tab5:
        st.dataframe(pd.DataFrame(ASSETS).T)
        if "corr_override" not in st.session_state: st.session_state["corr_override"] = build_corr(selected_assets)
        st.session_state["corr_override"] = st.data_editor(st.session_state["corr_override"])
else:
    st.info("Select assets and click 'Build Plan'.")
