import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# ---------------------------------------------------
# CARD STYLE (Original)
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
# ASSETS & LOGIC (Original)
# ---------------------------------------------------
ASSETS = {
    "World Equity": {"return": 0.075, "vol": 0.16, "cat": "Equity"},
    "US Equity": {"return": 0.078, "vol": 0.17, "cat": "Equity"},
    "Emerging Markets": {"return": 0.080, "vol": 0.22, "cat": "Equity"},
    "Global Small Cap": {"return": 0.082, "vol": 0.19, "cat": "Equity"},
    "Global REIT": {"return": 0.060, "vol": 0.19, "cat": "Real"},
    "Gold": {"return": 0.065, "vol": 0.17, "cat": "Real"},
    "Broad Commodities": {"return": 0.040, "vol": 0.20, "cat": "Real"},
    "Euro Gov Bonds": {"return": 0.030, "vol": 0.06, "cat": "Bond"},
    "Corp Bonds": {"return": 0.035, "vol": 0.07, "cat": "Bond"},
    "Global Inflation Bonds": {"return": 0.032, "vol": 0.05, "cat": "Bond"},
    "Cash": {"return": 0.020, "vol": 0.01, "cat": "Cash"}
}
CORR_RULES = {
    ("Equity", "Equity"): 0.85, ("Equity", "Bond"): 0.15, ("Equity", "Real"): 0.20,
    ("Bond", "Bond"): 0.60, ("Bond", "Real"): 0.15, ("Real", "Real"): 0.30,
    ("Cash", "Equity"): 0.05, ("Cash", "Bond"): 0.10, ("Cash", "Real"): 0.05, ("Cash", "Cash"): 1.0
}

# --- GLOBAL INITIALIZATION ---
if "asset_settings" not in st.session_state:
    st.session_state["asset_settings"] = pd.DataFrame(ASSETS).T.reset_index().rename(columns={'index': 'Asset'})

if "init" not in st.session_state:
    for a in ASSETS.keys(): st.session_state[f"asset_{a}"] = True
    for c in set(d["cat"] for d in ASSETS.values()): st.session_state[f"master_{c}"] = True
    st.session_state["init"] = True
    
ASSUMPTIONS = {"optimizer":{"return_shrinkage":0.4, "div_penalty":0.02, "max_asset_weight":0.4}, "simulation":{"paths":2000}}

def build_corr(selected):
    mat = pd.DataFrame(index=selected, columns=selected)
    for a in selected:
        for b in selected:
            ca, cb = ASSETS[a]["cat"], ASSETS[b]["cat"]
            if a == b: mat.loc[a,b] = 1.0
            else:
                key = (ca, cb) if (ca, cb) in CORR_RULES else (cb, ca)
                mat.loc[a,b] = CORR_RULES.get(key, 0.3)
    return mat.astype(float)

def optimize_portfolio(names, target_r):
    # PULL FROM EDITABLE TABLE
    df = st.session_state["asset_settings"].set_index("Asset")
    rets_list = df.loc[names, "return"].values
    vols_list = df.loc[names, "vol"].values
        
    original_returns = np.array(rets_list)
    vols = np.array(vols_list)
    
    shrink = 0.95 
    rets = shrink * original_returns + (1 - shrink) * np.mean(original_returns)
    
    corr = st.session_state["corr_override"].values
    cov = np.diag(vols) @ corr @ np.diag(vols)

    def objective(w, c, r, t):
        port_return = w @ r
        port_vol = np.sqrt(w.T @ c @ w) + 1e-6
        rf = 0.02
        sharpe_loss = -((port_return - rf) / port_vol)
        target_penalty = 500.0 * np.maximum(0, t - port_return)**2 
        div_penalty = 0.10 * np.sum(w**2)
        return sharpe_loss + target_penalty + div_penalty

    res = minimize(
        objective,
        np.ones(len(names)) / len(names), 
        args=(cov, rets, target_r), 
        bounds=[(0, 0.50)] * len(names), 
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    )
    
    w = np.round(res.x, 4)
    w[w < 0.01] = 0 
    if np.sum(w) > 0: w = w / np.sum(w)
    
    return w, w @ rets, np.sqrt(w.T @ cov @ w)
    
def simulate(mu, sigma, years, start, monthly, growth):
    sims = 2000
    df = 5
    scale = np.sqrt((df - 2) / df)
    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start
    for t in range(1, years + 1):
        shocks = (mu - 0.5 * sigma**2) + np.random.standard_t(df, sims) * sigma * scale
        contrib = monthly * 12 * ((1 + growth)**(t - 1))
        paths[:, t] = paths[:, t-1] * np.exp(shocks) + contrib
    return paths

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("LU Wealth Architect")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: initial = st.number_input("Initial Capital", 10000, 5000000, 100000)
with c2: monthly = st.number_input("Monthly Saving", 0, 20000, 3000)
with c3: years = st.slider("Horizon (years)", 1, 40, 20)
with c4: target_pct = st.number_input("Target Return %", 3.0, 10.0, 6.5)
with c5: growth_pct = st.slider("Saving Growth %", 0, 10, 3)

target, growth = target_pct / 100, growth_pct / 100

if selected_assets:
    df_lookup = st.session_state["asset_settings"].set_index("Asset")
    # Get the max return from the currently checked assets in the Tactical Engine
    max_available_return = df_lookup.loc[selected_assets, "return"].max()
    
    if target > max_available_return:
        st.error(f"⚠️ **Target Unachievable**: Your {target_pct}% target exceeds the highest return in your selection ({max_available_return*100:.1f}%).")
    elif target > 0.08:
        st.warning(f"⚡ **Aggressive**: A {target_pct}% target typically requires high Equity exposure.")
else:
    st.info("Select assets below to validate your target return.")

st.subheader("Asset Selection")

with st.expander("Configure Asset Universe", expanded=False):
    def sync_category(cat_name, assets_in_cat):
        m_key = f"master_{cat_name}"
        for a in assets_in_cat: st.session_state[f"asset_{a}"] = st.session_state[m_key]

    selected_assets = []
    cats = sorted(list(set(d["cat"] for d in ASSETS.values())))
    cols = st.columns(len(cats))

    for i, cat in enumerate(cats):
        with cols[i]:
            st.markdown(f"**{cat}**")
            cat_assets = [n for n, d in ASSETS.items() if d["cat"] == cat]
            st.checkbox(f"All {cat}", key=f"master_{cat}", on_change=sync_category, args=(cat, cat_assets))
            for a in cat_assets:
                if st.checkbox(a, key=f"asset_{a}"):
                    selected_assets.append(a)

if "corr_override" not in st.session_state or st.session_state.get("last_selected_corr") != sorted(selected_assets):
    st.session_state["corr_override"] = build_corr(sorted(selected_assets))
    st.session_state["last_selected_corr"] = sorted(selected_assets)

if st.button("Build Plan") and selected_assets:
    w, port_r, port_v = optimize_portfolio(selected_assets, target)
    paths = simulate(port_r, port_v, years, initial, monthly, growth)
    st.session_state['results'] = {'w': w, 'port_r': port_r, 'port_v': port_v, 'paths': paths, 'assets': selected_assets}

if 'results' in st.session_state:
    res = st.session_state['results']
    w, port_r, port_v, paths, selected_assets = res['w'], res['port_r'], res['port_v'], res['paths'], res['assets']
    
    worst, median, best = np.percentile(paths, [10, 50, 90], axis=0)
    total_invested = initial + sum([monthly * 12 * ((1 + growth)**i) for i in range(years)])
    growth_value = median[-1] - total_invested
    monthly_income = median[-1] * port_r / 12
    tipping = next((i for i, g in enumerate(median * (port_r / 12)) if g >= monthly), None)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Projection", "Risk", "Rebalance", "Engine"])

    with tab1:
        plan = pd.DataFrame({"Asset": selected_assets, "Weight": w, "Invest Now": w * initial, "Monthly": w * monthly})
        st.dataframe(plan[plan["Weight"] > 0.01].style.format({"Weight": "{:.1%}", "Invest Now": "€{:,.0f}", "Monthly": "€{:,.0f}"}), use_container_width=True)
        
        m_cols = st.columns(5)
        metrics = [
            (f"€{median[-1]:,.0f}", f"Final Portfolio Value ({years}y)"),
            (f"{port_r*100:.1f}%", "Typical Yearly Growth"),
            (f"€{growth_value:,.0f}", f"Growth on €{total_invested:,.0f} invested"),
            (f"Year {tipping}" if tipping else "N/A", "Compounding Tipping Point"),
            (f"€{monthly_income:,.0f}", "Est. Monthly Income")
        ]
        for i, (val, label) in enumerate(metrics):
            m_cols[i].markdown(f'<div class="card"><div class="card-value">{val}</div><div class="card-label">{label}</div></div>', unsafe_allow_html=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=best, name="Best Case", line=dict(color="green")))
        fig.add_trace(go.Scatter(y=median, name="Expected", line=dict(color="blue", width=3)))
        fig.add_trace(go.Scatter(y=worst, name="Worst Case", line=dict(color="red")))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(go.Figure(go.Histogram(x=paths[:, -1], nbinsx=40)), use_container_width=True)

    with tab4:
        st.subheader("Rebalancing")
        with st.form("rebalance_form"):
            current_vals = {}
            reb_cols = st.columns(2)
            for i, a in enumerate(selected_assets):
                recommended_default = float(plan.loc[plan['Asset'] == a, 'Invest Now'].iloc[0])
                with reb_cols[i % 2]:
                    current_vals[a] = st.number_input(f"Current {a}", value=recommended_default, key=f"rebal_input_{a}")
            st.form_submit_button("Update Rebalance Table")
        
        total_curr = sum(current_vals.values())
        rebal_df = pd.DataFrame({"Asset": selected_assets, "Target €": [wi * total_curr for wi in w], "Current €": [current_vals[a] for a in selected_assets]})
        rebal_df["Buy/Sell"] = rebal_df["Target €"] - rebal_df["Current €"]
        st.dataframe(rebal_df.style.format({"Target €": "€{:,.0f}", "Current €": "€{:,.0f}", "Buy/Sell": "€{:,.0f}"}), use_container_width=True)

    with tab5:
        st.header("Tactical Engine Overrides")
        st.info("Edit Asset DNA and Correlations directly in the tables below.")
    
        st.subheader("Asset Parameters")
        st.session_state["asset_settings"] = st.data_editor(
            st.session_state["asset_settings"],
            column_config={
                "return": st.column_config.NumberColumn("Return", format="%.3f"),
                "vol": st.column_config.NumberColumn("Volatility", format="%.3f"),
                "cat": st.column_config.TextColumn("Category", disabled=True),
                "Asset": st.column_config.TextColumn("Asset", disabled=True),
            },
            use_container_width=True,
            hide_index=True
        )
    
        st.divider()
        st.subheader("Correlation Matrix (Editable)")
        st.session_state["corr_override"] = st.data_editor(st.session_state["corr_override"], use_container_width=True)
