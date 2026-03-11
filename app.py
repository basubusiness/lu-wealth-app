import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="LU Wealth Architect", layout="wide")
if 'results' not in st.session_state:
    st.session_state.results = None

# --- 2. DATASET ---
default_market_data = {
    "Global Core": {"World Equity": [8.5, 15.0], "All-World": [8.2, 16.5], "World Small Cap": [9.0, 19.0]},
    "Regional Equity": {"US (S&P 500)": [10.5, 17.0], "Europe 600": [7.2, 16.0], "Emerging Mkts": [8.0, 21.0]},
    "Bonds & Cash": {"Euro Gov Bonds": [3.2, 5.0], "Corp Bonds": [4.5, 7.5], "Cash/MM": [3.3, 0.8]},
    "Alts & REITs": {"Global REITs": [7.5, 18.0], "Gold": [8.0, 14.0], "Crypto": [25.0, 68.0]}
}

# --- 3. SIDEBAR (MOBILE FRIENDLY) ---
st.sidebar.header("📂 JustETF Market Hub")

def global_toggle():
    val = st.session_state.global_master
    for cat, assets in default_market_data.items():
        st.session_state[f"m_{cat}"] = val
        for asset in assets: st.session_state[f"chk_{asset}"] = val

st.sidebar.checkbox("🌐 SELECT ALL ASSETS", key="global_master", on_change=global_toggle)
st.sidebar.divider()

risk_profile = st.sidebar.select_slider("Risk Profile", ["Conservative", "Balanced", "Aggressive"], "Balanced")
risk_caps = {"Conservative": 0.25, "Balanced": 0.50, "Aggressive": 0.90}

f_rets, f_vols, f_names = [], [], []

for cat, assets in default_market_data.items():
    def cat_toggle(c=cat):
        for a in default_market_data[c]: st.session_state[f"chk_{a}"] = st.session_state[f"m_{c}"]

    with st.sidebar.expander(cat, expanded=(cat == "Global Core")):
        st.checkbox(f"Toggle {cat}", key=f"m_{cat}", on_change=cat_toggle)
        st.caption("Asset | Ret% | Risk%")
        for asset, params in assets.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: 
                # Default selection to prevent start-up errors
                active = st.checkbox(asset, key=f"chk_{asset}", value=(asset == "All-World"))
            with c2: u_ret = st.number_input("R", 0.0, 100.0, float(params[0]), key=f"ret_{asset}", label_visibility="collapsed")
            with c3: u_vol = st.number_input("V", 0.1, 100.0, float(params[1]), key=f"vol_{asset}", label_visibility="collapsed")
            if active:
                f_rets.append(u_ret/100); f_vols.append(u_vol/100); f_names.append(asset)

# --- 4. MAIN INPUTS ---
st.title(f"🇱🇺 {risk_profile} Architect")
c1, c2 = st.columns(2) # Stacking 2x2 for mobile better than 3 wide
with c1:
    target_ret = st.number_input("Target Return %", 1.0, 25.0, 8.5) / 100
    current_val = st.number_input("Initial Capital (€)", value=100000)
with c2:
    monthly_add = st.number_input("Monthly Savings (€)", value=3000)
    horizon = st.slider("Horizon (Years)", 1, 40, 20)

conf_level = 0.95
inflation = 0.018

# --- 5. THE ENGINE ---
def run_analysis():
    n = len(f_rets)
    if n == 0: return 
    if target_ret > max(f_rets):
        st.warning(f"Target unattainable. Max: {max(f_rets)*100:.1f}%")
        return

    cov = np.outer(f_vols, f_vols) * (np.eye(n)*0.7 + 0.3)
    def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, {'type': 'ineq', 'fun': lambda w: np.sum(w*f_rets)-target_ret}]
    res = minimize(obj, [1/n]*n, bounds=[(0, risk_caps[risk_profile])]*n, constraints=cons)
    
    if res.success:
        r_an, v_an = np.sum(res.x*f_rets), np.sqrt(np.dot(res.x.T, np.dot(cov, res.x)))
        y = np.arange(0, horizon + 1)
        wth = [current_val*(1+r_an)**t + (monthly_add*12*((1+r_an)**t-1)/r_an if t>0 else 0) for t in y]
        inv = [current_val + (monthly_add * 12 * t) for t in y]
        tip = next((t for t, g in enumerate([0] + [wth[i]-wth[i-1]-(monthly_add*12) for i in range(1, len(y))]) if g > (monthly_add*12)), None)
        st.session_state.results = {"w": res.x, "ret": r_an, "vol": v_an, "wth": wth, "inv": inv, "y": y, "tip": tip, "names": f_names}

# Auto-run if no results exist
if st.session_state.results is None and len(f_rets) > 0:
    run_analysis()

if st.button("🚀 Update Strategy"):
    run_analysis()

# --- 6. OUTPUT ---
if st.session_state.results:
    res = st.session_state.results
    st.divider()
    t1, t2, t3 = st.tabs(["📊 Mix", "📈 Wealth", "⚖️ Rebalance"])

    with t1:
        tw, tp = res["wth"][-1], res["inv"][-1]
        m1, m2 = st.columns(2)
        m1.metric("Final Net Worth", f"€{tw:,.0f}")
        m2.metric("Total Invested", f"€{tp:,.0f}")
        
        st.write("**Strategy Table**")
        df = pd.DataFrame({
            "Asset": res["names"], 
            "Target %": res["w"], 
            "Lump Sum": res["w"]*current_val,
            "Monthly": res["w"]*monthly_add
        })
        st.table(df[df["Target %"] > 0.005].style.format({"Target %":"{:.1%}", "Lump Sum":"€{:,.0f}", "Monthly":"€{:,.0f}"}))

    with t2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["y"], y=res["wth"], name="Wealth", fill='tozeroy', line_color='#1f77b4'))
        fig.add_trace(go.Scatter(x=res["y"], y=res["inv"], name="Principal", line=dict(color='black', dash='dash')))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.write("**Current Portfolio Values**")
        total_act = 0; act_vals = []
        for i, n in enumerate(res["names"]):
            if res["w"][i] > 0.005:
                v = st.number_input(f"{n} (€)", value=float(res["w"][i]*current_val), key=f"re_{n}")
                act_vals.append(v); total_act += v
        if st.button("⚖️ Run Rebalance"):
            rdf = pd.DataFrame({"Asset":[n for i,n in enumerate(res["names"]) if res["w"][i]>0.005], "Actual":act_vals, "Target %":[w for w in res["w"] if w>0.005]})
            rdf["Action"] = (rdf["Target %"] * total_act) - rdf["Actual"]
            st.table(rdf.style.format({"Actual":"€{:,.0f}", "Target %":"{:.1%}", "Action":"€{:,.0f}"}))

    st.divider()
    real_ret = res["ret"] - inflation
    payout = (tw * real_ret) / 12
    st.info(f"**Monthly Payout:** €{max(0, payout):,.0f}")
    st.success(f"**Tax Shield:** €{(tw-tp)*0.40:,.0f}")
