import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# Initialize session state keys to prevent "AttributeError"
if 'results' not in st.session_state:
    st.session_state.results = None

# --- 2. DATASET ---
default_market_data = {
    "Global Core": {"World Equity": [8.5, 15.0], "All-World": [8.2, 16.5], "World Small Cap": [9.0, 19.0]},
    "Regional Equity": {"US (S&P 500)": [10.5, 17.0], "Europe 600": [7.2, 16.0], "Emerging Mkts": [8.0, 21.0]},
    "Sectors": {"AI & Tech": [12.5, 24.0], "Global Energy": [7.5, 19.0], "Healthcare": [8.0, 14.0]},
    "Bonds & Cash": {"Euro Gov Bonds": [3.2, 5.0], "Corp Bonds": [4.5, 7.5], "Cash/MM": [3.3, 0.8]},
    "Alts & REITs": {"Global REITs": [7.5, 18.0], "Gold": [8.0, 14.0], "Crypto": [25.0, 68.0]}
}

all_names = [a for cat in default_market_data.values() for a in cat.keys()]

if 'corr_matrix' not in st.session_state:
    df = pd.DataFrame(0.3, index=all_names, columns=all_names)
    for a in all_names: df.loc[a, a] = 1.0
    eq = ["World Equity", "All-World", "World Small Cap", "US (S&P 500)", "Europe 600", "Emerging Mkts", "AI & Tech", "Healthcare"]
    for a in eq:
        for b in eq: 
            if a != b: df.loc[a, b] = 0.85
    st.session_state.corr_matrix = df

# --- 3. SIDEBAR ---
st.sidebar.header("📂 JustETF Market Hub")
risk_profile = st.sidebar.select_slider("Risk Profile", ["Conservative", "Balanced", "Aggressive"], "Balanced")
risk_caps = {"Conservative": 0.25, "Balanced": 0.50, "Aggressive": 0.90}

f_rets, f_vols, f_names = [], [], []
for cat, assets in default_market_data.items():
    with st.sidebar.expander(cat, expanded=(cat == "Global Core")):
        for asset, params in assets.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: active = st.checkbox(asset, key=f"chk_{asset}", value=True)
            with c2: u_ret = st.number_input("R", 0.0, 100.0, float(params[0]), key=f"ret_{asset}", label_visibility="collapsed")
            with c3: u_vol = st.number_input("V", 0.1, 100.0, float(params[1]), key=f"vol_{asset}", label_visibility="collapsed")
            if active: f_rets.append(u_ret/100); f_vols.append(u_vol/100); f_names.append(asset)

# --- 4. INPUTS ---
st.title(f"🇱🇺 {risk_profile} Architect")
mode = st.radio("Choose Strategy Engine:", ["Original (Static Monthly)", "V2 (Annual Step-Up)"], horizontal=True)

c1, c2 = st.columns(2)
with c1:
    target_ret = st.number_input("Target Return %", 1.0, 25.0, 8.5) / 100
    current_val = st.number_input("Initial Capital (€)", value=100000)
    step_up = st.slider("Annual Contribution Increase %", 0, 15, 3) / 100 if mode == "V2 (Annual Step-Up)" else 0.0
with c2:
    monthly_add = st.number_input("Monthly Savings (€)", value=3000)
    horizon = st.slider("Horizon (Years)", 1, 40, 20)
    inflation = st.number_input("Inflation %", value=1.8) / 100
    adj_inflation = st.checkbox("Show Chart in Today's Euros", value=False)
conf_level = st.select_slider("Stress Confidence", [0.90, 0.95, 0.99], 0.95)

# --- 5. ENGINE ---
if st.button("🚀 Calculate Strategy"):
    n = len(f_rets)
    if n > 0 and target_ret <= max(f_rets):
        active_corr = st.session_state.corr_matrix.loc[f_names, f_names].values
        vol_diag = np.diag(f_vols)
        cov = vol_diag @ active_corr @ vol_diag
        
        def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, {'type': 'ineq', 'fun': lambda w: np.sum(w*f_rets)-target_ret}]
        res = minimize(obj, [1/n]*n, bounds=[(0, risk_caps[risk_profile])]*n, constraints=cons)
        
        if res.success:
            r_an, v_an = np.sum(res.x*f_rets), np.sqrt(np.dot(res.x.T, np.dot(cov, res.x)))
            # Fixed Risk Contribution Logic
            risk_cont = (res.x * (cov @ res.x)) / (v_an**2) if v_an > 0 else res.x
            
            y, wth, inv = np.arange(0, horizon+1), [current_val], [current_val]
            for t in range(1, horizon+1):
                yc = (monthly_add * 12) * ((1 + step_up)**(t-1))
                wth.append((wth[-1]*(1+r_an)) + yc); inv.append(inv[-1] + yc)
            
            if adj_inflation:
                wth = [w/((1+inflation)**i) for i, w in enumerate(wth)]
                inv = [v/((1+inflation)**i) for i, v in enumerate(inv)]

            st.session_state.results = {
                "w": res.x, "ret": r_an, "vol": v_an, "wth": wth, "inv": inv, 
                "y": y, "names": f_names, "risk_c": risk_cont, "current_val": current_val
            }

# --- 6. OUTPUT ---
if st.session_state.results:
    res = st.session_state.results
    tabs = st.tabs(["📊 Portfolio", "📈 Wealth Path", "🔔 Risk", "⚖️ Rebalance", "🧪 Advanced Risk", "⚙️ Engine Room"])
    
    with tabs[0]:
        st.write("### Strategy Overview")
        st.metric("Final Wealth", f"€{res['wth'][-1]:,.0f}")
        df_mix = pd.DataFrame({"Asset": res["names"], "Target %": res["w"]})
        st.table(df_mix[df_mix["Target %"] > 0.005].style.format({"Target %": "{:.1%}"}))

    with tabs[2]: # Risk Tab
        st.write("### Volatility Analysis")
        mu, sigma = res["ret"], res["vol"]
        x = np.linspace(mu-4*sigma, mu+4*sigma, 100)
        fig_r = go.Figure(go.Scatter(x=x, y=norm.pdf(x, mu, sigma), fill='tozeroy', name="Prob. Density"))
        st.plotly_chart(fig_r, use_container_width=True)

    with tabs[3]: # Rebalance Tab
        st.write("### Portfolio Alignment")
        total_act = 0; act_vals = []
        for n, w in zip(res["names"], res["w"]):
            if w > 0.005:
                v = st.number_input(f"Current {n} (€)", value=float(w*res["current_val"]), key=f"reb_{n}")
                act_vals.append(v); total_act += v
        if st.button("Generate Rebalance Plan"):
            active_names = [n for n, w in zip(res["names"], res["w"]) if w > 0.005]
            active_weights = [w for w in res["w"] if w > 0.005]
            reb_df = pd.DataFrame({"Asset": active_names, "Actual": act_vals, "Target %": active_weights})
            reb_df["Action"] = (reb_df["Target %"] * total_act) - reb_df["Actual"]
            st.table(reb_df.style.format({"Actual": "€{:,.0f}", "Target %": "{:.1%}", "Action": "€{:,.0f}"}))

    with tabs[4]: # Advanced Risk
        st.subheader("Risk Contribution Waterfall")
        risk_df = pd.DataFrame({"Asset": res["names"], "RC": res["risk_c"]})
        fig_w = go.Figure(go.Waterfall(x=risk_df["Asset"], y=risk_df["RC"]*100, measure=["relative"]*len(risk_df)))
        st.plotly_chart(fig_w, use_container_width=True)

    with tabs[5]: # Engine Room
        st.data_editor(st.session_state.corr_matrix.loc[res["names"], res["names"]])