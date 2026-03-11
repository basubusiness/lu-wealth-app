import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

# --- 1. CONFIG ---
st.set_page_config(page_title="LU Wealth Architect", layout="wide")
if 'results' not in st.session_state:
    st.session_state.results = None

# --- 2. EXTENDED DATASET ---
default_market_data = {
    "Global Core": {"World Equity": [8.5, 15.0], "All-World": [8.2, 16.5], "World Small Cap": [9.0, 19.0]},
    "Regional Equity": {"US (S&P 500)": [10.5, 17.0], "Europe 600": [7.2, 16.0], "Emerging Mkts": [8.0, 21.0]},
    "Sectors": {"AI & Tech": [12.5, 24.0], "Global Energy": [7.5, 19.0], "Healthcare": [8.0, 14.0]},
    "Bonds & Cash": {"Euro Gov Bonds": [3.2, 5.0], "Corp Bonds": [4.5, 7.5], "Cash/MM": [3.3, 0.8]},
    "Alts & REITs": {"Global REITs": [7.5, 18.0], "Gold": [8.0, 14.0], "Crypto": [25.0, 68.0]}
}

all_names = [a for cat in default_market_data.values() for a in cat.keys()]

# --- 3. SESSION STATE FOR CORRELATION ---
if 'corr_matrix' not in st.session_state:
    df = pd.DataFrame(0.3, index=all_names, columns=all_names)
    for a in all_names: df.loc[a, a] = 1.0
    # Equity Correlations
    eq = ["World Equity", "All-World", "World Small Cap", "US (S&P 500)", "Europe 600", "Emerging Mkts", "AI & Tech", "Healthcare"]
    for a in eq:
        for b in eq: 
            if a != b: df.loc[a, b] = 0.85
    df.loc["AI & Tech", "US (S&P 500)"] = 0.95
    df.loc["Gold", eq] = 0.1
    df.loc["Euro Gov Bonds", eq] = -0.1
    st.session_state.corr_matrix = df

# --- 4. SIDEBAR ---
st.sidebar.header("📂 JustETF Market Hub")
def global_toggle():
    val = st.session_state.global_master
    for cat, assets in default_market_data.items():
        st.session_state[f"m_{cat}"] = val
        for asset in assets: st.session_state[f"chk_{asset}"] = val

st.sidebar.checkbox("🌐 SELECT ALL ASSETS", key="global_master", on_change=global_toggle, value=True)
st.sidebar.divider()
risk_profile = st.sidebar.select_slider("Risk Profile", ["Conservative", "Balanced", "Aggressive"], "Balanced")
risk_caps = {"Conservative": 0.25, "Balanced": 0.50, "Aggressive": 0.90}

f_rets, f_vols, f_names = [], [], []
for cat, assets in default_market_data.items():
    with st.sidebar.expander(cat, expanded=(cat == "Global Core")):
        st.checkbox(f"Toggle {cat}", key=f"m_{cat}", value=True)
        for asset, params in assets.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: active = st.checkbox(asset, key=f"chk_{asset}", value=True)
            with c2: u_ret = st.number_input("R", 0.0, 100.0, float(params[0]), key=f"ret_{asset}", label_visibility="collapsed")
            with c3: u_vol = st.number_input("V", 0.1, 100.0, float(params[1]), key=f"vol_{asset}", label_visibility="collapsed")
            if active: f_rets.append(u_ret/100); f_vols.append(u_vol/100); f_names.append(asset)

# --- 5. NAVIGATION ---
st.title(f"🇱🇺 {risk_profile} Architect")
mode = st.radio("Choose Strategy Engine:", ["Original (Static Monthly)", "V2 (Annual Step-Up)"], horizontal=True)
st.divider()

# --- 6. MAIN INPUTS (FIXED SCOPE) ---
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

# --- 7. ENGINE ---
def run_analysis():
    n = len(f_rets)
    if n == 0 or target_ret > max(f_rets): return
    active_corr = st.session_state.corr_matrix.loc[f_names, f_names].values
    vol_diag = np.diag(f_vols)
    cov = vol_diag @ active_corr @ vol_diag
    
    def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, {'type': 'ineq', 'fun': lambda w: np.sum(w*f_rets)-target_ret}]
    res = minimize(obj, [1/n]*n, bounds=[(0, risk_caps[risk_profile])]*n, constraints=cons)
    
    if res.success:
        r_an, v_an = np.sum(res.x*f_rets), np.sqrt(np.dot(res.x.T, np.dot(cov, res.x)))
        risk_cont = (res.x * (cov @ res.x)) / (v_an**2)
        y, wth, inv = np.arange(0, horizon+1), [current_val], [current_val]
        for t in range(1, horizon+1):
            yc = (monthly_add * 12) * ((1 + step_up)**(t-1))
            wth.append((wth[-1]*(1+r_an)) + yc); inv.append(inv[-1] + yc)
        if adj_inflation:
            wth = [w/((1+inflation)**i) for i, w in enumerate(wth)]
            inv = [v/((1+inflation)**i) for i, v in enumerate(inv)]
        
        # Fixed Tipping Point for Step-Up context
        g_series = [0] + [wth[i]-wth[i-1]-((monthly_add*12)*((1+step_up)**(max(0,i-1)))/((1+inflation)**i if adj_inflation else 1)) for i in range(1, len(y))]
        tip = next((t for t, g in enumerate(g_series) if g > ((monthly_add*12)*((1+step_up)**(max(0,t-1)))/((1+inflation)**t if adj_inflation else 1))), None)

        st.session_state.results = {"w": res.x, "ret": r_an, "vol": v_an, "wth": wth, "inv": inv, "y": y, "names": f_names, "risk_c": risk_cont, "tip": tip}

if st.button("🚀 Calculate Strategy"): run_analysis()

# --- 8. OUTPUT ---
if st.session_state.results:
    res = st.session_state.results
    t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Portfolio", "📈 Wealth Path", "🔔 Risk", "⚖️ Rebalance", "🧪 Advanced Risk", "⚙️ Engine Room"])
    
    with t1:
        tw, tp = res["wth"][-1], res["inv"][-1]
        st.metric("Final Wealth", f"€{tw:,.0f}", f"+{(tw/tp-1)*100:.1f}% Profit")
        final_monthly = monthly_add * ((1 + step_up)**(horizon-1))
        df = pd.DataFrame({"Asset": res["names"], "Target %": res["w"], "Lump Sum": res["w"]*current_val, "Year 1 Monthly": res["w"]*monthly_add, f"Year {horizon} Monthly": res["w"]*final_monthly})
        st.table(df[df["Target %"] > 0.005].style.format({"Target %":"{:.1%}", "Lump Sum":"€{:,.0f}", "Year 1 Monthly":"€{:,.0f}", f"Year {horizon} Monthly":"€{:,.0f}"}))

    with t2:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=res["y"], y=res["wth"], name="Portfolio", fill='tozeroy'))
        fig_p.add_trace(go.Scatter(x=res["y"], y=res["inv"], name="Invested", line=dict(color='black', dash='dash')))
        if res["tip"]: fig_p.add_vline(x=res["tip"], line_dash="dot", line_color="#2ecc71", annotation_text="Tipping Point")
        st.plotly_chart(fig_p, use_container_width=True)

    with t5:
        st.subheader("Where is your Volatility coming from?")
        risk_df = pd.DataFrame({"Asset": res["names"], "RC": res["risk_c"]})
        fig = go.Figure(go.Waterfall(orientation="v", x=risk_df["Asset"], y=risk_df["RC"]*100, measure=["relative"]*len(risk_df)))
        st.plotly_chart(fig, use_container_width=True)

    with t6:
        st.subheader("Interactive Correlation Matrix")
        sub_corr = st.data_editor(st.session_state.corr_matrix.loc[f_names, f_names])
        if st.button("💾 Save Correlation Changes"):
            st.session_state.corr_matrix.update(sub_corr)
            st.success("Correlations updated! Re-run 'Calculate Strategy' to see the impact.")

    # --- NARRATIVE (FIXED) ---
    st.divider()
    st.header("💎 Strategic Narrative")
    real_ret = res["ret"] - inflation
    payout = (tw * real_ret) / 12
    # Fix NameError by ensuring conf_level is accessible
    worst = tw * np.exp(-norm.ppf(conf_level) * res["vol"] * np.sqrt(horizon))
    
    c_n1, c_n2 = st.columns(2)
    with c_n1:
        st.subheader("🏦 Passive Income")
        st.metric("Monthly Payout (Real)", f"€{max(0, payout):,.0f}", help="The Safe Withdrawal amount in today's purchasing power.")
        if res["tip"]: 
            st.write(f"✅ **Tipping Point: Year {res['tip']}**")
            st.caption("The year your portfolio growth exceeds your annual savings.")
    with c_n2:
        st.subheader("🛡️ Safety & Tax")
        st.metric("Stress Floor", f"€{worst:,.0f}", help="Conservative wealth estimate during a crisis.")
        st.write(f"💰 **LU Tax Advantage: €{(res['wth'][-1]-res['inv'][-1]) * 0.40:,.0f}**")
        st.caption("Estimated savings from the 0% long-term capital gains tax in Luxembourg.")