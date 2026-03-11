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

# --- 2. DATASET & CORRELATION MATRIX ---
asset_list = [
    "World Equity", "All-World", "World Small Cap", 
    "US (S&P 500)", "Europe 600", "Emerging Mkts", 
    "Euro Gov Bonds", "Corp Bonds", "Cash/MM", 
    "Global REITs", "Gold", "Crypto"
]

default_market_data = {
    "Global Core": {"World Equity": [8.5, 15.0], "All-World": [8.2, 16.5], "World Small Cap": [9.0, 19.0]},
    "Regional Equity": {"US (S&P 500)": [10.5, 17.0], "Europe 600": [7.2, 16.0], "Emerging Mkts": [8.0, 21.0]},
    "Bonds & Cash": {"Euro Gov Bonds": [3.2, 5.0], "Corp Bonds": [4.5, 7.5], "Cash/MM": [3.3, 0.8]},
    "Alts & REITs": {"Global REITs": [7.5, 18.0], "Gold": [8.0, 14.0], "Crypto": [25.0, 68.0]}
}

# 2026 Institutional Correlation Baseline
corr_df = pd.DataFrame(0.3, index=asset_list, columns=asset_list)
for a in asset_list: corr_df.loc[a, a] = 1.0
equity_group = ["World Equity", "All-World", "World Small Cap", "US (S&P 500)", "Europe 600", "Emerging Mkts"]
for a in equity_group:
    for b in equity_group:
        if a != b: corr_df.loc[a, b] = 0.85
corr_df.loc["Gold", equity_group] = 0.1
corr_df.loc["Euro Gov Bonds", equity_group] = -0.1
corr_df.loc["Crypto", equity_group] = 0.4
corr_df.loc["Cash/MM", asset_list] = 0.05
corr_df.loc["Cash/MM", "Cash/MM"] = 1.0

# --- 3. SIDEBAR ---
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
    def cat_toggle(c=cat):
        for a in default_market_data[c]: st.session_state[f"chk_{a}"] = st.session_state[f"m_{c}"]
    with st.sidebar.expander(cat, expanded=(cat == "Global Core")):
        st.checkbox(f"Toggle {cat}", key=f"m_{cat}", on_change=cat_toggle, value=True)
        for asset, params in assets.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: active = st.checkbox(asset, key=f"chk_{asset}", value=True)
            with c2: u_ret = st.number_input("R", 0.0, 100.0, float(params[0]), key=f"ret_{asset}", label_visibility="collapsed")
            with c3: u_vol = st.number_input("V", 0.1, 100.0, float(params[1]), key=f"vol_{asset}", label_visibility="collapsed")
            if active:
                f_rets.append(u_ret/100); f_vols.append(u_vol/100); f_names.append(asset)

# --- 4. NAVIGATION ---
st.title(f"🇱🇺 {risk_profile} Architect")
mode = st.radio("Choose Strategy Engine:", ["Original (Static Monthly)", "V2 (Annual Step-Up)"], horizontal=True)
st.divider()

# --- 5. MAIN INPUTS ---
c1, c2 = st.columns(2)
with c1:
    target_ret = st.number_input("Target Return %", 1.0, 25.0, 8.5) / 100
    current_val = st.number_input("Initial Capital (€)", value=100000)
    step_up = st.slider("Annual Contribution Increase %", 0, 15, 3) / 100 if mode == "V2 (Annual Step-Up)" else 0.0
with c2:
    monthly_add = st.number_input("Monthly Savings (€)", value=3000)
    horizon = st.slider("Horizon (Years)", 1, 40, 20)
    inflation = st.number_input("Inflation %", value=1.8) / 100
conf_level = st.select_slider("Stress Confidence", [0.90, 0.95, 0.99], 0.95)

# --- 6. ENGINE (RESTORING COVARIANCE) ---
def run_analysis():
    n = len(f_rets)
    if n == 0 or target_ret > max(f_rets): return
    active_corr = corr_df.loc[f_names, f_names].values
    vol_diag = np.diag(f_vols)
    cov = vol_diag @ active_corr @ vol_diag
    
    def obj(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, {'type': 'ineq', 'fun': lambda w: np.sum(w*f_rets)-target_ret}]
    res = minimize(obj, [1/n]*n, bounds=[(0, risk_caps[risk_profile])]*n, constraints=cons)
    
    if res.success:
        r_an, v_an = np.sum(res.x*f_rets), np.sqrt(np.dot(res.x.T, np.dot(cov, res.x)))
        # Calculate Risk Contribution %: (w * Cov * w.T) / Total_Var
        risk_cont = (res.x * (cov @ res.x)) / (v_an**2)
        y = np.arange(0, horizon + 1)
        wth, inv = [current_val], [current_val]
        for t in range(1, horizon + 1):
            yearly_contri = (monthly_add * 12) * ((1 + step_up)**(t-1))
            wth.append((wth[-1] * (1 + r_an)) + yearly_contri)
            inv.append(inv[-1] + yearly_contri)
        gns = [0] + [wth[i]-wth[i-1]-((monthly_add*12)*((1+step_up)**(max(0,i-1)))) for i in range(1, len(y))]
        tip = next((t for t, g in enumerate(gns) if g > (monthly_add*12)*((1+step_up)**(max(0,t-1)))), None)
        st.session_state.results = {"w": res.x, "ret": r_an, "vol": v_an, "wth": wth, "inv": inv, "y": y, "tip": tip, "names": f_names, "risk_c": risk_cont}

if st.button("🚀 Calculate Strategy"): run_analysis()

# --- 7. OUTPUT ---
if st.session_state.results:
    res = st.session_state.results
    st.divider()
    t1, t2, t3, t4, t5 = st.tabs(["📊 Portfolio", "📈 Wealth Path", "🔔 Risk", "⚖️ Rebalance", "🧪 Advanced Risk"])
    
    with t1:
        tw, tp = res["wth"][-1], res["inv"][-1]
        st.metric("Final Wealth", f"€{tw:,.0f}", f"+{(tw/tp-1)*100:.1f}% Gains")
        mix_df = pd.DataFrame({"Asset": res["names"], "Target %": res["w"], "Lump Sum": res["w"]*current_val, "Monthly Buy": res["w"]*monthly_add})
        st.table(mix_df[mix_df["Target %"] > 0.005].style.format({"Target %":"{:.1%}", "Lump Sum":"€{:,.0f}", "Monthly Buy":"€{:,.0f}"}))

    with t2:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=res["y"], y=res["wth"], name="Wealth", fill='tozeroy'))
        fig_p.add_trace(go.Scatter(x=res["y"], y=res["inv"], name="Principal", line=dict(color='black', dash='dash')))
        st.plotly_chart(fig_p, use_container_width=True)

    with t3: # Normal distribution plot
        mu, sigma = res["ret"], res["vol"]
        x = np.linspace(mu-4*sigma, mu+4*sigma, 100)
        st.plotly_chart(go.Figure(go.Scatter(x=x, y=norm.pdf(x, mu, sigma), fill='tozeroy', line_color='#2ecc71')), use_container_width=True)

    with t5: # Advanced Risk View
        st.subheader("Diversification Benefit Analysis")
        # Waterfall Chart for Risk Contribution
        risk_df = pd.DataFrame({"Asset": res["names"], "Risk Contribution %": res["risk_c"]})
        risk_df = risk_df[risk_df["Risk Contribution %"] > 0.001]
        fig_w = go.Figure(go.Waterfall(
            name = "Risk", orientation = "v",
            measure = ["relative"] * len(risk_df),
            x = risk_df["Asset"],
            y = risk_df["Risk Contribution %"] * 100,
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_w.update_layout(title="Where is your Risk coming from?", showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)
        st.info("💡 **Diversification Insight:** If an asset's 'Risk Contribution %' is lower than its 'Target Weight %', it is actively mitigating your overall risk.")

    # Narrative Section
    st.divider()
    real_ret = res["ret"] - inflation
    payout = (tw * real_ret) / 12
    z = norm.ppf(1 - (1 - conf_level))
    worst = tw * np.exp(-z * res["vol"] * np.sqrt(horizon))
    c_n1, c_n2 = st.columns(2)
    with c_n1:
        st.metric("Monthly Payout", f"€{max(0, payout):,.0f}", help="Inflation-adjusted safe withdrawal.")
    with c_n2:
        st.metric("Stress Floor", f"€{worst:,.0f}", help=f"Value at {conf_level*100}% confidence.")