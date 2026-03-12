import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
if 'results' not in st.session_state:
    st.session_state.results = None

# -----------------------------
# DATASET
# -----------------------------
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
    "Diversifiers": {
        "Gold": [4.0, 15],
        "Global REITs": [6.5, 18]
    },
    "Defensive": {
        "Euro Gov Bonds": [3.0, 6],
        "Corp Bonds": [3.5, 7],
        "Cash/MM": [2.0, 1]
    }
}

all_names = [a for cat in default_market_data.values() for a in cat.keys()]

# -----------------------------
# DEFAULT CORRELATION MATRIX
# -----------------------------
if 'corr_matrix' not in st.session_state:

    df = pd.DataFrame(0.25, index=all_names, columns=all_names)

    for a in all_names:
        df.loc[a, a] = 1

    equities = [
        "World Equity","All-World","World Small Cap",
        "US (S&P 500)","Europe 600","Emerging Mkts"
    ]

    for a in equities:
        for b in equities:
            if a != b:
                df.loc[a,b] = 0.85

    df.loc["Gold", equities] = 0.05
    df.loc[equities, "Gold"] = 0.05

    df.loc["Global REITs", equities] = 0.7
    df.loc[equities, "Global REITs"] = 0.7

    df.loc["Euro Gov Bonds", equities] = 0.2
    df.loc[equities, "Euro Gov Bonds"] = 0.2

    st.session_state.corr_matrix = df

# -----------------------------
# MATRIX SAFETY
# -----------------------------
def clean_corr_matrix(corr):

    corr = (corr + corr.T)/2
    np.fill_diagonal(corr.values,1)

    eigvals, eigvecs = np.linalg.eigh(corr)

    eigvals[eigvals < 0] = 0

    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    corr_psd = pd.DataFrame(
        corr_psd,
        index=corr.index,
        columns=corr.columns
    )

    return corr_psd


# -----------------------------
# MONTE CARLO
# -----------------------------
def monte_carlo_paths(mu, sigma, years, start, contrib, step_up, sims=2000):

    paths = np.zeros((sims, years+1))
    paths[:,0] = start

    for s in range(sims):

        val = start

        for t in range(1,years+1):

            r = np.random.normal(mu, sigma)

            yc = contrib*12*((1+step_up)**(t-1))

            val = val*(1+r) + yc

            paths[s,t] = val

    return paths


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Market Hub")

risk_profile = st.sidebar.select_slider(
    "Risk Profile",
    ["Conservative","Balanced","Aggressive"],
    "Balanced"
)

risk_caps = {
    "Conservative":0.25,
    "Balanced":0.45,
    "Aggressive":0.8
}

crisis_mode = st.sidebar.checkbox("Stress Crisis Correlations")

# -----------------------------
# ASSET INPUTS
# -----------------------------
f_rets=[]
f_vols=[]
f_names=[]

for cat,assets in default_market_data.items():

    with st.sidebar.expander(cat,expanded=True):

        for asset,params in assets.items():

            c1,c2,c3 = st.columns([2,1,1])

            with c1:
                active = st.checkbox(asset,value=True,key=f"chk_{asset}")

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

                f_names.append(asset)
                f_rets.append(r/100)
                f_vols.append(v/100)

# -----------------------------
# INPUTS
# -----------------------------
st.title("🇱🇺 Wealth Architect")

col1,col2 = st.columns(2)

with col1:

    target_ret = st.number_input("Target Return %",1.0,20.0,7.0)/100

    current_val = st.number_input("Initial Capital",value=100000)

    step_up = st.slider("Contribution Growth %",0,10,3)/100

with col2:

    monthly_add = st.number_input("Monthly Savings",value=3000)

    horizon = st.slider("Years",1,40,20)

    inflation = st.number_input("Inflation %",value=2.0)/100


# -----------------------------
# OPTIMIZATION
# -----------------------------
if st.button("Calculate Strategy"):

    n=len(f_rets)

    if n>0:

        # ---- return shrinkage ----
        rets = np.array(f_rets)

        avg_ret = np.mean(rets)

        shrink = 0.4

        shrunk_rets = shrink*rets + (1-shrink)*avg_ret

        # ---- correlations ----
        corr = st.session_state.corr_matrix.loc[f_names,f_names]

        if crisis_mode:

            eq_mask = corr.index.str.contains("Equity|S&P|Europe|Emerging")

            corr.loc[eq_mask,eq_mask] = 0.95

        corr = clean_corr_matrix(corr)

        vol_diag = np.diag(f_vols)

        cov = vol_diag @ corr.values @ vol_diag

        # ---- optimizer ----
        def obj(w):
            return np.sqrt(w.T @ cov @ w)

        cons = [
            {'type':'eq','fun':lambda w:np.sum(w)-1},
            {'type':'ineq','fun':lambda w:w@shrunk_rets-target_ret}
        ]

        bounds = [(0,risk_caps[risk_profile])]*n

        res = minimize(obj,[1/n]*n,bounds=bounds,constraints=cons)

        if res.success:

            w = res.x

            port_ret = w @ shrunk_rets

            port_vol = np.sqrt(w.T @ cov @ w)

            # Monte Carlo
            paths = monte_carlo_paths(
                port_ret,
                port_vol,
                horizon,
                current_val,
                monthly_add,
                step_up
            )

            median = np.median(paths,axis=0)
            p10 = np.percentile(paths,10,axis=0)
            p90 = np.percentile(paths,90,axis=0)

            st.session_state.results = {
                "weights":w,
                "ret":port_ret,
                "vol":port_vol,
                "median":median,
                "p10":p10,
                "p90":p90,
                "years":np.arange(horizon+1),
                "names":f_names
            }

# -----------------------------
# OUTPUT
# -----------------------------
if st.session_state.results:

    r = st.session_state.results

    tabs = st.tabs(["Portfolio","Wealth Simulation"])

    # -----------------------------
    # PORTFOLIO
    # -----------------------------
    with tabs[0]:

        st.metric("Expected Return",f"{r['ret']*100:.2f}%")

        st.metric("Volatility",f"{r['vol']*100:.2f}%")

        df = pd.DataFrame({
            "Asset":r["names"],
            "Weight":r["weights"]
        })

        st.table(df.style.format({"Weight":"{:.1%}"}))


    # -----------------------------
    # MONTE CARLO CHART
    # -----------------------------
    with tabs[1]:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=r["years"],
                y=r["median"],
                name="Median",
                line=dict(width=4)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=r["years"],
                y=r["p10"],
                name="Pessimistic (10%)",
                line=dict(dash="dash")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=r["years"],
                y=r["p90"],
                name="Optimistic (90%)",
                line=dict(dash="dash")
            )
        )

        st.plotly_chart(fig,use_container_width=True)