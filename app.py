import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# ---------------------------------------------------
# ASSUMPTION REGISTRY
# ---------------------------------------------------

MODEL_VERSION = "1.2"

ASSUMPTIONS = {

"optimizer": {
"return_shrinkage": 0.4,
"diversification_penalty": 0.05
},

"simulation": {
"paths": 2000
}

}

# ---------------------------------------------------
# ASSET UNIVERSE
# ---------------------------------------------------

ASSETS = {

"World Equity": {"return":0.075,"vol":0.16,"cat":"Equity"},
"US Equity": {"return":0.078,"vol":0.17,"cat":"Equity"},
"Emerging Markets": {"return":0.08,"vol":0.22,"cat":"Equity"},
"Global REIT": {"return":0.065,"vol":0.18,"cat":"Equity"},
"Euro Gov Bonds": {"return":0.03,"vol":0.06,"cat":"Bond"},
"Corp Bonds": {"return":0.035,"vol":0.07,"cat":"Bond"},
"Gold": {"return":0.04,"vol":0.15,"cat":"Commodity"},
"Cash": {"return":0.02,"vol":0.01,"cat":"Cash"}

}

CORR_RULES = {

("Equity","Equity"):0.85,
("Equity","Bond"):0.2,
("Equity","Commodity"):0.05,
("Bond","Bond"):0.6,
("Bond","Commodity"):0.1,
("Commodity","Commodity"):0.5,
("Cash","Equity"):0.05,
("Cash","Bond"):0.2,
("Cash","Commodity"):0.05,
("Cash","Cash"):1.0

}

# ---------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------

def build_corr(selected):

    mat = pd.DataFrame(index=selected,columns=selected)

    for a in selected:
        for b in selected:

            ca = ASSETS[a]["cat"]
            cb = ASSETS[b]["cat"]

            if a == b:
                mat.loc[a,b] = 1
            else:
                key = (ca,cb)

                if key not in CORR_RULES:
                    key = (cb,ca)

                mat.loc[a,b] = CORR_RULES[key]

    return mat.astype(float)

# ---------------------------------------------------
# PORTFOLIO OPTIMIZER
# ---------------------------------------------------

def optimize_portfolio(names,target_return):

    rets = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink*rets + (1-shrink)*np.mean(rets)

    max_possible = max(rets)

    if target_return > max_possible:
        raise ValueError(
            f"Target return too high. Max achievable ≈ {max_possible*100:.1f}%"
        )

    corr = build_corr(names)
    cov = np.diag(vols) @ corr.values @ np.diag(vols)

    div_penalty = ASSUMPTIONS["optimizer"]["diversification_penalty"]

    def objective(w):

        vol = np.sqrt(w.T @ cov @ w)

        concentration = np.sum(w**2)

        return vol + div_penalty * concentration

    constraints = [

        {"type":"eq","fun":lambda w: np.sum(w)-1},
        {"type":"ineq","fun":lambda w: w@rets - target_return}

    ]

    bounds = [(0,1)]*len(names)

    res = minimize(
        objective,
        np.ones(len(names))/len(names),
        bounds=bounds,
        constraints=constraints
    )

    if not res.success:
        raise ValueError("Target return not achievable with selected assets")

    w = res.x

    port_r = w @ rets
    port_v = np.sqrt(w.T @ cov @ w)

    return w,port_r,port_v

# ---------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------

def simulate(mu,sigma,years,start,monthly,growth):

    sims = ASSUMPTIONS["simulation"]["paths"]

    paths = np.zeros((sims,years+1))
    paths[:,0] = start

    for s in range(sims):

        wealth = start

        for t in range(1,years+1):

            r = np.random.normal(mu,sigma)

            contrib = monthly*12*((1+growth)**(t-1))

            wealth = wealth*(1+r) + contrib

            paths[s,t] = wealth

    return paths

# ---------------------------------------------------
# INSIGHT ENGINE
# ---------------------------------------------------

def compute_insights(paths,return_rate,monthly):

    median = np.median(paths,axis=0)

    monthly_growth = median*(return_rate/12)

    tipping = None

    for i,g in enumerate(monthly_growth):

        if g >= monthly:
            tipping = i
            break

    return median,monthly_growth,tipping

# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------

def validate_engine(weights,volatility,paths):

    if abs(sum(weights)-1) > 1e-6:
        raise ValueError("Weights invalid")

    if volatility <= 0:
        raise ValueError("Volatility invalid")

    returns = paths[:,1:] / paths[:,:-1] - 1

    if (returns < -1).any():
        raise ValueError("Impossible return generated")

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("LU Wealth Architect")

selected = st.multiselect(
"Select Assets",
list(ASSETS.keys()),
default=list(ASSETS.keys())[:5]
)

target_return_pct = st.number_input(
"Target Return %",
min_value=3.0,
max_value=10.0,
value=6.5,
step=0.1
)

target_return = target_return_pct / 100

initial = st.number_input("Initial Capital €",10000,5000000,100000)

monthly = st.number_input("Monthly Investment €",0,20000,3000)

growth = st.slider("Contribution Growth %",0,10,3)/100

years = st.slider("Horizon (years)",1,40,20)

# ---------------------------------------------------
# RUN MODEL
# ---------------------------------------------------

if st.button("Build Plan"):

    try:

        w,port_r,port_v = optimize_portfolio(selected,target_return)

        paths = simulate(port_r,port_v,years,initial,monthly,growth)

        validate_engine(w,port_v,paths)

        median,monthly_growth,tipping = compute_insights(paths,port_r,monthly)

    except Exception as e:

        st.error(str(e))
        st.stop()

    st.header("Investment Plan")

    df = pd.DataFrame({

    "Asset":selected,
    "Weight":w,
    "Invest Today €":w*initial,
    "Monthly €":w*monthly

    })

    st.dataframe(df.style.format({

    "Weight":"{:.1%}",
    "Invest Today €":"€{:,.0f}",
    "Monthly €":"€{:,.0f}"

    }))

    st.metric("Expected Wealth",f"€{median[-1]:,.0f}")

    st.write(f"Portfolio Return: {port_r*100:.2f}%")
    st.write(f"Portfolio Volatility: {port_v*100:.2f}%")

    if tipping:
        st.success(f"Compounding overtakes saving around year {tipping}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=monthly_growth,name="Monthly Growth"))
    fig.add_trace(go.Scatter(y=[monthly]*len(monthly_growth),name="Monthly Saving"))

    st.plotly_chart(fig,use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(y=median,name="Median Wealth"))

    st.plotly_chart(fig2,use_container_width=True)