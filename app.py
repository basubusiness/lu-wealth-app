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
}
.card-label {
font-size:14px;
color:#666;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# ASSUMPTIONS
# ---------------------------------------------------

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
# ASSET UNIVERSE
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
# OPTIMIZER
# ---------------------------------------------------

def optimize_portfolio(names,target):

    original_returns = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink*original_returns + (1-shrink)*np.mean(original_returns)

    max_possible = max(original_returns)

    if target > max_possible:

        st.warning(
            f"Target return too high given diversification rules. "
            f"Maximum achievable return ≈ {max_possible*100:.1f}%. "
            f"Portfolio optimized at this level instead."
        )

        target = max_possible

    if "corr_override" in st.session_state:
        corr = np.array(st.session_state["corr_override"])
    else:
        corr = build_corr(names).values

    cov = np.diag(vols) @ corr @ np.diag(vols)

    max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]

    equity_idx = [i for i,a in enumerate(names) if ASSETS[a]["cat"]=="Equity"]
    bond_idx = [i for i,a in enumerate(names) if ASSETS[a]["cat"]=="Bond"]
    real_idx = [i for i,a in enumerate(names) if ASSETS[a]["cat"]=="Real"]

    constraints = [
        {"type":"eq","fun":lambda w: np.sum(w)-1},
        {"type":"ineq","fun":lambda w: w@rets-target},

        {"type":"ineq","fun":lambda w: np.sum(w[equity_idx]) - 0.4},
        {"type":"ineq","fun":lambda w: 0.8 - np.sum(w[equity_idx])},

        {"type":"ineq","fun":lambda w: np.sum(w[bond_idx]) - 0.1},
        {"type":"ineq","fun":lambda w: 0.5 - np.sum(w[bond_idx])},

        {"type":"ineq","fun":lambda w: np.sum(w[real_idx]) - 0.05}
    ]

    bounds = [(0,max_w)]*len(names)

    def objective(w):
        vol = np.sqrt(w.T@cov@w)
        concentration = np.sum(w**2)
        return vol + 0.02*concentration

    res = minimize(
        objective,
        np.ones(len(names))/len(names),
        bounds=bounds,
        constraints=constraints
    )

    w = res.x
    port_r = w@rets
    port_v = np.sqrt(w.T@cov@w)

    return w,port_r,port_v


# ---------------------------------------------------
# MONTE CARLO
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
# SCENARIOS
# ---------------------------------------------------

def scenario_paths(paths,confidence):

    lower = (1-confidence)/2
    upper = 1-lower

    worst = np.percentile(paths,lower*100,axis=0)
    median = np.percentile(paths,50,axis=0)
    best = np.percentile(paths,upper*100,axis=0)

    return worst,median,best


# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

st.title("LU Wealth Architect")

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    initial = st.number_input("Initial Capital",10000,5000000,100000)

with c2:
    monthly = st.number_input("Monthly Saving",0,20000,3000)

with c3:
    years = st.slider("Horizon (years)",1,40,20)

with c4:
    target_pct = st.number_input("Target Return %",3.0,10.0,6.5)

with c5:
    growth_pct = st.slider("Saving Growth %",0,10,3)

confidence = st.slider("Projection confidence level",80,99,90)/100

target = target_pct/100
growth = growth_pct/100

assets = st.multiselect(
"Assets",
list(ASSETS.keys()),
default=list(ASSETS.keys())[:5]
)

# reset correlation override if asset universe changes
if "corr_override" in st.session_state:
    if st.session_state["corr_override"].shape[0] != len(assets):
        del st.session_state["corr_override"]


# ---------------------------------------------------
# BUILD PLAN
# ---------------------------------------------------

if st.button("Build Plan"):

    w,port_r,port_v = optimize_portfolio(assets,target)

    paths = simulate(port_r,port_v,years,initial,monthly,growth)

    worst,median,best = scenario_paths(paths,confidence)

    years_axis = list(range(len(median)))

    total_invested = initial
    for t in range(1, years + 1):
        total_invested += monthly * 12 * ((1 + growth) ** (t-1))

    growth_value = median[-1] - total_invested
    monthly_income = median[-1] * port_r / 12

    monthly_growth = median*(port_r/12)

    tipping = None
    for i,g in enumerate(monthly_growth):
        if g >= monthly:
            tipping = i
            break

    plan = pd.DataFrame({
        "Asset":assets,
        "Weight":w,
        "Invest Now":w*initial,
        "Monthly":w*monthly
    })

    plan = plan[plan["Weight"]>0.01]

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Plan","Projection","Risk","Rebalance","Engine"])

    # ---------------------------------------------------
    # ENGINE TAB
    # ---------------------------------------------------

    with tab5:

        st.subheader("Model assumptions")
        df = pd.DataFrame(ASSETS).T
        st.dataframe(df)

        st.subheader("Correlation matrix")

        if "corr_override" not in st.session_state:
            st.session_state["corr_override"] = build_corr(assets)

        corr = st.session_state["corr_override"].copy()

        edited_corr = st.data_editor(
            corr,
            use_container_width=True,
            key="corr_editor"
        )

        st.session_state["corr_override"] = edited_corr
