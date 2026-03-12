import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

# ---------------------------------------------------
# ASSUMPTIONS
# ---------------------------------------------------

ASSUMPTIONS = {
    "optimizer": {
        "return_shrinkage": 0.4,
        "div_penalty": 0.05
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

    mat = pd.DataFrame(index=selected, columns=selected)

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

def optimize_portfolio(names, target_return):

    rets = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink * rets + (1-shrink) * np.mean(rets)

    max_possible = max(rets)

    if target_return > max_possible:
        raise ValueError(
            f"Target return too high. Max achievable ≈ {max_possible*100:.1f}%"
        )

    corr = build_corr(names)
    cov = np.diag(vols) @ corr.values @ np.diag(vols)

    penalty = ASSUMPTIONS["optimizer"]["div_penalty"]

    def objective(w):

        vol = np.sqrt(w.T @ cov @ w)
        concentration = np.sum(w**2)

        return vol + penalty * concentration

    constraints = [
        {"type":"eq","fun":lambda w: np.sum(w)-1},
        {"type":"ineq","fun":lambda w: w@rets - target_return}
    ]

    bounds = [(0,1)] * len(names)

    res = minimize(
        objective,
        np.ones(len(names))/len(names),
        bounds=bounds,
        constraints=constraints
    )

    if not res.success:
        raise ValueError("Target return not achievable")

    w = res.x

    port_r = w @ rets
    port_v = np.sqrt(w.T @ cov @ w)

    return w, port_r, port_v

# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def simulate(mu, sigma, years, start, monthly, growth):

    sims = ASSUMPTIONS["simulation"]["paths"]

    paths = np.zeros((sims, years+1))
    paths[:,0] = start

    for s in range(sims):

        wealth = start

        for t in range(1, years+1):

            r = np.random.normal(mu, sigma)

            contrib = monthly * 12 * ((1+growth)**(t-1))

            wealth = wealth*(1+r) + contrib

            paths[s,t] = wealth

    return paths

# ---------------------------------------------------
# INSIGHTS
# ---------------------------------------------------

def compute_insights(paths, return_rate, monthly):

    median = np.median(paths, axis=0)

    monthly_growth = median * (return_rate/12)

    tipping = None

    for i,g in enumerate(monthly_growth):
        if g >= monthly:
            tipping = i
            break

    return median, tipping

# ---------------------------------------------------
# INPUTS
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

colA,colB,colC = st.columns(3)

with colA:
    initial = st.number_input("Initial Capital €",10000,5000000,100000)

with colB:
    monthly = st.number_input("Monthly Investment €",0,20000,3000)

with colC:
    years = st.slider("Horizon (years)",1,40,20)

growth = st.slider("Contribution Growth %",0,10,3)/100

# ---------------------------------------------------
# RUN MODEL
# ---------------------------------------------------

if st.button("Build Plan"):

    try:

        w, port_r, port_v = optimize_portfolio(selected, target_return)

        paths = simulate(port_r, port_v, years, initial, monthly, growth)

        median, tipping = compute_insights(paths, port_r, monthly)

    except Exception as e:

        st.error(str(e))
        st.stop()

    plan = pd.DataFrame({
        "Asset": selected,
        "Weight": w,
        "Invest Today": w * initial,
        "Monthly Invest": w * monthly
    })

    plan = plan[plan["Weight"] > 0.01]

    # ---------------------------------------------------
    # TABS (STEPPER)
    # ---------------------------------------------------

    tab1, tab2 = st.tabs(["Investment Plan", "Wealth Growth"])

    # ---------------------------------------------------
    # TAB 1 – PLAN
    # ---------------------------------------------------

    with tab1:

        st.subheader("Your Investment Plan")

        col1,col2,col3,col4 = st.columns([3,1,1,1])

        col1.markdown("**Asset**")
        col2.markdown("**Weight**")
        col3.markdown("**Invest Today**")
        col4.markdown("**Monthly**")

        for _,row in plan.iterrows():

            c1,c2,c3,c4 = st.columns([3,1,1,1])

            c1.write(row["Asset"])
            c2.write(f"{row['Weight']*100:.0f}%")
            c3.write(f"€{row['Invest Today']:,.0f}")
            c4.write(f"€{row['Monthly Invest']:,.0f}")

        st.divider()

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Expected Wealth", f"€{median[-1]:,.0f}")
        c2.metric("Portfolio Return", f"{port_r*100:.2f}%")
        c3.metric("Portfolio Risk", f"{port_v*100:.1f}%")

        if tipping:
            c4.metric("Compounding Year", tipping)

    # ---------------------------------------------------
    # TAB 2 – WEALTH PATH
    # ---------------------------------------------------

    with tab2:

        st.subheader("Wealth Projection")

        years_axis = list(range(len(median)))

        invested = [
            initial + monthly*12*i for i in years_axis
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=years_axis,
                y=median,
                name="Portfolio Value"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=years_axis,
                y=invested,
                name="Total Invested"
            )
        )

        fig.update_layout(
            xaxis_title="Years",
            yaxis_title="Portfolio Value (€)"
        )

        st.plotly_chart(fig, use_container_width=True)