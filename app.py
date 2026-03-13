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

ASSET_CLASSES = {

    "Equity": {
        "min":0.4,
        "max":0.9,
        "assets":{
            "World Equity":{"return":0.075,"vol":0.16},
            "US Equity":{"return":0.078,"vol":0.17},
            "Emerging Markets":{"return":0.08,"vol":0.22}
        }
    },

    "Bond":{
        "min":0.1,
        "max":0.5,
        "assets":{
            "Euro Gov Bonds":{"return":0.03,"vol":0.06},
            "Corp Bonds":{"return":0.035,"vol":0.07}
        }
    },

    "Real":{
        "min":0.05,
        "max":0.25,
        "assets":{
            "Global REIT":{"return":0.065,"vol":0.18}
        }
    },

    "Commodity":{
        "min":0.0,
        "max":0.15,
        "assets":{
            "Gold":{"return":0.065,"vol":0.17}
        }
    }
}

ASSETS = {}

for cls,data in ASSET_CLASSES.items():
    for name,params in data["assets"].items():

        ASSETS[name] = {
            "return":params["return"],
            "vol":params["vol"],
            "cat":cls
        }

# ---------------------------------------------------
# CORRELATION RULES (FIXED)
# ---------------------------------------------------

CORR_RULES = {

("Equity","Equity"):0.85,
("Equity","Bond"):0.2,
("Equity","Real"):0.15,
("Equity","Commodity"):0.10,

("Bond","Bond"):0.6,
("Bond","Real"):0.1,
("Bond","Commodity"):0.05,

("Real","Real"):0.5,
("Real","Commodity"):0.2,

("Commodity","Commodity"):0.3
}

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def build_corr(selected):

    mat = pd.DataFrame(index=selected, columns=selected)

    for a in selected:
        for b in selected:

            ca = ASSETS[a]["cat"]
            cb = ASSETS[b]["cat"]

            if a == b:
                mat.loc[a,b] = 1.0
            else:
                key = (ca, cb) if (ca, cb) in CORR_RULES else (cb, ca)

                # SAFE fallback correlation
                mat.loc[a,b] = CORR_RULES.get(key, 0.2)

    return mat.astype(float)


def objective(w, cov):

    vol = np.sqrt(w.T @ cov @ w)
    concentration = np.sum(w**2)

    return vol + 0.02 * concentration


# ---------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------

def optimize_portfolio(names, target):

    original_returns = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink * original_returns + (1 - shrink) * np.mean(original_returns)

    if "corr_override" in st.session_state:
        corr = np.array(st.session_state["corr_override"])
    else:
        corr = build_corr(names).values

    cov = np.diag(vols) @ corr @ np.diag(vols)
    max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]

    class_indices = {}

    for i, a in enumerate(names):

        cls = ASSETS[a]["cat"]

        if cls not in class_indices:
            class_indices[cls] = []

        class_indices[cls].append(i)

    constraints = [{"type":"eq","fun":lambda w:np.sum(w)-1}]

    ret_constraint = {"type":"ineq","fun":lambda w:w@rets-target}
    constraints.append(ret_constraint)

    for cls, idx in class_indices.items():

        min_w = ASSET_CLASSES[cls]["min"]
        max_w = ASSET_CLASSES[cls]["max"]

        constraints.append(
            {"type":"ineq",
             "fun":lambda w,idx=idx,min_w=min_w: np.sum(w[idx]) - min_w}
        )

        constraints.append(
            {"type":"ineq",
             "fun":lambda w,idx=idx,max_w=max_w: max_w - np.sum(w[idx])}
        )

    bounds = [(0, max_w)] * len(names)

    res = minimize(objective,
                   np.ones(len(names))/len(names),
                   args=(cov,),
                   bounds=bounds,
                   constraints=constraints,
                   method="SLSQP")

    if not res.success:

        st.warning("⚠️ Target return could not be achieved. Showing lowest-risk portfolio.")

        fallback_constraints = [c for c in constraints if c != ret_constraint]

        res = minimize(objective,
                       np.ones(len(names))/len(names),
                       args=(cov,),
                       bounds=bounds,
                       constraints=fallback_constraints,
                       method="SLSQP")

        if not res.success:
            st.error("Optimization failed. Try selecting more assets.")
            st.stop()

    w = res.x
    w[w < 0.002] = 0

    if np.sum(w) > 0:
        w = w / np.sum(w)

    port_r = w @ rets
    port_v = np.sqrt(w.T @ cov @ w)

    return w, port_r, port_v


# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def simulate(mu, sigma, years, start, monthly, growth):

    sims = ASSUMPTIONS["simulation"]["paths"]
    df = 5
    scale_factor = np.sqrt((df - 2) / df)

    mu_log = mu - (0.5 * sigma**2)

    paths = np.zeros((sims, years + 1))
    paths[:,0] = start

    shocks = mu_log + np.random.standard_t(df, (sims, years)) * sigma * scale_factor

    for t in range(1, years + 1):

        contrib = monthly * 12 * ((1 + growth)**(t - 1))
        paths[:,t] = paths[:,t-1] * np.exp(shocks[:,t-1]) + contrib

    return paths


def scenario_paths(paths, confidence):

    lower = (1-confidence)/2
    worst = np.percentile(paths, lower*100, axis=0)
    median = np.percentile(paths, 50, axis=0)
    best = np.percentile(paths, (1-lower)*100, axis=0)

    return worst, median, best


# ---------------------------------------------------
# INPUTS & UI
# ---------------------------------------------------

st.title("LU Wealth Architect")

c1, c2, c3, c4, c5 = st.columns(5)

with c1: initial = st.number_input("Initial Capital",10000,5000000,100000)
with c2: monthly = st.number_input("Monthly Saving",0,20000,3000)
with c3: years = st.slider("Horizon (years)",1,40,20)
with c4: target_pct = st.number_input("Target Return %",3.0,10.0,6.5)
with c5: growth_pct = st.slider("Saving Growth %",0,10,3)

confidence = st.slider("Projection confidence level",80,99,90)/100
target, growth = target_pct/100, growth_pct/100


selected_assets = []

for cls,data in ASSET_CLASSES.items():

    with st.expander(cls,expanded=True):

        for asset in data["assets"]:

            default = True if cls in ["Equity","Bond"] else False

            if st.checkbox(asset,value=default):

                selected_assets.append(asset)

assets = selected_assets

# Guard against empty selection
if len(assets) < 2:
    st.warning("Please select at least two assets.")
    st.stop()


# Reset correlation override safely
if "corr_override" in st.session_state:
    if st.session_state["corr_override"].shape[0] != len(assets):
        del st.session_state["corr_override"]


# ---------------------------------------------------
# RUN MODEL
# ---------------------------------------------------

if st.button("Build Plan"):

    w, port_r, port_v = optimize_portfolio(assets, target)

    paths = simulate(port_r, port_v, years, initial, monthly, growth)

    worst, median, best = scenario_paths(paths, confidence)

    years_axis = list(range(len(median)))

    total_invested = initial + sum([monthly * 12 * ((1 + growth)**i) for i in range(years)])
    growth_value = median[-1] - total_invested
    monthly_income = median[-1] * port_r / 12

    tipping = next((i for i, g in enumerate(median * (port_r / 12)) if g >= monthly), None)

    plan = pd.DataFrame({
        "Asset": assets,
        "Weight": w,
        "Invest Now": w * initial,
        "Monthly": w * monthly
    })

    plan = plan[plan["Weight"] > 0.01]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Projection", "Risk", "Rebalance", "Engine"])

# ---------------------------------------------------
# PLAN TAB
# ---------------------------------------------------

with tab1:

    st.dataframe(
        plan.style.format({
            "Weight":"{:.1%}",
            "Invest Now":"€{:,.0f}",
            "Monthly":"€{:,.0f}"
        }),
        use_container_width=True
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="card">
        <div class="card-value">€{median[-1]:,.0f}</div>
        <div class="card-label">Projected portfolio value after {years} years</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
        <div class="card-value">{port_r*100:.1f}%</div>
        <div class="card-label">Typical yearly growth rate</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
        <div class="card-value">€{growth_value:,.0f}</div>
        <div class="card-label">
        Investment growth on top of €{total_invested:,.0f} contributed
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        if tipping:
            income_at_tipping = median[tipping] * port_r / 12

            st.markdown(f"""
            <div class="card">
            <div class="card-value">Year {tipping}</div>
            <div class="card-label">
            Growth beats saving (~€{income_at_tipping:,.0f}/month)
            </div>
            </div>
            """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="card">
        <div class="card-value">€{monthly_income:,.0f}</div>
        <div class="card-label">
        Estimated monthly income after {years} years
        </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------
# PROJECTION TAB
# ---------------------------------------------------

with tab2:

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=years_axis,y=best,name="Best case",line=dict(color="green")))
    fig.add_trace(go.Scatter(x=years_axis,y=median,name="Expected",line=dict(color="blue",width=3)))
    fig.add_trace(go.Scatter(x=years_axis,y=worst,name="Worst case",line=dict(color="red")))

    smooth_path = [initial]

    for t in range(1, years + 1):
        contrib = monthly * 12 * ((1 + growth)**(t-1))
        val = smooth_path[-1] * (1 + port_r) + contrib
        smooth_path.append(val)

    fig.add_trace(go.Scatter(
        x=years_axis,
        y=smooth_path,
        name="Theoretical (Smooth)",
        line=dict(color="orange",dash="dot",width=2)
    ))

    st.plotly_chart(fig,use_container_width=True)


# ---------------------------------------------------
# RISK TAB
# ---------------------------------------------------

with tab3:

    fig = go.Figure()

    fig.add_histogram(
        x=paths[:,-1],
        nbinsx=40
    )

    fig.update_layout(
        title="Distribution of possible final wealth outcomes"
    )

    st.plotly_chart(fig)


# ---------------------------------------------------
# REBALANCE TAB
# ---------------------------------------------------

with tab4:

    st.subheader("Rebalancing calculator")

    current = {}

    for asset in plan["Asset"]:
        current[asset] = st.number_input(
            f"Current value {asset}",
            value=float(plan.loc[plan["Asset"]==asset,"Invest Now"])
        )

    total_current = sum(current.values())

    target_values = plan["Weight"] * total_current

    rebalance = target_values - list(current.values())

    rebalance_df = pd.DataFrame({
        "Asset":plan["Asset"],
        "Target €":target_values,
        "Current €":list(current.values()),
        "Buy / Sell €":rebalance
    })

    st.dataframe(rebalance_df)


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

    corr = st.session_state["corr_override"]

    edited_corr = st.data_editor(
        corr,
        key="corr_editor"
    )

    st.session_state["corr_override"] = edited_corr
