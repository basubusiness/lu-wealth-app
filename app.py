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

# Flatten structure for optimizer
ASSETS = {}
for cls,data in ASSET_CLASSES.items():
    for name,params in data["assets"].items():
        ASSETS[name] = {
            "return":params["return"],
            "vol":params["vol"],
            "cat":cls
        }

# ---------------------------------------------------
# CORRELATION RULES
# ---------------------------------------------------

CORR_RULES = {
    ("Equity","Equity"):0.85,
    ("Equity","Bond"):0.2,
    ("Equity","Real"):0.15,
    ("Equity","Commodity"):0.1,

    ("Bond","Bond"):0.6,
    ("Bond","Real"):0.1,
    ("Bond","Commodity"):0.05,

    ("Real","Real"):0.5,
    ("Real","Commodity"):0.2,

    ("Commodity","Commodity"):0.3
}

DEFAULT_CORR = 0.2

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def build_corr(selected):

    mat = pd.DataFrame(index=selected,columns=selected)

    for a in selected:
        for b in selected:

            ca = ASSETS[a]["cat"]
            cb = ASSETS[b]["cat"]

            if a == b:
                mat.loc[a,b] = 1.0
            else:
                key = (ca,cb)

                if key not in CORR_RULES:
                    key = (cb,ca)

                mat.loc[a,b] = CORR_RULES.get(key,DEFAULT_CORR)

    return mat.astype(float)

def objective(w,cov):

    vol = np.sqrt(w.T@cov@w)
    concentration = np.sum(w**2)

    return vol + 0.02*concentration

# ---------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------

def optimize_portfolio(names,target):

    original_returns = np.array([ASSETS[a]["return"] for a in names])
    vols = np.array([ASSETS[a]["vol"] for a in names])

    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink*original_returns + (1-shrink)*np.mean(original_returns)

    if "corr_override" in st.session_state:
        corr = np.array(st.session_state["corr_override"])
    else:
        corr = build_corr(names).values

    cov = np.diag(vols) @ corr @ np.diag(vols)

    max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]

    # dynamic class indices
    class_indices = {}

    for i,a in enumerate(names):

        cls = ASSETS[a]["cat"]

        if cls not in class_indices:
            class_indices[cls] = []

        class_indices[cls].append(i)

    constraints = [{"type":"eq","fun":lambda w:np.sum(w)-1}]

    ret_constraint = {"type":"ineq","fun":lambda w:w@rets-target}

    constraints.append(ret_constraint)

    for cls,idx in class_indices.items():

        min_w = ASSET_CLASSES[cls]["min"]
        max_w = ASSET_CLASSES[cls]["max"]

        constraints.append(
            {"type":"ineq",
             "fun":lambda w,idx=idx,min_w=min_w:np.sum(w[idx])-min_w}
        )

        constraints.append(
            {"type":"ineq",
             "fun":lambda w,idx=idx,max_w=max_w:max_w-np.sum(w[idx])}
        )

    bounds = [(0,max_w)]*len(names)

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
        w = w/np.sum(w)

    port_r = w@rets
    port_v = np.sqrt(w.T@cov@w)

    return w,port_r,port_v
