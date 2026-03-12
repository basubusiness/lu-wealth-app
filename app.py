import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

if "results" not in st.session_state:
    st.session_state.results=None


# ---------------------------------------------------
# DATA
# ---------------------------------------------------

default_market_data = {
"Global Core":{
"World Equity":[7.5,16],
"All-World":[7.3,16.5],
"World Small Cap":[8.5,20]
},
"Regional Equity":{
"US (S&P 500)":[7.8,17],
"Europe 600":[7.0,16],
"Emerging Mkts":[8.0,22]
},
"Diversifiers":{
"Global REITs":[6.5,18],
"Gold":[4.0,15]
},
"Defensive":{
"Euro Gov Bonds":[3.0,6],
"Corp Bonds":[3.5,7],
"Cash/MM":[2.0,1]
}
}

assets=[a for c in default_market_data.values() for a in c]


# ---------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------

if "corr_matrix" not in st.session_state:

    corr=pd.DataFrame(.25,index=assets,columns=assets)

    for a in assets:
        corr.loc[a,a]=1

    equities=[
    "World Equity",
    "All-World",
    "World Small Cap",
    "US (S&P 500)",
    "Europe 600",
    "Emerging Mkts"
    ]

    for a in equities:
        for b in equities:
            if a!=b:
                corr.loc[a,b]=.85

    corr.loc["Gold",equities]=.05
    corr.loc[equities,"Gold"]=.05

    corr.loc["Global REITs",equities]=.7
    corr.loc[equities,"Global REITs"]=.7

    corr.loc["Euro Gov Bonds",equities]=.2
    corr.loc[equities,"Euro Gov Bonds"]=.2

    st.session_state.corr_matrix=corr


def clean_corr(c):

    c=(c+c.T)/2
    np.fill_diagonal(c.values,1)

    eigvals,eigvecs=np.linalg.eigh(c)
    eigvals[eigvals<0]=0

    c_psd=eigvecs@np.diag(eigvals)@eigvecs.T

    return pd.DataFrame(c_psd,index=c.index,columns=c.columns)


# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def monte_carlo(mu,sigma,years,start,monthly,step_up,sims=2000):

    paths=np.zeros((sims,years+1))
    paths[:,0]=start

    for s in range(sims):

        val=start

        for t in range(1,years+1):

            r=np.random.normal(mu,sigma)

            yc=monthly*12*((1+step_up)**(t-1))

            val=val*(1+r)+yc

            paths[s,t]=val

    return paths


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Market Hub")

risk_profile=st.sidebar.select_slider(
"Risk profile",
["Conservative","Balanced","Aggressive"],
"Balanced"
)

caps={"Conservative":.25,"Balanced":.45,"Aggressive":.8}

crisis_mode=st.sidebar.checkbox("Crisis stress")

names=[]
rets=[]
vols=[]

for cat,asset_set in default_market_data.items():

    with st.sidebar.expander(cat,expanded=(cat=="Global Core")):

        for a,p in asset_set.items():

            c1,c2,c3=st.columns([2,1,1])

            with c1:
                active=st.checkbox(a,value=True)

            with c2:
                r=st.number_input("R",value=float(p[0]),key=f"r{a}")

            with c3:
                v=st.number_input("V",value=float(p[1]),key=f"v{a}")

            if active:
                names.append(a)
                rets.append(r/100)
                vols.append(v/100)


# ---------------------------------------------------
# MAIN INPUTS
# ---------------------------------------------------

st.title("🇱🇺 Wealth Architect")

c1,c2=st.columns(2)

with c1:
    initial=st.number_input("Initial capital €",10000,5000000,100000)
    target=st.number_input("Target return %",1.,15.,6.5)/100
    step_up=st.slider("Contribution growth %",0,10,3)/100

with c2:
    monthly=st.number_input("Monthly investment €",0,20000,3000)
    years=st.slider("Horizon",1,40,20)


# ---------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------

if st.button("Create Investment Plan"):

    rets=np.array(rets)
    vols=np.array(vols)

    avg=np.mean(rets)
    shrink=.4
    rets=shrink*rets+(1-shrink)*avg

    corr=st.session_state.corr_matrix.loc[names,names]

    if crisis_mode:
        eq_mask=corr.index.str.contains("Equity|World|S&P|Europe|Emerging")
        corr.loc[eq_mask,eq_mask]=.95

    corr=clean_corr(corr)

    cov=np.diag(vols)@corr.values@np.diag(vols)

    def vol(w):
        return np.sqrt(w.T@cov@w)

    cons=[
    {"type":"eq","fun":lambda w:np.sum(w)-1},
    {"type":"ineq","fun":lambda w:w@rets-target}
    ]

    bounds=[(0,caps[risk_profile])]*len(rets)

    res=minimize(vol,np.ones(len(rets))/len(rets),bounds=bounds,constraints=cons)

    w=res.x

    port_r=w@rets
    port_v=np.sqrt(w.T@cov@w)

    paths=monte_carlo(port_r,port_v,years,initial,monthly,step_up)

    median=np.median(paths,axis=0)
    p10=np.percentile(paths,10,axis=0)
    p90=np.percentile(paths,90,axis=0)

    # MONTHLY COMPOUNDING LOGIC

    monthly_growth=median*(port_r/12)

    tipping=None

    for i,g in enumerate(monthly_growth):
        if g>=monthly:
            tipping=i
            break

    risk_contrib=(w*(cov@w))/(port_v**2)

    st.session_state.results={
    "weights":w,
    "names":names,
    "median":median,
    "p10":p10,
    "p90":p90,
    "ret":port_r,
    "vol":port_v,
    "risk_contrib":risk_contrib,
    "monthly_growth":monthly_growth,
    "tip":tipping
    }


# ---------------------------------------------------
# OUTPUT
# ---------------------------------------------------

if st.session_state.results:

    r=st.session_state.results

    tabs=st.tabs([
    "Investment Plan",
    "Wealth Path",
    "Risk",
    "Rebalance",
    "Advanced Risk",
    "Engine Room"
    ])


    # ---------------------------------------------------
    # TAB 1
    # ---------------------------------------------------

    with tabs[0]:

        st.header("Your Investment Plan")

        plan=pd.DataFrame({
        "Asset":r["names"],
        "Invest Today (€)":r["weights"]*initial,
        "Monthly Investment (€)":r["weights"]*monthly
        })

        st.dataframe(plan.style.format({
        "Invest Today (€)":"€{:,.0f}",
        "Monthly Investment (€)":"€{:,.0f}"
        }))

        final=r["median"][-1]

        st.subheader("Expected Outcome")

        st.metric("Expected Wealth",f"€{final:,.0f}")

        if r["tip"]:

            st.success(
            f"Monthly compounding overtakes saving around **year {r['tip']}**."
            )


        # CROSSOVER VISUALIZATION

        fig=go.Figure()

        fig.add_trace(go.Scatter(
        y=r["monthly_growth"],
        name="Monthly Investment Growth"
        ))

        fig.add_trace(go.Scatter(
        y=[monthly]*len(r["monthly_growth"]),
        name="Monthly Investment",
        line=dict(dash="dash")
        ))

        fig.update_layout(
        title="When Your Money Works Harder Than You",
        yaxis_title="€ per month"
        )

        st.plotly_chart(fig,use_container_width=True)


    # ---------------------------------------------------
    # TAB 2
    # ---------------------------------------------------

    with tabs[1]:

        fig=go.Figure()

        fig.add_trace(go.Scatter(y=r["median"],name="Median"))
        fig.add_trace(go.Scatter(y=r["p10"],name="Pessimistic",line=dict(dash="dash")))
        fig.add_trace(go.Scatter(y=r["p90"],name="Optimistic",line=dict(dash="dash")))

        st.plotly_chart(fig,use_container_width=True)


    # ---------------------------------------------------
    # TAB 3
    # ---------------------------------------------------

    with tabs[2]:

        mu=r["ret"]
        sigma=r["vol"]

        x=np.linspace(mu-4*sigma,mu+4*sigma,200)

        fig=go.Figure()

        fig.add_trace(go.Scatter(x=x,y=norm.pdf(x,mu,sigma),fill="tozeroy"))

        st.plotly_chart(fig,use_container_width=True)


    # ---------------------------------------------------
    # TAB 4
    # ---------------------------------------------------

    with tabs[3]:

        total=0
        actual_vals=[]

        for n,w in zip(r["names"],r["weights"]):

            val=st.number_input(
            f"Current {n} (€)",
            value=float(w*initial),
            key=f"reb_{n}"
            )

            actual_vals.append(val)
            total+=val

        if st.button("Generate Rebalance Plan"):

            reb_df=pd.DataFrame({
            "Asset":r["names"],
            "Actual":actual_vals,
            "Target %":r["weights"]
            })

            reb_df["Action"]=(reb_df["Target %"]*total)-reb_df["Actual"]

            st.table(reb_df.style.format({
            "Actual":"€{:,.0f}",
            "Target %":"{:.1%}",
            "Action":"€{:,.0f}"
            }))


    # ---------------------------------------------------
    # TAB 5
    # ---------------------------------------------------

    with tabs[4]:

        rc_df=pd.DataFrame({
        "Asset":r["names"],
        "Risk Contribution":r["risk_contrib"]
        })

        fig=go.Figure(go.Waterfall(
        x=rc_df["Asset"],
        y=rc_df["Risk Contribution"]*100
        ))

        st.plotly_chart(fig,use_container_width=True)


    # ---------------------------------------------------
    # TAB 6
    # ---------------------------------------------------

    with tabs[5]:

        st.write("Editable Correlation Matrix")

        st.data_editor(
        st.session_state.corr_matrix.loc[r["names"],r["names"]],
        use_container_width=True
        )