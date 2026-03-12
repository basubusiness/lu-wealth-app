import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="LU Wealth Architect", layout="wide")

if "results" not in st.session_state:
    st.session_state.results=None

# -------------------------------
# DATA
# -------------------------------

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
"Cash":[2.0,1]
}
}

assets=[a for c in default_market_data.values() for a in c]

# -------------------------------
# CORRELATION
# -------------------------------

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

# -------------------------------
# MONTE CARLO
# -------------------------------

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

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("Inputs")

risk_profile=st.sidebar.select_slider(
"Risk profile",
["Conservative","Balanced","Aggressive"],
"Balanced"
)

caps={"Conservative":.25,"Balanced":.45,"Aggressive":.8}

crisis=st.sidebar.checkbox("Crisis stress")

names=[]
rets=[]
vols=[]

for cat,assets in default_market_data.items():

with st.sidebar.expander(cat,expanded=(cat=="Global Core")):

for a,p in assets.items():

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

# -------------------------------
# MAIN INPUTS
# -------------------------------

st.title("🇱🇺 Wealth Architect")

c1,c2=st.columns(2)

with c1:
initial=st.number_input("Initial capital €",10000,5000000,100000)
target=st.number_input("Target return %",1.,15.,6.5)/100
step_up=st.slider("Contribution growth %",0,10,3)/100

with c2:
monthly=st.number_input("Monthly investment €",0,20000,3000)
years=st.slider("Horizon",1,40,20)

# -------------------------------
# CALCULATE
# -------------------------------

if st.button("Create Investment Plan"):

rets=np.array(rets)
vols=np.array(vols)

avg=np.mean(rets)
shrink=.4
rets=shrink*rets+(1-shrink)*avg

corr=clean_corr(st.session_state.corr_matrix.loc[names,names])

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

# tipping point
contrib=[initial]

for t in range(1,years+1):
contrib.append(initial+monthly*12*t)

tipping=None

for i in range(len(median)):
if median[i]>contrib[i]:
tipping=i
break

st.session_state.results={
"w":w,
"names":names,
"median":median,
"paths":paths,
"ret":port_r,
"vol":port_v,
"tip":tipping
}

# -------------------------------
# LANDING VIEW
# -------------------------------

if st.session_state.results:

r=st.session_state.results

st.header("Your Investment Plan")

plan=pd.DataFrame({
"Asset":r["names"],
"Initial Investment €":r["w"]*initial,
"Monthly Investment €":r["w"]*monthly
})

st.dataframe(plan.style.format({
"Initial Investment €":"€{:,.0f}",
"Monthly Investment €":"€{:,.0f}"
}))

st.subheader("Wealth Path")

fig=go.Figure()

fig.add_trace(go.Scatter(y=r["median"],name="Expected wealth"))

st.plotly_chart(fig,use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------

st.subheader("Insights")

final=r["median"][-1]

st.info(f"""
Expected wealth after {years} years: **€{final:,.0f}**
""")

if r["tip"]:

st.success(f"""
Your portfolio growth is expected to exceed your cumulative contributions after **year {r['tip']}**.

This is the **compounding tipping point**.
""")

st.write(f"""
Expected return: **{r['ret']*100:.2f}%**

Volatility: **{r['vol']*100:.2f}%**

Typical yearly range:  
{(r['ret']-r['vol'])*100:.1f}% to {(r['ret']+r['vol'])*100:.1f}%
""")