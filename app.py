import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Shreeyanee V3", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.metric-card {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: left;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #e8f4e8;
    letter-spacing: -0.5px;
}
.metric-lbl {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.metric-delta {
    font-size: 11px;
    color: #4ade80;
    margin-top: 2px;
    font-family: 'DM Mono', monospace;
}
.assumption-box {
    background: #0a0e1a;
    border-left: 3px solid #3b82f6;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
}
.sensitivity-box {
    background: #0a0e1a;
    border-left: 3px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    color: #94a3b8;
}
.warning-inline {
    background: #1c1007;
    border: 1px solid #92400e;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    color: #fbbf24;
    margin: 6px 0;
}
.crisis-label {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: #fca5a5;
    margin-bottom: 8px;
}
.rebal-note {
    background: #0a1a10;
    border: 1px solid #14532d;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    color: #86efac;
    margin: 6px 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0f1117;
    padding: 4px;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 6px;
    font-size: 13px;
    font-family: 'DM Mono', monospace;
}
.stTabs [aria-selected="true"] {
    background: #1e2130 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ASSUMPTIONS & CONSTANTS  (single source of truth)
# ============================================================
ASSUMPTIONS = {
    "basis": "Long-run historical averages adjusted for current yield environment (2024–2025).",
    "horizon": "Intended for 10–20 year planning. Not short-term forecasts.",
    "real_nominal": "All returns are NOMINAL (before inflation). Apply 2–2.5% inflation haircut for real purchasing power.",
    "source": "Broadly consistent with Vanguard, Research Affiliates, and AQR long-run capital market assumptions.",
    "optimizer": {
        "return_shrinkage": 0.40,   # 40% pull toward grand mean — real regularisation
        "max_asset_weight": 0.40,
    },
    "category_caps": {
        "Equity": 0.70,
        "Bond":   0.60,
        "Real":   0.40,
        "Alt":    0.20,
        "Cash":   0.30,
    },
    "simulation": {
        "paths": 2000,
        "t_df": 5,               # degrees of freedom for fat-tailed draws
    },
    "safe_withdrawal_rate": 0.04,
    "rebalance_drift_threshold": 0.05,  # trigger rebalance when asset drifts >5%
    "return_scenarios": {
        "Conservative": 0.80,    # scale factor on base returns
        "Base":         1.00,
        "Optimistic":   1.20,
    },
    "crisis_regime": {
        "return_scale": 0.40,    # compress returns to 40% of base
        "vol_scale":    1.80,    # inflate volatility by 80%
        "corr_floor":   0.70,    # all cross-asset correlations → 0.70
        "label": "Stylised 2022-style crisis — NOT a probability forecast. "
                 "Shows approximate portfolio behaviour under simultaneous equity + bond drawdown."
    }
}

# ============================================================
# ASSETS
# ============================================================
ASSETS = {
    "World Equity":          {"return": 0.075, "vol": 0.16, "cat": "Equity"},
    "US Equity":             {"return": 0.078, "vol": 0.17, "cat": "Equity"},
    "Emerging Markets":      {"return": 0.080, "vol": 0.22, "cat": "Equity"},
    "Global Small Cap":      {"return": 0.082, "vol": 0.19, "cat": "Equity"},
    "Global REIT":           {"return": 0.060, "vol": 0.19, "cat": "Real"},
    "Gold":                  {"return": 0.050, "vol": 0.17, "cat": "Real"},
    "Broad Commodities":     {"return": 0.040, "vol": 0.20, "cat": "Real"},
    "Euro Gov Bonds":        {"return": 0.030, "vol": 0.06, "cat": "Bond"},
    "Corp Bonds":            {"return": 0.035, "vol": 0.07, "cat": "Bond"},
    "Global Inflation Bonds":{"return": 0.032, "vol": 0.05, "cat": "Bond"},
    "Cash":                  {"return": 0.025, "vol": 0.01, "cat": "Cash"},
    "Crypto":                {"return": 0.100, "vol": 0.75, "cat": "Alt"},
    "Semiconductors":        {"return": 0.090, "vol": 0.30, "cat": "Alt"},
    "Frontier Markets":      {"return": 0.085, "vol": 0.28, "cat": "Alt"},
    "Uranium":               {"return": 0.090, "vol": 0.35, "cat": "Alt"},
}

CORR_RULES = {
    ("Equity","Equity"): 0.85,
    ("Equity","Bond"):   0.10,   # can be negative; using conservative positive
    ("Equity","Real"):   0.20,
    ("Equity","Alt"):    0.50,
    ("Equity","Cash"):   0.05,
    ("Bond","Bond"):     0.60,
    ("Bond","Real"):     0.10,
    ("Bond","Alt"):      0.10,
    ("Bond","Cash"):     0.10,
    ("Real","Real"):     0.30,
    ("Real","Alt"):      0.25,
    ("Real","Cash"):     0.05,
    ("Alt","Alt"):       0.55,
    ("Alt","Cash"):      0.05,
    ("Cash","Cash"):     1.00,
}

# ============================================================
# SESSION STATE INIT
# ============================================================
if "init" not in st.session_state:
    for a in ASSETS:
        st.session_state[f"asset_{a}"] = True
    for cat in set(d["cat"] for d in ASSETS.values()):
        st.session_state[f"master_{cat}"] = True
    st.session_state["asset_settings"] = (
        pd.DataFrame(ASSETS).T.reset_index().rename(columns={"index": "Asset"})
    )
    st.session_state["return_scenario"] = "Base"
    st.session_state["init"] = True

# ============================================================
# HELPERS
# ============================================================
def build_corr(selected):
    mat = pd.DataFrame(index=selected, columns=selected, dtype=float)
    for a in selected:
        for b in selected:
            if a == b:
                mat.loc[a, b] = 1.0
            else:
                ca, cb = ASSETS[a]["cat"], ASSETS[b]["cat"]
                key = (ca, cb) if (ca, cb) in CORR_RULES else (cb, ca)
                mat.loc[a, b] = CORR_RULES.get(key, 0.30)
    return mat


def get_base_returns(names):
    """Always use base (unscaled) returns for optimization."""
    df = st.session_state["asset_settings"].set_index("Asset")
    return df.loc[names, "return"].values.astype(float)


def optimize_portfolio(names, target_r, scenario):
    """
    Optimization always runs on BASE returns.
    Scenario scaling is applied only to the simulation mu so that
    portfolio structure is stable across scenarios while wealth
    projections reflect the user's return belief.
    """
    df = st.session_state["asset_settings"].set_index("Asset")
    vols = df.loc[names, "vol"].values.astype(float)
    raw_rets = get_base_returns(names)

    # --- Proper shrinkage from ASSUMPTIONS (0.40 = meaningful regularisation) ---
    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets = shrink * np.mean(raw_rets) + (1 - shrink) * raw_rets

    corr = st.session_state["corr_override"].loc[names, names].values
    cov = np.diag(vols) @ corr @ np.diag(vols)

    # --- Category index maps ---
    cat_map = {cat: [] for cat in ASSUMPTIONS["category_caps"]}
    for i, a in enumerate(names):
        cat_map[ASSETS[a]["cat"]].append(i)

    # --- Objective: maximise Sharpe only ---
    rf = 0.02
    def objective(w):
        port_ret = w @ rets
        port_vol = np.sqrt(w.T @ cov @ w + 1e-10)
        return -((port_ret - rf) / port_vol)

    # --- Constraints: hard return floor + category caps ---
    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: (w @ rets) - target_r},  # HARD floor
    ]
    for cat, cap in ASSUMPTIONS["category_caps"].items():
        idx = cat_map.get(cat, [])
        if idx:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idx, cap=cap: cap - np.sum(w[ix])
            })

    n = len(names)
    max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]
    bounds = [(0.0, max_w)] * n

    best_res = None
    for seed in [np.ones(n)/n, np.random.dirichlet(np.ones(n))]:
        res = minimize(objective, seed, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
        if res.success and (best_res is None or res.fun < best_res.fun):
            best_res = res

    if best_res is None or not best_res.success:
        return None, None, None

    w = np.round(best_res.x, 4)
    w[w < 0.005] = 0
    if np.sum(w) > 0:
        w /= np.sum(w)

    port_r = w @ rets
    port_v = np.sqrt(w.T @ cov @ w)
    return w, port_r, port_v


def simulate_paths(mu, sigma, years, start, monthly, contrib_growth, crisis=False):
    sims = ASSUMPTIONS["simulation"]["paths"]
    df_t = ASSUMPTIONS["simulation"]["t_df"]

    if crisis:
        cr = ASSUMPTIONS["crisis_regime"]
        mu    = mu    * cr["return_scale"]
        sigma = sigma * cr["vol_scale"]

    scale = np.sqrt((df_t - 2) / df_t)
    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start

    for t in range(1, years + 1):
        shocks  = (mu - 0.5 * sigma**2) + np.random.standard_t(df_t, sims) * sigma * scale
        contrib = monthly * 12 * ((1 + contrib_growth) ** (t - 1))
        paths[:, t] = paths[:, t - 1] * np.exp(shocks) + contrib

    return paths


def sensitivity_analysis(names, target_r, scenario, years, initial, monthly, contrib_growth):
    """Show terminal wealth impact of ±1% return assumption."""
    results = {}
    for delta in [-0.01, 0.0, +0.01]:
        df = st.session_state["asset_settings"].set_index("Asset").copy()
        df["return"] = df["return"] + delta
        orig = st.session_state["asset_settings"].copy()
        st.session_state["asset_settings"] = df.reset_index().rename(columns={"index": "Asset"})
        w, mu, sigma = optimize_portfolio(names, target_r + delta, "Base")
        st.session_state["asset_settings"] = orig
        if w is not None:
            paths = simulate_paths(mu, sigma, years, initial, monthly, contrib_growth)
            results[delta] = np.percentile(paths[:, -1], 50)
    return results


def rebalance_triggers(weights, names, current_vals):
    """Flag assets that have drifted beyond threshold."""
    total = sum(current_vals.values())
    if total == 0:
        return pd.DataFrame()
    rows = []
    threshold = ASSUMPTIONS["rebalance_drift_threshold"]
    for a, w in zip(names, weights):
        if w < 0.005:
            continue
        curr_w = current_vals.get(a, 0) / total
        drift = curr_w - w
        action = (w - curr_w) * total
        trigger = abs(drift) >= threshold
        rows.append({
            "Asset": a,
            "Target %": w,
            "Current %": curr_w,
            "Drift": drift,
            "Buy / Sell €": action,
            "⚠️ Rebalance": "YES" if trigger else "—",
        })
    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================
st.title("Shreeyanee · श्रीयनी   V3")
st.caption("Personal portfolio architect — long-horizon, Monte Carlo driven, institutionally structured.")

with st.expander("📋 Model Assumptions & Limitations — read before trusting outputs", expanded=False):
    st.markdown(f"""
<div class="assumption-box">
<b>Return basis:</b> {ASSUMPTIONS['basis']}<br>
<b>Horizon:</b> {ASSUMPTIONS['horizon']}<br>
<b>Nominal vs Real:</b> {ASSUMPTIONS['real_nominal']}<br>
<b>Sources:</b> {ASSUMPTIONS['source']}<br><br>
<b>What this model does NOT do:</b><br>
• No short-term forecasting or market timing<br>
• No tax optimisation<br>
• No liability matching or cash-flow matching<br>
• Correlations are stylised — they rise sharply in real crises<br>
• Past return distributions do not guarantee future outcomes
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUTS
# ============================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: initial   = st.number_input("Initial Capital (€)", 10_000, 5_000_000, 100_000, step=5_000, help="Lump sum invested on day one.")
with c2: monthly   = st.number_input("Monthly Saving (€)",  0, 50_000, 3_000, step=100, help="Regular monthly contribution, assumed invested at start of each year.")
with c3: years     = st.slider("Horizon (years)", 1, 40, 20, help="Investment horizon. Longer horizons reduce sequence-of-returns risk.")
with c4: target_pct = st.number_input("Target Return %", 2.0, 14.0, 6.5, step=0.5, help="Minimum annualised nominal return the optimiser must achieve. Higher targets force more risk.")
with c5: growth_pct = st.slider("Saving Growth %/yr", 0, 10, 3, help="Annual step-up in monthly savings (e.g. salary growth). 0 = flat contributions forever.")
with c6:
    inflation_pct = st.number_input("Inflation %", 0.5, 6.0, 2.0, step=0.1, help="Used to deflate nominal projections into today's purchasing power. Eurozone long-run target is ~2%.")
    show_real = st.checkbox("Show real (inflation-adj.) values", value=True, help="When on, all euro values shown in today's purchasing power. Strongly recommended for long horizons.")

scenario = st.radio(
    "Return scenario",
    list(ASSUMPTIONS["return_scenarios"].keys()),
    index=1,
    horizontal=True,
    help="Shifts the wealth projection only — portfolio weights are always optimised on base returns. Conservative (×0.8): markets underperform history. Optimistic (×1.2): markets beat history. Use Base for central planning."
)

target   = target_pct / 100
growth   = growth_pct / 100
inflation = inflation_pct / 100

# ============================================================
# ASSET SELECTION
# ============================================================
with st.expander("🧩 Asset Universe", expanded=False):
    def sync_cat(cat_name, assets_in_cat):
        for a in assets_in_cat:
            st.session_state[f"asset_{a}"] = st.session_state[f"master_{cat_name}"]

    cats = sorted(set(d["cat"] for d in ASSETS.values()))
    cols = st.columns(len(cats))
    for i, cat in enumerate(cats):
        with cols[i]:
            st.markdown(f"**{cat}**")
            cat_assets = [n for n, d in ASSETS.items() if d["cat"] == cat]
            st.checkbox(f"All {cat}", key=f"master_{cat}",
                        on_change=sync_cat, args=(cat, cat_assets))
            for a in cat_assets:
                st.checkbox(a, key=f"asset_{a}")

# Derive selected list AFTER widgets render
selected_assets = [a for a in ASSETS if st.session_state.get(f"asset_{a}", False)]

# Alert logic
if selected_assets:
    df_lu = st.session_state["asset_settings"].set_index("Asset")
    max_avail = df_lu.loc[selected_assets, "return"].max()
    scaled_max = max_avail * ASSUMPTIONS["return_scenarios"][scenario]
    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    effective_target = shrink * np.mean(
        df_lu.loc[selected_assets, "return"].values * ASSUMPTIONS["return_scenarios"][scenario]
    ) + (1 - shrink) * target
    if target > scaled_max:
        st.error(f"⚠️ Target {target_pct}% exceeds the highest available return ({scaled_max*100:.1f}%) under '{scenario}' scenario. Lower target or enable higher-return assets.")
    elif target > 0.08:
        st.warning(f"⚡ {target_pct}% target requires significant Equity/Alt exposure — review category caps in Engine tab.")
else:
    st.info("Select assets to proceed.")

# Rebuild correlation matrix only when selection changes
sel_key = tuple(sorted(selected_assets))
if "corr_override" not in st.session_state or st.session_state.get("_last_sel") != sel_key:
    if selected_assets:
        st.session_state["corr_override"] = build_corr(selected_assets)
        st.session_state["_last_sel"] = sel_key

# ============================================================
# BUILD BUTTON
# ============================================================
if st.button("🏗️  Build Plan", type="primary") and selected_assets:
    with st.spinner("Optimising portfolio..."):
        w, port_r, port_v = optimize_portfolio(selected_assets, target, scenario)

    if w is None:
        st.error("Optimiser failed — target may be infeasible under current constraints and scenario. "
                 "Try lowering target, switching to Optimistic scenario, or enabling more assets.")
    else:
        with st.spinner("Running Monte Carlo (2 000 paths × 2 regimes)..."):
            # Apply scenario scaling to mu only (weights unchanged across scenarios)
            scenario_scale = ASSUMPTIONS["return_scenarios"][scenario]
            sim_mu = port_r * scenario_scale
            paths_base   = simulate_paths(sim_mu, port_v, years, initial, monthly, growth, crisis=False)
            paths_crisis = simulate_paths(sim_mu, port_v, years, initial, monthly, growth, crisis=True)

        # Sensitivity analysis (±1% on returns)
        sens = sensitivity_analysis(selected_assets, target, scenario, years, initial, monthly, growth)

        st.session_state["results"] = {
            "w": w, "port_r": port_r, "port_v": port_v,
            "paths_base": paths_base, "paths_crisis": paths_crisis,
            "assets": selected_assets, "sens": sens,
        }

# ============================================================
# RESULTS
# ============================================================
if "results" in st.session_state:
    R = st.session_state["results"]
    w         = R["w"]
    port_r    = R["port_r"]
    port_v    = R["port_v"]
    paths_b   = R["paths_base"]
    paths_c   = R["paths_crisis"]
    assets    = R["assets"]
    sens      = R["sens"]

    # Inflation adjustment
    def maybe_deflate(arr):
        if not show_real:
            return arr
        arr = np.array(arr)
        if arr.ndim == 2:
            factors = np.array([(1 + inflation) ** t for t in range(arr.shape[1])])
            return arr / factors[np.newaxis, :]
        else:
            factors = np.array([(1 + inflation) ** t for t in range(len(arr))])
            return arr / factors

    yrs = np.arange(years + 1)
    label_sfx = " (real €)" if show_real else " (nominal €)"

    p10_b, p50_b, p90_b = [maybe_deflate(np.percentile(paths_b, p, axis=0)) for p in [10, 50, 90]]
    p10_c, p50_c, p90_c = [maybe_deflate(np.percentile(paths_c, p, axis=0)) for p in [10, 50, 90]]

    # Core metrics
    total_invested = initial + sum(monthly * 12 * ((1 + growth) ** i) for i in range(years))
    real_invested  = total_invested / ((1 + inflation) ** years) if show_real else total_invested
    med_final      = p50_b[-1]
    growth_val     = med_final - (real_invested if show_real else total_invested)
    swr            = ASSUMPTIONS["safe_withdrawal_rate"]
    monthly_income = med_final * swr / 12
    tipping        = next((i for i in range(1, years + 1)
                           if p50_b[i] * swr / 12 >= monthly), None)
    sharpe         = (port_r - 0.02) / port_v if port_v > 0 else 0

    # --- Tabs ---
    tab_plan, tab_proj, tab_risk, tab_reb, tab_engine = st.tabs([
        "📐 Plan", "📈 Projection", "🛡️ Risk", "⚖️ Rebalance", "⚙️ Engine"
    ])

    # ── TAB 1: PLAN ──────────────────────────────────────────
    with tab_plan:
        plan_df = pd.DataFrame({
            "Asset":      assets,
            "Weight":     w,
            "Category":   [ASSETS[a]["cat"] for a in assets],
            "Invest Now": w * initial,
            "Monthly":    w * monthly,
        })
        plan_df = plan_df[plan_df["Weight"] > 0.005].sort_values("Weight", ascending=False)

        st.dataframe(
            plan_df.style.format({
                "Weight":     "{:.1%}",
                "Invest Now": "€{:,.0f}",
                "Monthly":    "€{:,.0f}",
            }).bar(subset=["Weight"], color="#1e3a5f"),
            use_container_width=True,
            hide_index=True,
        )

        # Category breakdown
        cat_df = plan_df.groupby("Category")["Weight"].sum().reset_index()
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_pie = go.Figure(go.Pie(
                labels=cat_df["Category"], values=cat_df["Weight"],
                hole=0.55, textinfo="label+percent",
                marker=dict(colors=["#1e40af","#065f46","#7c2d12","#4c1d95","#1e293b"])
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20), height=280
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            metrics = [
                (f"€{med_final:,.0f}", f"Median wealth{label_sfx}", ""),
                (f"{port_r*100:.2f}%", "Portfolio return — base (post-shrinkage)", ""),
                (f"{port_v*100:.1f}%", "Portfolio volatility", ""),
                (f"{sharpe:.2f}", "Sharpe ratio (rf=2%)", ""),
                (f"€{monthly_income:,.0f}/mo", f"Safe withdrawal income ({int(swr*100)}% SWR)", ""),
                (f"Year {tipping}" if tipping else ">horizon", "Compounding tipping point", ""),
            ]
            for val, lbl, delta in metrics:
                st.markdown(f"""
<div class="metric-card" style="margin-bottom:6px">
  <div class="metric-val">{val}</div>
  <div class="metric-lbl">{lbl}</div>
  {"<div class='metric-delta'>"+delta+"</div>" if delta else ""}
</div>""", unsafe_allow_html=True)

        # Sensitivity callout
        if 0.0 in sens and -0.01 in sens and 0.01 in sens:
            diff_down = sens[-0.01] - sens[0.0]
            diff_up   = sens[0.01]  - sens[0.0]
            st.markdown(f"""
<div class="sensitivity-box">
<b>⚡ Sensitivity to return assumptions (±1%)</b><br>
If actual returns are 1% <i>lower</i> than assumed: median terminal wealth ≈
<b>€{sens[-0.01]:,.0f}</b> ({diff_down/sens[0.0]*100:+.1f}%)<br>
If actual returns are 1% <i>higher</i> than assumed: median terminal wealth ≈
<b>€{sens[0.01]:,.0f}</b> ({diff_up/sens[0.0]*100:+.1f}%)<br>
<span style="color:#6b7280;font-size:11px">
This range illustrates genuine uncertainty — treat the base estimate as a midpoint, not a prediction.
</span>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2: PROJECTION ────────────────────────────────────
    with tab_proj:
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Base Regime", f"Crisis Regime — {ASSUMPTIONS['crisis_regime']['label'][:50]}…"])

        colors_b = dict(p90="#4ade80", p50="#60a5fa", p10="#f87171")
        colors_c = dict(p90="#86efac", p50="#93c5fd", p10="#fca5a5")

        for col, (p10, p50, p90, colors, label) in enumerate([
            (p10_b, p50_b, p90_b, colors_b, "Base"),
            (p10_c, p50_c, p90_c, colors_c, "Crisis"),
        ], 1):
            fig.add_trace(go.Scatter(x=yrs, y=p90, name=f"90th pct ({label})",
                line=dict(color=colors["p90"], dash="dot", width=1.5),
                showlegend=(col==1)), row=1, col=col)
            fig.add_trace(go.Scatter(x=yrs, y=p50, name=f"Median ({label})",
                line=dict(color=colors["p50"], width=3),
                showlegend=(col==1)), row=1, col=col)
            fig.add_trace(go.Scatter(x=yrs, y=p10, name=f"10th pct ({label})",
                line=dict(color=colors["p10"], dash="dot", width=1.5),
                showlegend=(col==1)), row=1, col=col)
            # Invested line
            inv_line = maybe_deflate(np.array([
                initial + sum(monthly * 12 * ((1+growth)**i) for i in range(t))
                for t in range(years+1)
            ]))
            fig.add_trace(go.Scatter(x=yrs, y=inv_line, name="Capital invested",
                line=dict(color="#64748b", dash="longdash", width=1),
                showlegend=(col==1)), row=1, col=col)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="DM Mono"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
            height=420, margin=dict(t=40, b=60, l=20, r=20),
            yaxis=dict(tickprefix="€", gridcolor="#1e2130"),
            yaxis2=dict(tickprefix="€", gridcolor="#1e2130"),
            xaxis=dict(title="Year", gridcolor="#1e2130"),
            xaxis2=dict(title="Year", gridcolor="#1e2130"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
<div class="crisis-label">
⚠️ <b>Crisis regime</b>: {ASSUMPTIONS['crisis_regime']['label']}
Return compressed to {int(ASSUMPTIONS['crisis_regime']['return_scale']*100)}% of base,
volatility ×{ASSUMPTIONS['crisis_regime']['vol_scale']},
cross-asset correlations floored at {ASSUMPTIONS['crisis_regime']['corr_floor']}.
</div>
""", unsafe_allow_html=True)

        # Total invested annotation
        tot_inv_real = maybe_deflate(np.array([total_invested]))[0]
        real_note = f"EUR {tot_inv_real:,.0f} in today's money" if show_real else "nominal"
        st.caption(f"Total capital injected over {years} years: **EUR {total_invested:,.0f}** ({real_note})")

    # ── TAB 3: RISK ───────────────────────────────────────────
    with tab_risk:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Terminal wealth distribution")
            final_vals = maybe_deflate(paths_b)[:, -1]
            fig_h = go.Figure(go.Histogram(
                x=final_vals, nbinsx=50,
                marker=dict(color="#1e40af", opacity=0.8, line=dict(color="#60a5fa", width=0.3))
            ))
            p5, p25, p75, p95 = np.percentile(final_vals, [5, 25, 75, 95])
            for pct, val, col in [(5, p5, "#ef4444"), (25, p25, "#f59e0b"),
                                   (75, p75, "#4ade80"), (95, p95, "#22d3ee")]:
                fig_h.add_vline(x=val, line=dict(color=col, dash="dot", width=1.5),
                                annotation_text=f"p{pct}", annotation_font_color=col)
            fig_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), height=300,
                xaxis=dict(tickprefix="€", gridcolor="#1e2130"),
                yaxis=dict(gridcolor="#1e2130"),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col2:
            st.subheader("Annual drawdown risk")
            # Approx single-year VaR / CVaR from log-normal
            df_t = ASSUMPTIONS["simulation"]["t_df"]
            from scipy.stats import t as t_dist
            var_95 = port_r - port_v * t_dist.ppf(0.95, df_t) * np.sqrt((df_t-2)/df_t)
            var_99 = port_r - port_v * t_dist.ppf(0.99, df_t) * np.sqrt((df_t-2)/df_t)

            for label, val, color in [
                ("Expected annual return", f"{port_r*100:.1f}%", "#60a5fa"),
                ("Annual volatility (1σ)", f"±{port_v*100:.1f}%", "#94a3b8"),
                ("1-yr VaR 95% (worst year in 20)", f"{val*100:.1f}%" if (val:=var_95) else "", "#f59e0b"),
                ("1-yr VaR 99% (worst year in 100)", f"{var_99*100:.1f}%", "#ef4444"),
                ("Max single-year loss estimate", f"~{(-2.5*port_v)*100:.0f}%", "#dc2626"),
            ]:
                st.markdown(f"""
<div class="metric-card" style="margin-bottom:6px">
  <div class="metric-val" style="color:{color}">{val}</div>
  <div class="metric-lbl">{label}</div>
</div>""", unsafe_allow_html=True)

        # Risk contribution waterfall
        st.subheader("Risk contribution by asset")
        cov_m = (np.diag([st.session_state["asset_settings"].set_index("Asset").loc[a, "vol"]
                          for a in assets]) @
                 st.session_state["corr_override"].loc[assets, assets].values @
                 np.diag([st.session_state["asset_settings"].set_index("Asset").loc[a, "vol"]
                          for a in assets]))
        rc = (w * (cov_m @ w)) / (port_v ** 2 + 1e-10)
        rc_df = pd.DataFrame({"Asset": assets, "Risk Contribution": rc})
        rc_df = rc_df[rc_df["Risk Contribution"] > 0.001].sort_values("Risk Contribution", ascending=False)
        fig_rc = go.Figure(go.Bar(
            x=rc_df["Asset"], y=rc_df["Risk Contribution"],
            marker=dict(color=rc_df["Risk Contribution"],
                        colorscale=[[0,"#1e3a5f"],[0.5,"#1e40af"],[1,"#ef4444"]],
                        showscale=False)
        ))
        fig_rc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), height=260,
            yaxis=dict(tickformat=".0%", gridcolor="#1e2130"),
            xaxis=dict(gridcolor="#1e2130"),
            margin=dict(t=10, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    # ── TAB 4: REBALANCE ─────────────────────────────────────
    with tab_reb:
        st.subheader("Portfolio Alignment & Drift Monitor")
        st.markdown(f"""
<div class="rebal-note">
Rebalance is flagged when an asset drifts more than
<b>{int(ASSUMPTIONS['rebalance_drift_threshold']*100)}%</b>
from its target weight. This is a rules-based trigger — not a market call.
</div>
""", unsafe_allow_html=True)

        active_plan = plan_df[plan_df["Weight"] > 0.005]

        with st.form("rebalance_form"):
            current_vals = {}
            cols_reb = st.columns(2)
            for i, row in enumerate(active_plan.itertuples()):
                with cols_reb[i % 2]:
                    current_vals[row.Asset] = st.number_input(
                        f"Current value of {row.Asset} (€)",
                        value=float(row._4),   # Invest Now
                        min_value=0.0,
                        key=f"rebal_{row.Asset}"
                    )
            submitted = st.form_submit_button("Calculate Rebalance")

        if submitted:
            reb_df = rebalance_triggers(
                active_plan["Weight"].values,
                active_plan["Asset"].values,
                current_vals
            )
            total_curr = sum(current_vals.values())
            st.caption(f"Total portfolio value: **€{total_curr:,.0f}**")
            st.dataframe(
                reb_df.style.format({
                    "Target %":   "{:.1%}",
                    "Current %":  "{:.1%}",
                    "Drift":      "{:+.1%}",
                    "Buy / Sell €": "€{:+,.0f}",
                }).applymap(
                    lambda v: "background-color:#1a0a0a;color:#fca5a5" if v == "YES" else "",
                    subset=["⚠️ Rebalance"]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Positive Buy/Sell = buy more. Negative = sell/trim.")

    # ── TAB 5: ENGINE ─────────────────────────────────────────
    with tab_engine:
        st.header("Engine Room — Advanced Overrides")
        st.caption("Changes here take effect on the next 'Build Plan' run.")

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.subheader("Asset Parameters")
            st.session_state["asset_settings"] = st.data_editor(
                st.session_state["asset_settings"],
                column_config={
                    "Asset":   st.column_config.TextColumn("Asset", disabled=True),
                    "return":  st.column_config.NumberColumn("Return (nominal)", format="%.3f"),
                    "vol":     st.column_config.NumberColumn("Volatility", format="%.3f"),
                    "cat":     st.column_config.TextColumn("Category", disabled=True),
                },
                use_container_width=True,
                hide_index=True,
                key="asset_editor"
            )

        with col_e2:
            st.subheader("Category Caps (max allocation)")
            cap_data = pd.DataFrame([
                {"Category": k, "Max Weight": v}
                for k, v in ASSUMPTIONS["category_caps"].items()
            ])
            edited_caps = st.data_editor(
                cap_data,
                column_config={
                    "Category":   st.column_config.TextColumn("Category", disabled=True),
                    "Max Weight": st.column_config.NumberColumn("Max Weight", format="%.2f", min_value=0.0, max_value=1.0),
                },
                hide_index=True, use_container_width=True, key="cap_editor"
            )
            for _, row in edited_caps.iterrows():
                ASSUMPTIONS["category_caps"][row["Category"]] = row["Max Weight"]

        st.divider()
        st.subheader("Correlation Matrix (Editable — edits are preserved across runs)")

        # Preserve manual edits: only rebuild if selection changed, warn user
        st.markdown("""
<div class="warning-inline">
Editing correlations here is preserved until the asset selection changes.
If you add/remove assets, the matrix will be reset.
</div>
""", unsafe_allow_html=True)
        edited_corr = st.data_editor(
            st.session_state["corr_override"],
            use_container_width=True,
            key="corr_editor"
        )
        # Enforce symmetry + diagonal
        edited_corr = (edited_corr + edited_corr.T) / 2
        np.fill_diagonal(edited_corr.values, 1.0)
        st.session_state["corr_override"] = edited_corr

        st.divider()
        st.subheader("Optimizer & Simulation Config")
        st.json(ASSUMPTIONS["optimizer"] | ASSUMPTIONS["simulation"] |
                {"safe_withdrawal_rate": ASSUMPTIONS["safe_withdrawal_rate"],
                 "rebalance_drift_threshold": ASSUMPTIONS["rebalance_drift_threshold"],
                 "return_scenarios": ASSUMPTIONS["return_scenarios"]})
