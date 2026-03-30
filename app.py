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
.metric-tooltip {
    display: none;
    font-size: 12px;
    color: #9ca3af;
    margin-top: 5px;
    line-height: 1.5;
    border-top: 1px solid #1e2130;
    padding-top: 5px;
}
.metric-card:hover .metric-tooltip {
    display: block;
}
.metric-card {
    cursor: default;
    transition: border-color 0.15s;
}
.metric-card:hover {
    border-color: #3b82f6 !important;
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
    # Drawdown guardrail: no portfolio (including spicy ones) should have a
    # simulated worst-year loss (5th pct across paths) worse than this.
    # Principle: asymmetric ambition — upside optionality, not catastrophic downside.
    "max_acceptable_worst_year": -0.35,   # -35% worst year at 5th percentile
    "guardrail_warning_threshold": -0.25, # -25% triggers a yellow warning
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
# "overlap" = list of other assets this one partially contains (for UI warning)
# "note"    = short plain-English description shown in the asset selector
# "max_w"   = optional per-asset hard cap (overrides global max_asset_weight)
ASSETS = {
    # ── Broad / All-World ──────────────────────────────────────────────────
    # Contains ~65% US by market cap. Fine as a single holding (e.g. VWCE).
    # Overlap with US Equity if both selected — effective US exposure doubles.
    "World Equity":          {
        "return": 0.075, "vol": 0.16, "cat": "Equity",
        "note": "All-world market cap (MSCI World / FTSE All-World). ~65% US.",
        "overlap": ["US Equity"],
        "regions": {"US": 0.65, "Europe": 0.15, "Japan/Pacific": 0.10, "Other DM": 0.10},
    },
    # ── US ────────────────────────────────────────────────────────────────
    "US Equity":             {
        "return": 0.078, "vol": 0.17, "cat": "Equity",
        "note": "S&P 500 or total US market. Overlaps with World Equity.",
        "overlap": ["World Equity"],
        "regions": {"US": 1.00},
    },
    # ── Non-US Developed ──────────────────────────────────────────────────
    # Clean complement to US Equity — no overlap with World Equity if used
    # in place of it, or a useful explicit ex-US tilt alongside World.
    "Europe Equity":         {
        "return": 0.070, "vol": 0.16, "cat": "Equity",
        "note": "Developed Europe (STOXX 600 / MSCI Europe). No US overlap.",
        "overlap": [],
        "regions": {"Europe": 1.00},
    },
    "Japan / Pacific":       {
        "return": 0.065, "vol": 0.17, "cat": "Equity",
        "note": "Japan, Australia, Singapore. Low US correlation.",
        "overlap": [],
        "regions": {"Japan/Pacific": 1.00},
    },
    # ── Emerging & Frontier ───────────────────────────────────────────────
    "Emerging Markets":      {
        "return": 0.080, "vol": 0.22, "cat": "Equity",
        "note": "MSCI EM — China, India, Brazil, Taiwan, Korea, etc.",
        "overlap": [],
        "regions": {"China": 0.30, "India": 0.18, "Other EM": 0.52},
    },
    "Frontier Markets":      {
        "return": 0.085, "vol": 0.28, "cat": "Alt",
        "note": "Pre-EM countries. Higher growth potential, lower liquidity.",
        "overlap": [],
        "regions": {"Frontier": 1.00},
    },
    # ── Size factor ───────────────────────────────────────────────────────
    "Global Small Cap":      {
        "return": 0.082, "vol": 0.19, "cat": "Equity",
        "note": "Size premium tilt. Overlaps with World/US if both held.",
        "overlap": ["World Equity", "US Equity"],
        "regions": {"US": 0.55, "Europe": 0.20, "Japan/Pacific": 0.10, "Other DM": 0.15},
    },
    # ── Real Assets ───────────────────────────────────────────────────────
    "Global REIT":           {
        "return": 0.060, "vol": 0.19, "cat": "Real",
        "note": "Listed real estate. Rate-sensitive; correlates with equity in stress.",
        "overlap": [],
        "regions": {"US": 0.55, "Europe": 0.20, "Asia": 0.25},
    },
    "Gold":                  {
        "return": 0.050, "vol": 0.17, "cat": "Real",
        "note": "Crisis hedge, low long-run real return. Useful as ballast.",
        "overlap": [],
        "regions": {"Global": 1.00},
    },
    "Broad Commodities":     {
        "return": 0.040, "vol": 0.20, "cat": "Real",
        "note": "Energy, metals, agri basket (e.g. GSCI, Bloomberg Commodity). ~40% energy.",
        "overlap": [],
        "regions": {"Global": 1.00},
    },
    "Precious Metals":       {
        "return": 0.045, "vol": 0.18, "cat": "Real",
        "note": "Silver, platinum, palladium. Higher industrial demand sensitivity than gold. Lower correlation to equities.",
        "overlap": [],
        "regions": {"Global": 1.00},
    },
    "Energy":                {
        "return": 0.050, "vol": 0.28, "cat": "Real",
        "note": "Oil, gas, energy producers. Strong inflation hedge but high vol and policy risk.",
        "overlap": [],
        "regions": {"Global": 1.00},
    },
    # ── Bonds ─────────────────────────────────────────────────────────────
    "Euro Gov Bonds":        {
        "return": 0.030, "vol": 0.06, "cat": "Bond",
        "note": "EUR sovereign debt. Capital preservation, low return.",
        "overlap": [],
        "regions": {"Europe": 1.00},
    },
    "Corp Bonds":            {
        "return": 0.035, "vol": 0.07, "cat": "Bond",
        "note": "Investment-grade corporate bonds. Slight credit premium.",
        "overlap": [],
        "regions": {"US": 0.55, "Europe": 0.35, "Other": 0.10},
    },
    "Global Inflation Bonds":{
        "return": 0.032, "vol": 0.05, "cat": "Bond",
        "note": "TIPS / linkers. Real return protection.",
        "overlap": [],
        "regions": {"US": 0.50, "Europe": 0.30, "Other": 0.20},
    },
    # ── Cash ──────────────────────────────────────────────────────────────
    "Cash":                  {
        "return": 0.025, "vol": 0.01, "cat": "Cash",
        "note": "Money market / short-term deposits. Liquidity buffer.",
        "overlap": [],
        "regions": {"EUR": 1.00},
    },
    # ── Satellite / Speculative ───────────────────────────────────────────
    "Semiconductors":        {
        "return": 0.090, "vol": 0.30, "cat": "Alt",
        "note": "AI & chip supply chain thematic. High vol, high US concentration.",
        "overlap": ["World Equity", "US Equity"],
        "regions": {"US": 0.65, "Taiwan": 0.15, "South Korea": 0.10, "Other": 0.10},
    },
    "Uranium / Nuclear":     {
        "return": 0.090, "vol": 0.38, "cat": "Alt",
        "note": "Nuclear energy renaissance theme. Illiquid, policy-sensitive. Hard-capped at 5%.",
        "overlap": [],
        "regions": {"Canada": 0.35, "US": 0.25, "Australia": 0.20, "Other": 0.20},
        "max_w": 0.05,
    },
    "Crypto":                {
        "return": 0.100, "vol": 0.75, "cat": "Alt",
        "note": "Bitcoin / broad crypto. Extreme vol. Treat as speculative satellite only.",
        "overlap": [],
        "regions": {"Global": 1.00},
        "max_w": 0.05,
    },
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
        {"type": "ineq", "fun": lambda w: (w @ raw_rets) - target_r},  # HARD floor on RAW returns = user intent
    ]
    for cat, cap in ASSUMPTIONS["category_caps"].items():
        idx = cat_map.get(cat, [])
        if idx:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idx, cap=cap: cap - np.sum(w[ix])
            })

    n = len(names)
    global_max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]
    # Respect per-asset hard caps (e.g. Uranium, Crypto)
    bounds = [(0.0, ASSETS[a].get("max_w", global_max_w)) for a in names]

    best_res = None
    for seed in [np.ones(n)/n, np.random.dirichlet(np.ones(n))]:
        res = minimize(objective, seed, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
        if res.success and (best_res is None or res.fun < best_res.fun):
            best_res = res

    if best_res is None or not best_res.success:
        # Provide diagnostic: what is the actual max achievable return?
        raw_rets = get_base_returns(names)
        shrink   = ASSUMPTIONS["optimizer"]["return_shrinkage"]
        rets_diag = shrink * np.mean(raw_rets) + (1 - shrink) * raw_rets
        cat_map_diag = {cat: [] for cat in ASSUMPTIONS["category_caps"]}
        for i, a in enumerate(names):
            cat_map_diag[ASSETS[a]["cat"]].append(i)
        # Diagnostic uses raw returns (consistent with constraint semantics post shrinkage-fix)
        raw_rets_diag = get_base_returns(names)
        max_uncapped = float(np.max(raw_rets_diag))
        # Find max under category caps only (no return target)
        res_diag = minimize(
            lambda w: -(w @ raw_rets_diag),
            np.ones(len(names))/len(names),
            bounds=[(0, ASSETS[a].get("max_w", ASSUMPTIONS["optimizer"]["max_asset_weight"])) for a in names],
            constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}] + [
                {"type":"ineq","fun":lambda w, ix=idx, cap=cap: cap-np.sum(w[ix])}
                for cat, cap in ASSUMPTIONS["category_caps"].items()
                for idx in [cat_map_diag.get(cat,[])] if idx
            ],
            method="SLSQP"
        )
        max_under_caps = float(-(res_diag.fun)) if res_diag.success else max_uncapped  # raw return ceiling
        # Note: diagnostic also respects per-asset caps
        # Identify which cap is binding
        binding = []
        if res_diag.success:
            w_diag = res_diag.x
            for cat, cap in ASSUMPTIONS["category_caps"].items():
                idx = cat_map_diag.get(cat, [])
                if idx and np.sum(w_diag[idx]) >= cap - 0.01:
                    binding.append(f"{cat} (cap={int(cap*100)}%)")
        return None, {
            "target": target_r,
            "max_achievable": max_under_caps,
            "binding_caps": binding,
        }, None

    w = np.round(best_res.x, 4)
    w[w < 0.005] = 0
    if np.sum(w) > 0:
        w /= np.sum(w)

    port_r = w @ rets
    port_v = np.sqrt(w.T @ cov @ w)
    return w, port_r, port_v


def generate_shocks(years, sims=None):
    """
    Pre-generate standardised t-distributed shocks shared across all simulations.
    Shape: (sims, years). Apply different (mu, sigma) per portfolio for
    apples-to-apples comparison — identical market scenarios, different portfolios.
    This is standard institutional simulation hygiene.
    """
    if sims is None:
        sims = ASSUMPTIONS["simulation"]["paths"]
    df_t  = ASSUMPTIONS["simulation"]["t_df"]
    scale = np.sqrt((df_t - 2) / df_t)
    # Raw standardised draws — mean 0, std ~1
    return np.random.standard_t(df_t, size=(sims, years)) * scale


def simulate_paths(mu, sigma, years, start, monthly, contrib_growth,
                   crisis=False, precomputed_shocks=None):
    """
    Simulate wealth paths using Student-t shocks (fat tails).

    If precomputed_shocks provided (shape: sims x years), uses those directly
    so multiple portfolios see the SAME market scenarios — enabling fair comparison.

    In crisis mode: mu compressed, sigma inflated.
    """
    sims = precomputed_shocks.shape[0] if precomputed_shocks is not None            else ASSUMPTIONS["simulation"]["paths"]
    df_t = ASSUMPTIONS["simulation"]["t_df"]

    if crisis:
        cr    = ASSUMPTIONS["crisis_regime"]
        mu    = mu    * cr["return_scale"]
        sigma = sigma * cr["vol_scale"]

    paths = np.zeros((sims, years + 1))
    paths[:, 0] = start

    for t in range(1, years + 1):
        if precomputed_shocks is not None:
            z = precomputed_shocks[:, t - 1]
        else:
            scale = np.sqrt((df_t - 2) / df_t)
            z = np.random.standard_t(df_t, sims) * scale

        shocks  = (mu - 0.5 * sigma**2) + z * sigma
        contrib = monthly * 12 * ((1 + contrib_growth) ** (t - 1))
        paths[:, t] = paths[:, t - 1] * np.exp(shocks) + contrib

    return paths


def worst_year_loss(paths):
    """
    Empirical worst-year loss: median of the worst single-year return
    across all simulated paths. More intuitive than VaR — answers
    'in a bad year, how much could I lose?'
    """
    yearly_rets = paths[:, 1:] / (paths[:, :-1] + 1e-10) - 1
    worst_per_path = yearly_rets.min(axis=1)
    return float(np.percentile(worst_per_path, 5))  # 5th pct = 1-in-20 year event


def sensitivity_analysis(names, target_r, scenario, years, initial, monthly, contrib_growth):
    """
    Show terminal wealth impact of +-1% return assumption.
    Passes modified returns directly into a local optimizer call
    rather than mutating global session state — avoids subtle race conditions.
    """
    results = {}
    df_base = st.session_state["asset_settings"].set_index("Asset")
    vols = df_base.loc[names, "vol"].values.astype(float)
    corr = st.session_state["corr_override"].loc[names, names].values
    cov  = np.diag(vols) @ corr @ np.diag(vols)
    shrink = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rf = 0.02

    for delta in [-0.01, 0.0, +0.01]:
        raw_rets = df_base.loc[names, "return"].values.astype(float) + delta
        rets = shrink * np.mean(raw_rets) + (1 - shrink) * raw_rets

        cat_map = {cat: [] for cat in ASSUMPTIONS["category_caps"]}
        for i, a in enumerate(names):
            cat_map[ASSETS[a]["cat"]].append(i)

        def objective(w, r=rets):
            pret = w @ r
            pvol = np.sqrt(w.T @ cov @ w + 1e-10)
            return -((pret - rf) / pvol)

        constraints = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, r=raw_rets, t=target_r+delta: (w @ r) - t},  # raw returns for constraint = user intent
        ]
        for cat, cap in ASSUMPTIONS["category_caps"].items():
            idx = cat_map.get(cat, [])
            if idx:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, ix=idx, cap=cap: cap - np.sum(w[ix])
                })

        n = len(names)
        sens_bounds = [(0, ASSETS[a].get("max_w", ASSUMPTIONS["optimizer"]["max_asset_weight"])) for a in names]
        res = minimize(objective, np.ones(n)/n, bounds=sens_bounds,
                       constraints=constraints, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-8})
        if res.success:
            w = res.x; w[w < 0.005] = 0
            if np.sum(w) > 0: w /= np.sum(w)
            mu_sim    = w @ rets
            sigma_sim = np.sqrt(w.T @ cov @ w)
            paths = simulate_paths(mu_sim, sigma_sim, years, initial, monthly, contrib_growth)
            results[delta] = np.percentile(paths[:, -1], 50)
    return results


def check_drawdown_guardrail(port_v, port_r, shared_shocks, years, initial, monthly, growth):
    """
    Check if a portfolio breaches the drawdown guardrail.
    Returns (passes: bool, worst_year_loss: float, severity: str)
    Severity: "ok" | "warning" | "breach"
    """
    paths = simulate_paths(port_r, port_v, years, initial, monthly, growth,
                           crisis=False, precomputed_shocks=shared_shocks)
    wy = worst_year_loss(paths)
    limit   = ASSUMPTIONS["max_acceptable_worst_year"]
    warning = ASSUMPTIONS["guardrail_warning_threshold"]

    if wy < limit:
        return False, wy, "breach"
    elif wy < warning:
        return True, wy, "warning"
    else:
        return True, wy, "ok"


def compute_effective_exposure(weights, names):
    """
    Compute effective geographic/regional exposure by multiplying each asset
    weight by its regional decomposition. Makes World+US overlap visible.
    """
    exposure = {}
    for w_i, name in zip(weights, names):
        if w_i < 0.005:
            continue
        regions = ASSETS[name].get("regions", {"Unclassified": 1.0})
        for region, fraction in regions.items():
            exposure[region] = exposure.get(region, 0.0) + w_i * fraction
    return dict(sorted(exposure.items(), key=lambda x: -x[1]))


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


def _dynamic_label(w_c, names, cat_map):
    """
    Generate an honest label based on what the portfolio *actually* is,
    not what the optimizer intended. Reads the realized allocation.
    """
    def cat_w(cat):
        idx = cat_map.get(cat, [])
        return float(np.sum(w_c[idx])) if idx else 0.0

    eq  = cat_w("Equity")
    bn  = cat_w("Bond")
    re  = cat_w("Real")
    alt = cat_w("Alt")
    cash= cat_w("Cash")

    # Dominant philosophy based on actual weights
    if alt >= 0.19 and eq >= 0.55:
        return "Ultra Spicy (Max Ambition)"
    elif alt >= 0.18:
        return "Moonshot (High Alt)"
    elif alt >= 0.12 and eq >= 0.45:
        return "Growth + Alt Satellite"
    elif alt >= 0.10:
        return "Alt Satellite"
    elif eq >= 0.65:
        return "Equity Dominant"
    elif eq >= 0.50 and bn <= 0.20:
        return "Growth Tilt"
    elif re >= 0.25 and bn <= 0.15:
        return "Inflation Hedge"
    elif re >= 0.18 and eq >= 0.30:
        return "Real Asset Tilt"
    elif bn >= 0.38:
        return "Defensive / Bond Heavy"
    elif bn >= 0.22 and eq >= 0.35 and eq <= 0.62:
        return "Balanced"
    elif cash >= 0.15:
        return "Capital Preservation"
    elif eq >= 0.45:
        return "Growth Tilt"
    else:
        return "Diversified"


def find_alternative_portfolios(names, target_r, n_alternatives=6, sharpe_tol=0.25):
    # Higher sharpe_tol: satellite/moonshot portfolios intentionally sacrifice
    # some Sharpe efficiency for asymmetric return potential. We want to show
    # these even if they're 0.15 Sharpe below optimal.
    """
    Generate structurally distinct portfolios with similar risk/return.

    Key design principle: each philosophy is enforced via HARD minimum
    constraints on category weights — not just biased seeds. This guarantees
    labels reflect actual allocation, not optimizer intent.

    Philosophies:
      1. Max Efficiency    — pure Sharpe, no structural bias
      2. Defensive         — minimise volatility directly (different objective)
      3. Equity Growth     — hard floor: equity >= 60%
      4. Balanced          — hard floors: equity 40-60%, bonds >= 20%
      5. Real Asset Heavy  — hard floor: real >= 20%
      6. Max Diversified   — entropy-maximising, no concentration

    Returns list of dicts with tail_p10 included for comparison.
    """
    df       = st.session_state["asset_settings"].set_index("Asset")
    vols     = df.loc[names, "vol"].values.astype(float)
    raw_rets = get_base_returns(names)
    shrink   = ASSUMPTIONS["optimizer"]["return_shrinkage"]
    rets     = shrink * np.mean(raw_rets) + (1 - shrink) * raw_rets
    corr     = st.session_state["corr_override"].loc[names, names].values
    cov      = np.diag(vols) @ corr @ np.diag(vols)
    rf       = 0.02
    n        = len(names)

    cat_map = {cat: [] for cat in ASSUMPTIONS["category_caps"]}
    for i, a in enumerate(names):
        cat_map[ASSETS[a]["cat"]].append(i)

    # Base constraints shared across all philosophies
    def base_cons(extra_min_floors=None, return_override=None):
        ret_floor = return_override if return_override is not None else target_r
        cons = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, f=ret_floor: (w @ raw_rets) - f},
        ]
        # Category caps (upper bounds)
        for cat, cap in ASSUMPTIONS["category_caps"].items():
            idx = cat_map.get(cat, [])
            if idx:
                cons.append({"type": "ineq",
                             "fun": lambda w, ix=idx, c=cap: c - np.sum(w[ix])})
        # Hard minimum floors for this philosophy
        if extra_min_floors:
            for cat, floor in extra_min_floors.items():
                idx = cat_map.get(cat, [])
                if idx:
                    cons.append({"type": "ineq",
                                 "fun": lambda w, ix=idx, f=floor: np.sum(w[ix]) - f})
        return cons

    global_max_w = ASSUMPTIONS["optimizer"]["max_asset_weight"]
    bounds = [(0.0, ASSETS[a].get("max_w", global_max_w)) for a in names]

    # Define philosophies: (objective_fn, min_floors, seeds)
    eq_idx   = cat_map.get("Equity", [])
    bn_idx   = cat_map.get("Bond",   [])
    re_idx   = cat_map.get("Real",   [])

    philosophies = [
        {
            "name": "max_efficiency",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))),
            "floors": {},
            "seed": np.ones(n) / n,
        },
        {
            # Different objective: minimise portfolio variance directly
            # Accepts a lower return floor (target * 0.75) — explicitly trades
            # return for safety. Label will reflect this honestly.
            "name": "defensive",
            "objective": lambda w: np.sqrt(w.T @ cov @ w + 1e-10) * 100,
            "floors": {"Bond": min(0.30, ASSUMPTIONS["category_caps"].get("Bond", 0.60))},
            "defensive_return_floor": max(0.015, target_r * 0.75),  # adaptive: 75% of target
            "seed": (lambda s: (s.__setitem__(slice(None), 0.02), [s.__setitem__(i, 0.15) for i in bn_idx], s.__itruediv__(s.sum()), s)[-1])(np.ones(n) * 0.02) if bn_idx else np.ones(n)/n,
        },
        {
            # Hard equity floor — guaranteed equity-heavy
            "name": "equity_growth",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))) + 0.05 * np.sum(w**2),
            "floors": {"Equity": min(0.60, ASSUMPTIONS["category_caps"].get("Equity", 0.70))},
            "seed": (lambda s: (s.__setitem__(slice(None), 0.01), [s.__setitem__(i, 0.18) for i in eq_idx], s.__itruediv__(s.sum()), s)[-1])(np.ones(n) * 0.01) if eq_idx else np.ones(n)/n,
        },
        {
            # Balanced: equity + bond ranges (floor AND ceiling)
            # Two-sided constraints give Balanced a real structural identity
            "name": "balanced",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))) + 0.15 * np.sum(w**2),
            "floors": {
                "Equity": min(0.35, ASSUMPTIONS["category_caps"].get("Equity", 0.70)),
                "Bond":   min(0.20, ASSUMPTIONS["category_caps"].get("Bond",   0.60)),
            },
            # Equity ceiling enforced separately below
            "equity_ceiling": 0.60,
            "seed": np.ones(n) / n,
        },
        {
            # Real asset tilt — hard floor on real assets (Sharpe-optimised)
            # Distinct from inflation_hedge which has a higher real floor (0.25)
            "name": "real_assets_sharpe",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))) + 0.10 * np.sum(w**2),
            "floors": {"Real": min(0.18, ASSUMPTIONS["category_caps"].get("Real", 0.40))},
            "seed": (lambda s: (s.__setitem__(slice(None), 0.02), [s.__setitem__(i, 0.15) for i in re_idx], s.__itruediv__(s.sum()), s)[-1])(np.ones(n) * 0.02) if re_idx else np.ones(n)/n,
        },
        {
            # Max entropy — spread as evenly as possible
            # Marked analytical_only: shown in comparison table but not in the
            # active portfolio selector — equal weight != risk-diversified
            "name": "max_diversified",
            "analytical_only": True,
            "objective": lambda w: np.sum(w**2) - 0.3 * (-np.sum(w * np.log(w + 1e-10))),
            "floors": {},
            "seed": np.ones(n) / n,
        },
        {
            # Satellite: Crypto/Alt at meaningful weight alongside solid core
            # Hard floor on Alt — forces the optimizer to actually use it
            # This is the "small bet, asymmetric upside" option
            "name": "alt_satellite",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))) + 0.08 * np.sum(w**2),
            "floors": {
                "Alt":    min(0.10, ASSUMPTIONS["category_caps"].get("Alt", 0.20)),
                "Equity": min(0.40, ASSUMPTIONS["category_caps"].get("Equity", 0.70)),
            },
            "seed": (lambda s: (
                s.__setitem__(slice(None), 0.02),
                [s.__setitem__(i, 0.12) for i in cat_map.get("Alt", [])],
                [s.__setitem__(i, 0.10) for i in cat_map.get("Equity", [])],
                s.__itruediv__(s.sum()), s)[-1]
            )(np.ones(n) * 0.02),
        },
        {
            # Moonshot: maximise SHRUNK return directly — accepts vol, concentrates
            # Alt floor at 15%, Equity floor at 55% — forces concentration
            # Objective has no risk or diversification penalty whatsoever
            "name": "moonshot",
            "objective": lambda w: -(w @ rets + 0.5 * (w @ raw_rets)),  # blend shrunk+raw for concentration
            "floors": {
                "Alt":    min(0.15, ASSUMPTIONS["category_caps"].get("Alt", 0.20)),
                "Equity": min(0.55, ASSUMPTIONS["category_caps"].get("Equity", 0.70)),
            },
            "seed": (lambda s: (
                s.__setitem__(slice(None), 0.01),
                [s.__setitem__(i, 0.15) for i in cat_map.get("Alt", [])],
                [s.__setitem__(i, 0.14) for i in cat_map.get("Equity", [])],
                s.__itruediv__(s.sum()), s)[-1]
            )(np.ones(n) * 0.01),
        },
        {
            # Inflation Hedge: high Real floor (28%+), modest equity
            # Clearly distinct from real_assets_sharpe (18% Real) by hard floor
            "name": "inflation_hedge",
            "objective": lambda w: -((w @ rets - rf) / (np.sqrt(w.T @ cov @ w + 1e-10))) + 0.10 * np.sum(w**2),
            "floors": {
                "Real":   min(0.28, ASSUMPTIONS["category_caps"].get("Real", 0.40)),
                "Equity": min(0.25, ASSUMPTIONS["category_caps"].get("Equity", 0.70)),
            },
            "seed": (lambda s: (
                s.__setitem__(slice(None), 0.02),
                [s.__setitem__(i, 0.13) for i in cat_map.get("Real", [])],
                s.__itruediv__(s.sum()), s)[-1]
            )(np.ones(n) * 0.02),
        },
        {
            # Ultra Spicy: maximum ambition portfolio
            # Objective: pure return maximisation with NO risk penalty
            # Hard floors: Alt at cap (20%), Equity at cap (70% if available)
            # Crypto + high-vol assets will dominate the Alt sleeve
            # This is the "shoot for the moon" option — highest expected return,
            # highest vol, worst tail. Shows what maximum ambition actually costs.
            "name": "ultra_spicy",
            "objective": lambda w: -(w @ raw_rets),  # raw returns — no shrinkage, no risk penalty
            "floors": {
                "Alt":    ASSUMPTIONS["category_caps"].get("Alt", 0.20),   # max out alts
                "Equity": min(0.60, ASSUMPTIONS["category_caps"].get("Equity", 0.70)),
            },
            "seed": (lambda s: (
                s.__setitem__(slice(None), 0.005),
                [s.__setitem__(i, 0.20) for i in cat_map.get("Alt", [])],
                [s.__setitem__(i, 0.15) for i in cat_map.get("Equity", [])],
                s.__itruediv__(s.sum()), s)[-1]
            )(np.ones(n) * 0.005),
        },
    ]

    np.random.seed(42)
    candidates = []

    for phil in philosophies:
        # Defensive uses a lower return floor — explicitly trades return for safety
        effective_target = phil.get("defensive_return_floor", target_r)
        cons = base_cons(phil["floors"], return_override=effective_target)
        # Two-sided equity constraint for Balanced
        if phil.get("equity_ceiling") and eq_idx:
            eq_ceil = phil["equity_ceiling"]
            cons.append({"type": "ineq",
                         "fun": lambda w, ix=eq_idx, c=eq_ceil: c - np.sum(w[ix])})
        obj  = phil["objective"]
        best = None
        for s in [phil["seed"], np.random.dirichlet(np.ones(n)), np.ones(n)/n]:
            try:
                res = minimize(obj, s, bounds=bounds, constraints=cons,
                               method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                continue

        if best is None or not best.success:
            continue

        w_c = np.round(best.x, 4)
        w_c[w_c < 0.005] = 0
        if np.sum(w_c) < 0.01:
            continue
        w_c /= np.sum(w_c)

        pr  = w_c @ rets
        pv  = np.sqrt(w_c.T @ cov @ w_c)
        sh  = (pr - rf) / pv if pv > 0 else 0

        # Honest label from actual allocation
        label = _dynamic_label(w_c, names, cat_map)

        candidates.append({
            "weights": w_c, "port_r": pr, "port_v": pv,
            "sharpe": sh, "label": label,
            "philosophy": phil["name"],
        })

    if not candidates:
        return []

    # Filter: keep within sharpe_tol of best, then deduplicate by L1 > 0.12
    best_sharpe = max(c["sharpe"] for c in candidates)
    filtered    = [c for c in candidates if c["sharpe"] >= best_sharpe - sharpe_tol]

    # Always keep moonshot and ultra_spicy if they exist — they're intentionally
    # different even if L1 distance happens to be small
    priority_philos = {"moonshot", "ultra_spicy", "alt_satellite"}
    priority = [c for c in filtered if c.get("philosophy") in priority_philos]
    rest     = [c for c in filtered if c.get("philosophy") not in priority_philos]

    kept = []
    # First pass: add priority philosophies regardless of distance
    for c in sorted(priority, key=lambda x: -x["sharpe"]):
        kept.append(c)
        if len(kept) >= n_alternatives:
            break

    # Second pass: fill remaining slots with dedup logic
    for c in sorted(rest, key=lambda x: -x["sharpe"]):
        if len(kept) >= n_alternatives:
            break
        if not kept:
            kept.append(c); continue
        min_dist = min(np.sum(np.abs(c["weights"] - k["weights"])) for k in kept)
        if min_dist > 0.15:  # tighter — if two non-priority portfolios are this similar, drop one
            kept.append(c)

    # Post-dedup: ensure all labels are unique
    # If two portfolios ended up with the same dynamic label, use the
    # philosophy-based fallback for the lower-Sharpe one
    PHILOSOPHY_FALLBACK = {
        "max_efficiency":    "★ Max Efficiency",
        "defensive":         "Defensive",
        "equity_growth":     "Equity Growth",
        "balanced":          "Balanced",
        "real_assets_sharpe":"Real Asset Tilt",
        "max_diversified":   "Max Diversified",
        "alt_satellite":     "Alt Satellite",
        "moonshot":          "Moonshot",
        "inflation_hedge":   "Inflation Hedge",
        "ultra_spicy":       "Ultra Spicy",
    }
    seen_labels = {}
    for c in sorted(kept, key=lambda x: -x["sharpe"]):
        lbl = c["label"]
        if lbl in seen_labels:
            # Duplicate — use the philosophy name as the unique label
            c["label"] = PHILOSOPHY_FALLBACK.get(c.get("philosophy", ""), lbl + " (alt)")
        seen_labels[c["label"]] = True

    return sorted(kept, key=lambda x: -x["sharpe"])


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
• No liability matching or cash-flow matching<br>• Target return constraint is applied to <b>base (unshrunk)</b> returns — what you ask for is what the optimizer targets. Shrinkage applies only to the Sharpe objective, adding allocation stability without distorting your intent.<br>
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

    # ── World vs US equity mode ───────────────────────────────────────────
    st.markdown("**Equity Coverage Mode**")
    eq_mode = st.radio(
        "How do you want to cover global equities?",
        [
            "🌍 Single all-world ETF (e.g. VWCE) — World Equity only",
            "🧩 Regional building blocks — US + Europe + Japan/Pacific separately",
            "🔀 All-world + tilt — World Equity as core, add regions for conviction tilts",
        ],
        index=0,
        key="eq_coverage_mode",
        help=(
            "Single all-world: simplest, one ETF covers everything (World Equity ~65% US). "
            "Regional blocks: you control exact country weights, no overlap. "
            "All-world + tilt: World as base, add Europe or Japan to overweight them."
        )
    )

    # Apply mode: enforce mutual exclusivity between World and US Equity
    if "🌍 Single all-world" in eq_mode:
        # Lock: World Equity on, US/Europe/Japan off
        st.session_state["asset_World Equity"]   = True
        st.session_state["asset_US Equity"]       = False
        st.session_state["asset_Europe Equity"]   = False
        st.session_state["asset_Japan / Pacific"] = False
        _locked_assets = {"US Equity", "Europe Equity", "Japan / Pacific"}
        _forced_on     = {"World Equity"}

    elif "🧩 Regional building blocks" in eq_mode:
        # Lock: US/Europe/Japan on, World off
        st.session_state["asset_World Equity"]   = False
        st.session_state["asset_US Equity"]       = True
        st.session_state["asset_Europe Equity"]   = True
        st.session_state["asset_Japan / Pacific"] = True
        _locked_assets = {"World Equity"}
        _forced_on     = {"US Equity", "Europe Equity", "Japan / Pacific"}

    else:
        # All-world + tilt: World on, regions available but optional
        st.session_state["asset_World Equity"] = True
        st.session_state["asset_US Equity"]    = False  # still exclude pure US to avoid double-count
        _locked_assets = {"US Equity"}
        _forced_on     = {"World Equity"}

    st.caption(
        "ℹ️ World Equity and US Equity are mutually exclusive by default — "
        "holding both double-counts your US exposure (~65% of World is already US stocks)."
    )
    st.divider()

    def sync_cat(cat_name, assets_in_cat):
        for a in assets_in_cat:
            if a not in _locked_assets and a not in _forced_on:
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
                is_locked  = a in _locked_assets
                is_forced  = a in _forced_on
                overlaps   = ASSETS[a].get("overlap", [])
                # Only flag overlaps not already handled by mode
                active_overlaps = [
                    o for o in overlaps
                    if st.session_state.get(f"asset_{o}", False)
                    and o not in _locked_assets
                ]
                note = ASSETS[a].get("note", "")
                if is_locked:
                    help_txt = f"🔒 Excluded in '{eq_mode[:20]}...' mode to prevent overlap. Change equity mode above to enable."
                elif is_forced:
                    help_txt = f"✅ Active in '{eq_mode[:20]}...' mode. " + note
                else:
                    help_txt = note + (f" ⚠️ Overlaps with: {', '.join(active_overlaps)}" if active_overlaps else "")

                st.checkbox(
                    a,
                    key=f"asset_{a}",
                    disabled=is_locked or is_forced,
                    help=help_txt,
                )

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
        st.error(f"⚠️ Target {target_pct}% exceeds the highest available raw return ({max_avail*100:.1f}%) in your selected assets. Lower target or enable higher-return assets.")
    elif target > 0.08:
        st.warning(f"⚡ {target_pct}% target requires significant Equity/Alt exposure — review category caps in Engine tab.")
else:
    st.info("Select assets to proceed.")

# Overlap detection banner
if selected_assets:
    overlap_pairs = []
    for a in selected_assets:
        for o in ASSETS[a].get("overlap", []):
            if o in selected_assets:
                pair = tuple(sorted([a, o]))
                if pair not in overlap_pairs:
                    overlap_pairs.append(pair)
    if overlap_pairs:
        # Filter out World/US pair — handled cleanly by the equity mode toggle above
        unhandled = [p for p in overlap_pairs
                     if not (set(p) <= {"World Equity", "US Equity"})]
        if unhandled:
            pairs_str = ", ".join(f"**{p[0]}** + **{p[1]}**" for p in unhandled)
            st.warning(
                f"⚠️ **Overlap detected:** {pairs_str}. "
                f"These assets share underlying holdings. "
                f"Your effective exposure to the overlapping region is higher than the weights suggest. "
                f"This is fine if intentional — just be aware."
            )

# Re-derive selected_assets after equity mode toggle may have changed checkboxes
# This ensures the list reflects the locked/forced state set by the mode
selected_assets = [a for a in ASSETS if st.session_state.get(f"asset_{a}", False)]

# Rebuild correlation matrix when selection changes OR when matrix is stale
# (stale = contains assets not in selected_assets, or missing assets that are selected)
sel_key = tuple(sorted(selected_assets))
corr_stale = (
    "corr_override" not in st.session_state
    or st.session_state.get("_last_sel") != sel_key
    or (selected_assets and not all(
        a in st.session_state["corr_override"].index for a in selected_assets
    ))
)
if corr_stale and selected_assets:
    st.session_state["corr_override"] = build_corr(selected_assets)
    st.session_state["_last_sel"] = sel_key

# ============================================================
# BUILD BUTTON
# ============================================================
if st.button("🏗️  Build Plan", type="primary") and selected_assets:
    with st.spinner("Optimising portfolio..."):
        w, port_r, port_v = optimize_portfolio(selected_assets, target, scenario)

    if w is None:
        diag = port_r  # port_r carries diagnostics dict when w is None
        if isinstance(diag, dict):
            max_r   = diag.get("max_achievable", 0)
            binding = diag.get("binding_caps", [])
            target  = diag.get("target", target)
            msg = (
                f"**Optimiser could not meet your {target*100:.1f}% target** under current constraints.\n\n"
                f"- Maximum achievable return (raw, with caps applied): **{max_r*100:.2f}%**\n"
                f"- Your target exceeds this by **{(target - max_r)*100:.2f}%**\n"
            )
            if binding:
                msg += f"- Binding category caps: {', '.join(binding)}\n"
            msg += "\n**Try:** lower your target, raise a category cap in the Engine tab, or enable higher-return assets."
            st.error(msg)
        else:
            st.error("Optimiser failed — target may be infeasible. Try lowering target or enabling more assets.")
    else:
        with st.spinner("Running Monte Carlo (2 000 paths x 2 regimes)..."):
            # Generate shocks once — reused across main + crisis + alternatives
            # Same market scenarios, different portfolios = fair comparison
            shared_shocks = generate_shocks(years)
            scenario_scale = ASSUMPTIONS["return_scenarios"][scenario]
            sim_mu = port_r * scenario_scale
            paths_base   = simulate_paths(sim_mu, port_v, years, initial, monthly, growth,
                                          crisis=False, precomputed_shocks=shared_shocks)
            crisis_shocks = generate_shocks(years)  # crisis gets own draws (different regime)
            paths_crisis = simulate_paths(sim_mu, port_v, years, initial, monthly, growth,
                                          crisis=True, precomputed_shocks=crisis_shocks)

        # Sensitivity analysis (±1% on returns)
        sens = sensitivity_analysis(selected_assets, target, scenario, years, initial, monthly, growth)

        # Alternative portfolios with similar risk/return
        with st.spinner("Finding alternative portfolio structures..."):
            alternatives = find_alternative_portfolios(selected_assets, target)

        # Run drawdown guardrail check on each alternative
        guardrail_results = {}
        for alt in alternatives:
            passes, wy, severity = check_drawdown_guardrail(
                alt["port_v"], alt["port_r"] * ASSUMPTIONS["return_scenarios"][scenario],
                shared_shocks, years, initial, monthly, growth
            )
            alt["guardrail_passes"]  = passes
            alt["guardrail_wy"]      = wy
            alt["guardrail_severity"]= severity

        st.session_state["results"] = {
            "w": w, "port_r": port_r, "port_v": port_v,
            "paths_base": paths_base, "paths_crisis": paths_crisis,
            "assets": selected_assets, "sens": sens,
            "alternatives": alternatives,
            "shared_shocks": shared_shocks,
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

    # ── Resolve active portfolio selection ───────────────────────────────────
    # Build all_opts early so the radio widget can be shown above the plan table
    alts_early = R.get("alternatives", [])
    # Separate selectable vs analytical-only alternatives
    selectable_alts = [a for a in alts_early if not a.get("analytical_only", False)]
    all_alts_incl   = alts_early  # keep all for comparison table

    all_opts_early = [{"label": "★ Optimal", "weights": w,
                        "port_r": port_r, "port_v": port_v,
                        "sharpe": (port_r - 0.02) / port_v if port_v > 0 else 0,
                        "analytical_only": False}] + selectable_alts

    # Portfolio selector — lives above everything, drives all downstream metrics
    # Append guardrail indicator to labels in selector
    limit   = ASSUMPTIONS["max_acceptable_worst_year"]
    warning = ASSUMPTIONS["guardrail_warning_threshold"]
    for o in all_opts_early:
        wy  = o.get("guardrail_wy", None)
        sev = o.get("guardrail_severity", "ok")
        if wy is not None:
            if sev == "breach":
                o["label_display"] = o["label"] + "  🔴"
            elif sev == "warning":
                o["label_display"] = o["label"] + "  🟡"
            else:
                o["label_display"] = o["label"] + "  🟢"
        else:
            o["label_display"] = o["label"]
    # Optimal portfolio gets its own guardrail check from main paths
    opt_wy = worst_year_loss(maybe_deflate(paths_b))
    all_opts_early[0]["guardrail_wy"] = opt_wy
    all_opts_early[0]["guardrail_severity"] = (
        "breach"  if opt_wy < limit else
        "warning" if opt_wy < warning else "ok"
    )
    all_opts_early[0]["label_display"] = (
        "★ Optimal  🔴" if opt_wy < limit else
        "★ Optimal  🟡" if opt_wy < warning else
        "★ Optimal  🟢"
    )
    sel_labels = [o["label_display"] for o in all_opts_early]
    active_key = st.session_state.get("active_portfolio_label", "★ Optimal")
    if active_key not in sel_labels:
        active_key = sel_labels[0]

    st.markdown("### Active Portfolio")
    # Map display labels back to portfolio objects
    display_to_opt = {o["label_display"]: o for o in all_opts_early}
    active_display = st.session_state.get("active_portfolio_display", sel_labels[0])
    if active_display not in sel_labels:
        active_display = sel_labels[0]

    chosen_display = st.radio(
        "Select portfolio to view across all tabs:",
        sel_labels,
        index=sel_labels.index(active_display),
        horizontal=True,
        key="active_portfolio_radio",
        help=(
            "🟢 within guardrail  🟡 high volatility warning  🔴 breaches drawdown guardrail. "
            "Switching updates plan, metrics, projection and risk tabs."
        )
    )
    st.session_state["active_portfolio_display"] = chosen_display
    active = display_to_opt[chosen_display]
    st.session_state["active_portfolio_label"] = active["label"]

    # Override active portfolio variables — everything downstream uses these
    w_active      = active["weights"]
    port_r_active = active["port_r"]
    port_v_active = active["port_v"]
    sharpe_active = active["sharpe"]

    active_label = active["label"]
    active_sev   = active.get("guardrail_severity", "ok")
    active_wy    = active.get("guardrail_wy", None)

    if active_sev == "breach":
        st.error(
            f"🔴 **Drawdown guardrail breached** — **{active_label}** has a simulated "
            f"worst-year loss of **{active_wy*100:.1f}%** (5th percentile), exceeding the "
            f"{int(abs(ASSUMPTIONS['max_acceptable_worst_year']*100))}% limit. "
            f"This portfolio can cause significant real-money loss in bad years. "
            f"Consider reducing Alt/Equity exposure or selecting a less aggressive philosophy."
        )
    elif active_sev == "warning":
        st.warning(
            f"🟡 **High volatility warning** — **{active_label}** has a simulated "
            f"worst-year loss of **{active_wy*100:.1f}%** (5th percentile). "
            f"This is within the guardrail but expect uncomfortable years. "
            f"Ensure your investment horizon and cash reserves can absorb this."
        )
    elif active_label != "★ Optimal":
        st.info(
            f"Viewing **{active_label}** — all metrics, projections and risk figures "
            f"below reflect this portfolio. Return to **★ Optimal** to see the base plan."
        )

    # Recompute paths for active portfolio using shared shocks
    shared_shocks = R.get("shared_shocks", None)
    scenario_scale = ASSUMPTIONS["return_scenarios"][scenario]
    sim_mu_active  = port_r_active * scenario_scale
    paths_b_active = simulate_paths(sim_mu_active, port_v_active, years, initial, monthly, growth,
                                    crisis=False, precomputed_shocks=shared_shocks)
    paths_c_active = simulate_paths(sim_mu_active, port_v_active, years, initial, monthly, growth,
                                    crisis=True,  precomputed_shocks=R.get("shared_shocks", None))

    # Recompute percentiles for active portfolio
    p10_b, p50_b, p90_b = [maybe_deflate(np.percentile(paths_b_active, p, axis=0)) for p in [10, 50, 90]]
    p10_c, p50_c, p90_c = [maybe_deflate(np.percentile(paths_c_active, p, axis=0)) for p in [10, 50, 90]]

    # Core metrics — all derived from active portfolio
    total_invested = initial + sum(monthly * 12 * ((1 + growth) ** i) for i in range(years))
    real_invested  = total_invested / ((1 + inflation) ** years) if show_real else total_invested
    med_final      = p50_b[-1]
    swr            = ASSUMPTIONS["safe_withdrawal_rate"]

    # Withdrawal mode toggle — sits above the metrics cards
    withdrawal_mode = st.radio(
        "Withdrawal mode",
        ["🏦 Preserve capital (SWR 4%)", "📉 Spend it all (drawdown over N years)"],
        horizontal=True,
        key="withdrawal_mode",
        help=(
            "Preserve capital: withdraw 4% per year indefinitely — principal stays intact and keeps growing. "
            "Spend it all: consume the full portfolio over a set number of years. "
            "Higher monthly income but nothing left at the end."
        )
    )

    if "Preserve" in withdrawal_mode:
        monthly_income     = med_final * swr / 12
        income_label       = f"Monthly income ({int(swr*100)}% SWR, capital preserved)"
        income_tooltip     = (
            f"SWR = Safe Withdrawal Rate. Withdraw {int(swr*100)}% per year — "
            f"portfolio keeps growing and is never fully consumed. "
            f"Sustainable indefinitely if returns hold."
        )
        tipping = next((i for i in range(1, years + 1)
                        if p50_b[i] * swr / 12 >= monthly), None)
    else:
        drawdown_years = st.slider(
            "Spend over how many years?", 5, 40, 25,
            help="How many years do you want the portfolio to last after you stop contributing. "
                 "25 years = retire at 65, run out at 90.",
            key="drawdown_years"
        )
        # Simple annuity: level drawdown ignoring growth (conservative)
        # More accurate: PMT formula accounting for portfolio return during drawdown
        dr = port_r_active  # portfolio still earns during drawdown phase
        if dr > 0:
            # PMT = PV * r / (1 - (1+r)^-n)  where r = annual, n = years
            ann_r = dr
            monthly_income = med_final * (ann_r/12) / (1 - (1 + ann_r/12)**(-drawdown_years*12))
        else:
            monthly_income = med_final / (drawdown_years * 12)
        income_label   = f"Monthly income (spend all over {drawdown_years}y)"
        income_tooltip = (
            f"Draws down the full portfolio over {drawdown_years} years using an annuity formula "
            f"(assumes portfolio earns {port_r_active*100:.1f}% during drawdown). "
            f"Nothing left at the end. Use the SWR mode if you want to preserve capital."
        )
        tipping = next((i for i in range(1, years + 1)
                        if p50_b[i] / (drawdown_years * 12) >= monthly), None)

    # Use active portfolio values throughout
    port_r  = port_r_active
    port_v  = port_v_active
    sharpe  = sharpe_active
    w       = w_active

    # --- Tabs ---
    with st.expander("📖 Glossary — key terms used in this tool", expanded=False):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **SWR (Safe Withdrawal Rate)** | The % of your portfolio you can spend each year in retirement without running out. The classic figure is 4% (Bengen 1994). You keep the rest invested. |
| **Sharpe Ratio** | Return earned per unit of risk taken. (Portfolio return − risk-free rate) ÷ volatility. Higher = more efficient. >0.5 is good, >1.0 is excellent. |
| **Shrinkage** | A statistical technique that pulls return estimates toward the average, reducing overconfidence in any single asset's forecast. |
| **Volatility** | Annualised standard deviation of returns — a measure of how much the portfolio value swings year to year. |
| **VaR (Value at Risk)** | The worst expected loss at a given confidence level. VaR 95% = you would expect to do better than this in 19 out of 20 years. |
| **Monte Carlo** | Running thousands of random simulations of possible future market paths to build a probability distribution of outcomes. |
| **Compounding tipping point** | The year when your portfolio's investment return (at SWR) first exceeds your monthly savings contribution. |
| **Nominal vs Real** | Nominal = future euros as a number. Real = adjusted for inflation, showing today's purchasing power equivalent. |
""")
    tab_plan, tab_proj, tab_risk, tab_reb, tab_etf, tab_engine = st.tabs([
        "📐 Plan", "📈 Projection", "🛡️ Risk", "⚖️ Rebalance", "🔍 ETF Lookup", "⚙️ Engine"
    ])

    # Build plan_df BEFORE tabs so it's available in all tabs (Rebalance needs it)
    _df_lu  = st.session_state["asset_settings"].set_index("Asset")
    _vols_a = np.array([_df_lu.loc[a, "vol"] for a in assets])
    _corr_idx = st.session_state["corr_override"].index.tolist()
    if not all(a in _corr_idx for a in assets):
        st.session_state["corr_override"] = build_corr(assets)
    _corr_a = st.session_state["corr_override"].loc[assets, assets].values
    _cov_a  = np.diag(_vols_a) @ _corr_a @ np.diag(_vols_a)
    _rc     = (w * (_cov_a @ w)) / (port_v ** 2 + 1e-10)

    plan_df = pd.DataFrame({
        "Asset":        assets,
        "Weight":       w,
        "Category":     [ASSETS[a]["cat"] for a in assets],
        "Risk Contrib": _rc,
        "Invest Now":   w * initial,
        "Monthly":      w * monthly,
    })
    plan_df = plan_df[plan_df["Weight"] > 0.005].sort_values("Weight", ascending=False)

    # ── TAB 1: PLAN ──────────────────────────────────────────
    with tab_plan:

        st.dataframe(
            plan_df.style.format({
                "Weight":       "{:.1%}",
                "Risk Contrib": "{:.1%}",
                "Invest Now":   "€{:,.0f}",
                "Monthly":      "€{:,.0f}",
            }).bar(subset=["Weight"], color="#1e3a5f"
            ).bar(subset=["Risk Contrib"], color="#7f1d1d"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**Risk Contrib** = % of total portfolio variance each asset contributes. "
            "A 5% weight with 20% risk contribution means that asset punches above its weight — "
            "useful for spotting hidden concentration (e.g. Crypto at 5% weight, 25%+ risk)."
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
                (
                    f"€{med_final:,.0f}",
                    f"Median wealth{label_sfx}",
                    "",
                    "The 50th percentile outcome across 2,000 Monte Carlo simulations. Half of simulated paths end above this, half below. Not a guarantee — treat as central estimate."
                ),
                (
                    f"{port_r*100:.2f}%",
                    "Portfolio return — base (post-shrinkage)",
                    "",
                    f"Weighted average of base asset returns after {int(ASSUMPTIONS['optimizer']['return_shrinkage']*100)}% shrinkage toward the grand mean. Shrinkage reduces overconfidence in return estimates. This is used for optimisation; your selected scenario scales the simulation mu."
                ),
                (
                    f"{port_v*100:.1f}%",
                    "Portfolio volatility",
                    "",
                    "Annualised standard deviation of portfolio returns, accounting for cross-asset correlations. A 15% vol means roughly a 15% swing (up or down) in a typical year."
                ),
                (
                    f"{sharpe:.2f}",
                    "Sharpe ratio (rf=2%)",
                    "",
                    "Excess return per unit of risk: (portfolio return − 2% risk-free rate) / volatility. Above 0.5 is considered good; above 1.0 is excellent. The optimiser maximises this."
                ),
                (
                    f"€{monthly_income:,.0f}/mo",
                    income_label,
                    "",
                    income_tooltip
                ),
                (
                    f"Year {tipping}" if tipping else ">horizon",
                    "Compounding tipping point",
                    "",
                    "Compounding tipping point: the year when your portfolio's projected annual income (at 4% SWR) first exceeds your monthly savings. After this, the market contributes more to your wealth than your salary does."
                ),
            ]
            for val, lbl, delta, tooltip in metrics:
                st.markdown(f"""
<div class="metric-card" style="margin-bottom:6px" title="{tooltip}">
  <div class="metric-val">{val}</div>
  <div class="metric-lbl">{lbl}</div>
  <div class="metric-tooltip" style="font-size:12px;color:#9ca3af;margin-top:4px;line-height:1.5">{tooltip}</div>
  {"<div class='metric-delta'>"+delta+"</div>" if delta else ""}
</div>""", unsafe_allow_html=True)

        # ── Effective Regional Exposure ───────────────────────
        st.divider()
        st.subheader("Effective Regional Exposure")
        st.caption(
            "Actual geographic exposure after looking through each ETF. "
            "World Equity (65% US) + US Equity = heavily US-concentrated even if weights look balanced."
        )
        eff_exp = compute_effective_exposure(w, assets)
        if eff_exp:
            us_exp = eff_exp.get("US", 0.0)
            if us_exp > 0.50:
                st.error(
                    f"US concentration: {us_exp*100:.1f}% of your portfolio is effectively US-exposed "
                    f"(direct + via World Equity, REIT, Corp Bonds etc). "
                    f"Consider replacing World Equity with Europe/Japan/Pacific, or reducing US Equity."
                )
            elif us_exp > 0.35:
                st.warning(
                    f"US exposure: {us_exp*100:.1f}% — meaningful. Includes indirect exposure "
                    f"via World Equity and sector ETFs."
                )

            col_exp1, col_exp2 = st.columns([2, 3])
            with col_exp1:
                exp_df = pd.DataFrame([
                    {"Region": r, "Effective %": v}
                    for r, v in eff_exp.items()
                ])
                st.dataframe(
                    exp_df.style.format({"Effective %": "{:.1%}"})
                    .bar(subset=["Effective %"], color="#1e3a5f"),
                    use_container_width=True, hide_index=True
                )
            with col_exp2:
                fig_exp = go.Figure(go.Bar(
                    x=list(eff_exp.keys()),
                    y=[v*100 for v in eff_exp.values()],
                    marker=dict(
                        color=[v*100 for v in eff_exp.values()],
                        colorscale=[[0,"#1e3a5f"],[0.5,"#1e40af"],[1,"#ef4444"]],
                        showscale=False,
                    ),
                    text=[f"{v*100:.1f}%" for v in eff_exp.values()],
                    textposition="outside",
                ))
                fig_exp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", size=11),
                    yaxis=dict(ticksuffix="%", gridcolor="#1e2130"),
                    xaxis=dict(gridcolor="#1e2130"),
                    height=280, margin=dict(t=20,b=20,l=10,r=10),
                )
                st.plotly_chart(fig_exp, use_container_width=True)

        # ── Alternative Portfolio Comparison ──────────────────
        alts = R.get("alternatives", [])
        if alts:
            st.divider()
            st.subheader("Compare All Portfolio Structures")
            st.caption(
                "The selector above drives all tabs. This table shows all alternatives "
                "side-by-side so you can compare before choosing. "
                "Worst single year = empirical 1-in-20 year loss from simulation."
            )
            # Comparison table shows ALL alternatives including analytical-only
            all_opts = [{"label": "★ Optimal", "weights": w,
                          "port_r": port_r, "port_v": port_v,
                          "sharpe": (port_r - 0.02) / port_v if port_v > 0 else 0}] + all_alts_incl
            chosen   = active          # driven by top radio

            col_cmp1, col_cmp2 = st.columns([3, 2])
            with col_cmp1:
                cmp_df = pd.DataFrame({
                    "Asset":    assets,
                    "Category": [ASSETS[a]["cat"] for a in assets],
                })
                for o in all_opts:
                    cmp_df[o["label"]] = [f"{v*100:.1f}%" if v > 0.005 else "—"
                                           for v in o["weights"]]
                st.dataframe(cmp_df.set_index("Asset"), use_container_width=True)

            with col_cmp2:
                # Compute tail p10 for each alternative via quick simulation
                scenario_scale = ASSUMPTIONS["return_scenarios"][scenario]
                colors_cmp = ["#60a5fa","#4ade80","#f59e0b","#f87171","#a78bfa","#34d399","#fb923c"]

                fig_cmp = go.Figure()
                summary_rows = []
                shared_shocks_ui = R.get("shared_shocks", None)
                for ci, o in enumerate(all_opts):
                    # Reuse shared shocks — same market scenarios across all portfolios
                    # This gives apples-to-apples comparison of philosophies
                    sim_mu_o  = o["port_r"] * scenario_scale
                    tail_paths = simulate_paths(sim_mu_o, o["port_v"], years,
                                                initial, monthly, growth, crisis=False,
                                                precomputed_shocks=shared_shocks_ui)
                    tail_paths = maybe_deflate(tail_paths)
                    p10_val   = float(np.percentile(tail_paths[:, -1], 10))
                    p50_val   = float(np.percentile(tail_paths[:, -1], 50))
                    wy_loss   = worst_year_loss(tail_paths)
                    o["tail_p10"]       = p10_val
                    o["tail_p50"]       = p50_val
                    o["worst_year"]     = wy_loss

                    fig_cmp.add_trace(go.Bar(
                        name=o["label"],
                        x=["Return %", "Vol %", "Sharpe", "Worst 10% (kEUR)"],
                        y=[o["port_r"]*100, o["port_v"]*100,
                           o["sharpe"], p10_val/1000],
                        marker_color=colors_cmp[ci % len(colors_cmp)],
                        opacity=0.85,
                    ))
                    o_sev = o.get("guardrail_severity", "ok")
                    guardrail_icon = {"ok": "🟢", "warning": "🟡", "breach": "🔴"}.get(o_sev, "")
                    summary_rows.append({
                        "Portfolio":          o["label"],
                        "Return":             f"{o['port_r']*100:.2f}%",
                        "Vol":                f"{o['port_v']*100:.1f}%",
                        "Sharpe":             f"{o['sharpe']:.2f}",
                        f"Median{label_sfx}": f"EUR {p50_val:,.0f}",
                        f"Worst 10%{label_sfx}": f"EUR {p10_val:,.0f}",
                        "Worst year (5th pct)": f"{wy_loss*100:.1f}%",
                        "Guardrail":          guardrail_icon,
                    })

                fig_cmp.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", size=11),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis=dict(gridcolor="#1e2130"),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            # Full comparison summary table below
            st.caption("Tail outcomes computed via Monte Carlo — Worst 10% = the outcome in the worst 1-in-10 scenario.")
            summary_df = pd.DataFrame(summary_rows).set_index("Portfolio")
            st.dataframe(summary_df, use_container_width=True)

            # Active portfolio quick stats
            st.markdown(f"""
<div class="metric-card" style="margin-top:8px;border-color:#3b82f6">
  <div class="metric-val" style="font-size:15px">Active: {chosen["label"]}</div>
  <div class="metric-lbl" style="margin-top:8px">
    Return: {chosen["port_r"]*100:.2f}% &nbsp;|&nbsp;
    Vol: {chosen["port_v"]*100:.1f}% &nbsp;|&nbsp;
    Sharpe: {chosen["sharpe"]:.2f}
  </div>
  <div class="metric-tooltip" style="display:block;font-size:12px;margin-top:8px;color:#9ca3af">
    Select a portfolio in the radio above to update all tabs.
    ETF mapping coming in next version.
  </div>
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

        # ── Source selection ──────────────────────────────────
        holdings     = st.session_state.get("etf_holdings") or []
        active_plan  = plan_df[plan_df["Weight"] > 0.005]
        has_holdings = len([h for h in holdings if float(h.get("Value (EUR)", 0)) > 0]) > 0

        if has_holdings:
            # Aggregate holdings by asset class
            h_df = pd.DataFrame(holdings)
            h_agg = h_df.groupby("Asset Class")["Value (EUR)"].sum()
            total_held = h_agg.sum()

            st.success(
                f"Using **{len(holdings)} ETF holdings** from the ETF Lookup tab "
                f"(total: **€{total_held:,.0f}**) — mapped to model asset classes. "
                f"Add more holdings in the ETF Lookup tab to improve accuracy."
            )
            source_mode = st.radio(
                "Current portfolio values from:",
                ["📂 My ETF holdings (from ETF Lookup tab)", "✏️ Manual entry"],
                horizontal=True, key="reb_source"
            )
        else:
            st.info(
                "No holdings added yet. Go to the **ETF Lookup** tab to add your ETFs, "
                "or use manual entry below."
            )
            source_mode = "✏️ Manual entry"

        st.markdown(f"""
<div class="rebal-note">
Rebalance triggered when drift exceeds
<b>{int(ASSUMPTIONS['rebalance_drift_threshold']*100)}%</b>
from target. Rules-based — not a market call.
Comparing against: <b>{active_label}</b> portfolio.
</div>
""", unsafe_allow_html=True)

        # Auto-rebalance: track holdings version so we recompute when holdings change
        holdings_sig = str(sorted([
            (h.get("Asset Class",""), h.get("Value (EUR)", 0))
            for h in holdings
        ]))
        last_sig = st.session_state.get("_reb_last_sig", "")
        auto_rebalance = (has_holdings and holdings_sig != last_sig)
        if not has_holdings:
            auto_rebalance = False
        if auto_rebalance:
            st.session_state["_reb_last_sig"] = holdings_sig

        # ── Build current_vals from source ────────────────────
        if has_holdings and source_mode and "My ETF" in source_mode:
            # Auto-populate from holdings — map asset class to current value
            current_vals = {}
            for _, row in active_plan.iterrows():
                ac = row["Asset"]
                current_vals[ac] = float(h_agg.get(ac, 0.0))

            # Show what was pulled in
            st.markdown("**Holdings mapped to model asset classes:**")
            mapped_df = pd.DataFrame([
                {"Asset Class": ac, "Value from holdings": current_vals[ac],
                 "Source ETFs": ", ".join(
                     h["Name"] for h in holdings if h["Asset Class"] == ac
                 ) or "—"}
                for ac in active_plan["Asset"].values
            ])
            unmapped = [
                h["Name"] for h in holdings
                if h["Asset Class"] not in active_plan["Asset"].values
            ]
            st.dataframe(
                mapped_df.style.format({"Value from holdings": "€{:,.0f}"}),
                use_container_width=True, hide_index=True
            )
            if unmapped:
                st.warning(
                    f"These holdings are not in your active model portfolio and are excluded: "
                    f"**{', '.join(unmapped)}**. "
                    f"Either add their asset class to the model or change the active portfolio."
                )
            do_rebalance = st.button("Calculate Rebalance", key="reb_auto_btn")

        else:
            # Manual entry form — pre-fill with model Invest Now amounts
            with st.form("rebalance_form"):
                current_vals = {}
                cols_reb = st.columns(2)
                for i, row in enumerate(active_plan.itertuples()):
                    with cols_reb[i % 2]:
                        current_vals[row.Asset] = st.number_input(
                            f"{row.Asset} (€)",
                            value=float(row._4),
                            min_value=0.0,
                            key=f"rebal_{row.Asset}"
                        )
                do_rebalance = st.form_submit_button("Calculate Rebalance")

        # ── Run rebalance logic ───────────────────────────────────
        if "do_rebalance" not in locals():
            do_rebalance = False
        if do_rebalance or auto_rebalance:
            reb_df = rebalance_triggers(
                active_plan["Weight"].values,
                active_plan["Asset"].values,
                current_vals
            )
            total_curr = sum(current_vals.values())

            st.divider()
            col_r1, col_r2 = st.columns([3, 2])
            with col_r1:
                st.caption(
                    f"Total current portfolio: **€{total_curr:,.0f}** | "
                    f"Target: **{active_label}** | "
                    f"Drift threshold: {int(ASSUMPTIONS['rebalance_drift_threshold']*100)}%"
                )
                reb_flagged = reb_df[reb_df["⚠️ Rebalance"] == "YES"]
                if len(reb_flagged) == 0:
                    st.success("✅ Portfolio is within drift thresholds — no rebalance needed.")
                else:
                    st.warning(f"⚠️ {len(reb_flagged)} asset(s) need rebalancing.")

                st.dataframe(
                    reb_df.style.format({
                        "Target %":     "{:.1%}",
                        "Current %":    "{:.1%}",
                        "Drift":        "{:+.1%}",
                        "Buy / Sell €": "€{:+,.0f}",
                    }).applymap(
                        lambda v: "background-color:#1a0a0a;color:#fca5a5" if v == "YES" else "",
                        subset=["⚠️ Rebalance"]
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "**Buy / Sell €**: positive = buy more, negative = trim. "
                    "Amounts are at the asset-class level — split across your ETFs within each class as you see fit."
                )

            with col_r2:
                # Drift waterfall chart
                reb_plot = reb_df[reb_df["Buy / Sell €"].abs() > 10].copy()
                if not reb_plot.empty:
                    fig_reb = go.Figure(go.Bar(
                        x=reb_plot["Asset"],
                        y=reb_plot["Buy / Sell €"],
                        marker=dict(
                            color=["#4ade80" if v > 0 else "#ef4444"
                                   for v in reb_plot["Buy / Sell €"]],
                            opacity=0.85,
                        ),
                        text=[f"€{v:+,.0f}" for v in reb_plot["Buy / Sell €"]],
                        textposition="outside",
                    ))
                    fig_reb.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", size=11),
                        yaxis=dict(tickprefix="€", gridcolor="#1e2130"),
                        xaxis=dict(gridcolor="#1e2130"),
                        height=320,
                        margin=dict(t=20, b=20, l=10, r=10),
                        title=dict(text="Buy / Sell by Asset Class", font=dict(size=12)),
                    )
                    st.plotly_chart(fig_reb, use_container_width=True)

            # Monthly contribution split
            if monthly > 0:
                st.divider()
                st.markdown("**Allocate this month's savings:**")
                monthly_alloc = []
                for _, row in active_plan.iterrows():
                    amt = row["Weight"] * monthly
                    if amt > 1:
                        # If asset is underweight, direct more savings there
                        drift_row = reb_df[reb_df["Asset"] == row["Asset"]]
                        is_under = (not drift_row.empty and
                                    drift_row["Buy / Sell €"].values[0] > 0)
                        monthly_alloc.append({
                            "Asset":      row["Asset"],
                            "Base (€/mo)": amt,
                            "Priority":   "⬆️ Underweight — add here first" if is_under else "—",
                        })
                alloc_df = pd.DataFrame(monthly_alloc)
                st.dataframe(
                    alloc_df.style.format({"Base (€/mo)": "€{:,.0f}"}),
                    use_container_width=True, hide_index=True
                )
                st.caption(
                    f"Your €{monthly:,.0f}/mo contribution split by target weight. "
                    f"Prioritise underweight assets to rebalance gradually without selling."
                )

    # ── TAB 5: ENGINE ─────────────────────────────────────────
    # ── TAB: ETF LOOKUP ──────────────────────────────────────────
    with tab_etf:
        st.subheader("ETF Lookup & Holdings Builder")
        st.caption(
            "Search by ISIN or name. We attempt to fetch key data from justETF. "
            "If lookup fails, a direct link is provided so you can fill in details manually. "
            "Holdings added here automatically populate the Rebalance tab."
        )

        # Holdings store in session state
        if "etf_holdings" not in st.session_state:
            st.session_state["etf_holdings"] = []

        col_srch1, col_srch2 = st.columns([3, 1])
        with col_srch1:
            etf_query = st.text_input(
                "ISIN or ETF name",
                placeholder="e.g. IE00B4L5Y983 or IWDA",
                key="etf_search_input"
            )
        with col_srch2:
            do_search = st.button("🔍 Look up", key="etf_search_btn")

        if do_search and etf_query.strip():
            query = etf_query.strip()
            isin_like = len(query) >= 10 and query[:2].isalpha() and query[2:].replace('-','').isalnum()

            with st.spinner("Fetching from justETF..."):
                import urllib.request, urllib.parse, json, re as _re

                result = {"success": False, "name": None, "ter": None,
                          "index": None, "asset_class": None, "isin": query,
                          "url": None}

                # Build justETF search URL
                if isin_like:
                    search_url = f"https://www.justetf.com/en/etf-profile.html?isin={query}"
                    result["url"] = search_url
                else:
                    enc = urllib.parse.quote(query)
                    search_url = f"https://www.justetf.com/en/search.html?search=ETF&query={enc}"
                    result["url"] = search_url

                # ── Known ISIN lookup table — check this FIRST ────────────
                # Reliable hardcoded data, no scraping needed for common ETFs
                KNOWN_ISINS = {
                        # World / All-world
                        "IE00B4L5Y983": ("iShares Core MSCI World UCITS ETF (IWDA)", 0.20, "World Equity"),
                        "IE00B3RBWM25": ("Vanguard FTSE All-World UCITS ETF (VWRL)", 0.22, "World Equity"),
                        "IE00BK5BQT80": ("Vanguard FTSE All-World UCITS ETF Acc (VWCE)", 0.22, "World Equity"),
                        "LU0274208692": ("Xtrackers MSCI World Swap UCITS ETF", 0.19, "World Equity"),
                        "IE00B4L5YX21": ("iShares MSCI World SRI UCITS ETF", 0.20, "World Equity"),
                        # US Equity
                        "IE00B5BMR087": ("iShares Core S&P 500 UCITS ETF (CSPX)", 0.07, "US Equity"),
                        "IE00BYML9W36": ("iShares Core S&P 500 UCITS ETF EUR Hedged", 0.10, "US Equity"),
                        "LU0490618542": ("Xtrackers S&P 500 Swap UCITS ETF", 0.15, "US Equity"),
                        "IE00B3XXRP09": ("Vanguard S&P 500 UCITS ETF (VUSA)", 0.07, "US Equity"),
                        "IE00BFMXXD54": ("iShares S&P 500 EUR Hedged UCITS ETF", 0.10, "US Equity"),
                        # Europe
                        "IE00B4K48X80": ("iShares Core MSCI Europe UCITS ETF", 0.12, "Europe Equity"),
                        "IE00B3ZW0K18": ("iShares STOXX Europe 600 UCITS ETF", 0.20, "Europe Equity"),
                        "LU0274209237": ("Xtrackers Euro Stoxx 50 UCITS ETF", 0.09, "Europe Equity"),
                        # Emerging Markets
                        "IE00B4L5YC18": ("iShares Core MSCI EM IMI UCITS ETF (EIMI)", 0.18, "Emerging Markets"),
                        "IE00BKM4GZ66": ("iShares Core MSCI Emerging Markets UCITS ETF", 0.18, "Emerging Markets"),
                        "LU0292107645": ("Xtrackers MSCI Emerging Markets Swap UCITS ETF", 0.20, "Emerging Markets"),
                        # Small Cap
                        "IE00BF4RFH31": ("iShares MSCI World Small Cap UCITS ETF", 0.35, "Global Small Cap"),
                        "IE00B3VVMM84": ("Vanguard FTSE All-World Small-Cap UCITS ETF", 0.38, "Global Small Cap"),
                        # Bonds
                        "IE00B3F81R35": ("iShares Core Global Aggregate Bond UCITS ETF (AGGG)", 0.10, "Corp Bonds"),
                        "IE00BDBRDM35": ("iShares Core Global Aggregate Bond EUR Hedged (AGGH)", 0.10, "Corp Bonds"),
                        "IE00B4WXJJ64": ("iShares Euro Government Bond UCITS ETF", 0.15, "Euro Gov Bonds"),
                        "LU0290358497": ("Xtrackers Eurozone Government Bond UCITS ETF", 0.15, "Euro Gov Bonds"),
                        "IE00B3DKXQ41": ("iShares EUR Corporate Bond UCITS ETF", 0.20, "Corp Bonds"),
                        "IE00B0M62X26": ("iShares Global Inflation Linked Govt Bond UCITS ETF", 0.25, "Global Inflation Bonds"),
                        # Real Assets
                        "IE00B8GF1M35": ("iShares Developed Markets Property Yield UCITS ETF", 0.59, "Global REIT"),
                        "IE00B3CNHF87": ("iShares Physical Gold ETC (IGLN)", 0.12, "Gold"),
                        "IE00B4ND3602": ("iShares Gold Producers UCITS ETF", 0.55, "Gold"),
                        "IE00BYXYX521": ("iShares Diversified Commodity Swap UCITS ETF", 0.19, "Broad Commodities"),
                        # Sector / Alt
                        "IE00BGV5VN51": ("iShares MSCI Global Semiconductors UCITS ETF", 0.35, "Semiconductors"),
                        "LU1834983477": ("Amundi MSCI Semiconductors ESG Screened UCITS ETF", 0.35, "Semiconductors"),
                        "IE0005E9BX43": ("Global X Uranium UCITS ETF", 0.65, "Uranium / Nuclear"),
                        "IE000CNSFAR2": ("VanEck Uranium and Nuclear Technologies UCITS ETF", 0.55, "Uranium / Nuclear"),
                        # Precious Metals
                        "IE00B4NCWG09": ("iShares Physical Silver ETC", 0.20, "Precious Metals"),
                        "IE00B579F325": ("iShares Physical Platinum ETC", 0.20, "Precious Metals"),
                        "IE00B4NFYF94": ("iShares Physical Palladium ETC", 0.35, "Precious Metals"),
                        "DE000A0N62G8": ("Xetra-Gold ETC", 0.36, "Gold"),
                        "DE000A0S9GB0": ("Xtrackers IE Physical Gold ETC", 0.25, "Gold"),
                        # Energy
                        "IE00B6R51Z18": ("iShares Oil & Gas Exploration & Production UCITS ETF", 0.55, "Energy"),
                        "IE00BYM31M36": ("iShares MSCI World Energy Sector UCITS ETF", 0.25, "Energy"),
                        "LU1829218749": ("Lyxor MSCI World Energy TR UCITS ETF", 0.30, "Energy"),
                        # Additional Broad Commodities
                        "IE00BYXYX521": ("iShares Diversified Commodity Swap UCITS ETF", 0.19, "Broad Commodities"),
                        "LU1829218582": ("Lyxor Commodities Refinitiv/CoreCommodity CRB UCITS ETF", 0.35, "Broad Commodities"),
                    }

                    # Check known ISIN first — most reliable
                isin_upper = query.upper().replace(" ", "")
                if isin_upper in KNOWN_ISINS:
                    name, ter, ac = KNOWN_ISINS[isin_upper]
                    result.update({
                        "success": True, "name": name,
                        "ter": ter, "asset_class": ac,
                        "isin": isin_upper, "source": "known_db"
                    })
                else:
                    # Fall back to justETF scrape
                    try:
                            req = urllib.request.Request(
                                search_url,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                                    "Accept-Language": "en-GB,en;q=0.9",
                                    "Accept": "text/html,application/xhtml+xml",
                                }
                            )
                            with urllib.request.urlopen(req, timeout=8) as resp:
                                html = resp.read().decode("utf-8", errors="ignore")

                            # Name: title tag is most reliable
                            m = _re.search(r'<title>([^|<]{5,80})', html)
                            if m:
                                result["name"] = m.group(1).strip().rstrip(" -")
                                result["success"] = True

                            # TER: look for the specific justETF pattern
                            # justETF renders: <span class="vallabel">TER</span>...<span>X.XX%</span>
                            ter_block = _re.search(
                                r'TER[^<]{0,200}?([0-9]{1,2}\.[0-9]{2})\s*%',
                                html, _re.IGNORECASE | _re.DOTALL
                            )
                            if ter_block:
                                ter_val = float(ter_block.group(1))
                                if ter_val < 5.0:  # sanity check — TER > 5% is nonsense
                                    result["ter"] = ter_val

                            # Asset class: match ETF NAME first (most reliable signal),
                            # then fall back to page body keywords
                            name_lower = (result.get("name") or "").lower()
                            # Priority: match on name first
                            if any(x in name_lower for x in ["s&p 500", "sp500", "russell", "nasdaq", "us equit"]):
                                result["asset_class"] = "US Equity"
                            elif any(x in name_lower for x in [
                                "msci world", "ftse all-world", "ftse all world", "acwi",
                                "global equit", "prime global", "world ucits", "all world",
                                "global stocks", "world stocks", "world index"
                            ]):
                                result["asset_class"] = "World Equity"
                            elif any(x in name_lower for x in ["emerging market", "msci em", "ftse em"]):
                                result["asset_class"] = "Emerging Markets"
                            elif any(x in name_lower for x in ["europe", "stoxx", "eurostoxx", "euro stoxx"]):
                                result["asset_class"] = "Europe Equity"
                            elif any(x in name_lower for x in ["japan", "topix", "nikkei", "pacific"]):
                                result["asset_class"] = "Japan / Pacific"
                            elif any(x in name_lower for x in ["small cap", "small-cap", "smallcap"]):
                                result["asset_class"] = "Global Small Cap"
                            elif any(x in name_lower for x in ["reit", "real estate", "property"]):
                                result["asset_class"] = "Global REIT"
                            elif any(x in name_lower for x in ["gold", "precious metal"]):
                                result["asset_class"] = "Gold"
                            elif any(x in name_lower for x in ["silver", "platinum", "palladium", "copper", "metal"]):
                                result["asset_class"] = "Precious Metals"
                            elif any(x in name_lower for x in ["commodity", "commodities", "energy", "oil", "gas"]):
                                result["asset_class"] = "Broad Commodities"
                            elif any(x in name_lower for x in ["inflation", "tips", "linker", "inflation-link"]):
                                result["asset_class"] = "Global Inflation Bonds"
                            elif any(x in name_lower for x in ["government bond", "govt bond", "treasury", "bund", "sovereign"]):
                                result["asset_class"] = "Euro Gov Bonds"
                            elif any(x in name_lower for x in ["corporate bond", "corp bond", "aggregate bond", "credit"]):
                                result["asset_class"] = "Corp Bonds"
                            elif any(x in name_lower for x in ["semiconductor", "chip"]):
                                result["asset_class"] = "Semiconductors"
                            elif any(x in name_lower for x in ["uranium", "nuclear"]):
                                result["asset_class"] = "Uranium / Nuclear"

                            # Attempt to extract top 5 holdings from the page
                            # justETF renders holdings in a table with class "holdings-table" or similar
                            holdings_block = _re.search(
                                r'(?:Top Holdings|top 10|largest holdings)[^<]{0,500}?' +
                                r'(<table[^>]*>.*?</table>)',
                                html, _re.IGNORECASE | _re.DOTALL
                            )
                            if holdings_block:
                                rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', holdings_block.group(1), _re.DOTALL)
                                holdings_list = []
                                for row in rows[1:6]:  # skip header, take up to 5
                                    cells = _re.findall(r'<td[^>]*>([^<]{2,50})</td>', row)
                                    if len(cells) >= 2:
                                        name_c = cells[0].strip()
                                        pct_m  = _re.search(r'([0-9]+\.?[0-9]*)\s*%', cells[-1])
                                        if name_c and pct_m:
                                            holdings_list.append((name_c, float(pct_m.group(1))))
                                if holdings_list:
                                    result["top_holdings"] = holdings_list

                    except Exception as e:
                        result["error"] = str(e)

            # Persist result so add form survives beyond the do_search rerun
            st.session_state["etf_last_result"] = result

            # Show result
            if result["success"]:
                src_tag = " *(from database)*" if result.get("source") == "known_db" else " *(from justETF)*"
                st.success(f"Found: **{result['name']}**{src_tag}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("ISIN", result["isin"])
                with c2:
                    st.metric("TER", f"{result['ter']}%" if result["ter"] else "—")
                with c3:
                    st.metric("Suggested mapping", result["asset_class"] or "Unknown — select below")

                # Show top holdings if extracted
                if result.get("top_holdings"):
                    st.markdown("**Top holdings from justETF:**")
                    h_df = pd.DataFrame(result["top_holdings"], columns=["Holding", "Weight %"])
                    st.dataframe(
                        h_df.style.format({"Weight %": "{:.2f}%"})
                        .bar(subset=["Weight %"], color="#1e3a5f"),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(
                        "Top holdings shown for reference only — composition changes over time. "
                        "Check the ETF factsheet for the latest data."
                    )
            else:
                st.warning(
                    f"Could not auto-fetch data for **{query}**. "
                    f"Fill in manually below, or "
                    f"[view on justETF]({result['url']}) to look up the details."
                )

        # ── Add from last search result (always visible after search) ──
        last_result = st.session_state.get("etf_last_result")
        if last_result:
            st.divider()
            if last_result.get("success"):
                st.markdown(f"**Add to holdings:** *{last_result.get('name', '')}*")
            else:
                st.markdown("**Add to holdings manually:**")

            ac_opts_raw = list(ASSETS.keys())
            ac_opts_with_blank = ["— select asset class —"] + ac_opts_raw
            sug_ac  = last_result.get("asset_class") or ""
            sug_idx = ac_opts_with_blank.index(sug_ac) if sug_ac in ac_opts_with_blank else 0
            def_name = last_result.get("name") or last_result.get("isin", "")
            def_isin = last_result.get("isin", "")

            fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 2])
            with fc1:
                sf_name = st.text_input("Name / Ticker", value=def_name, key="sr_f_name")
            with fc2:
                sf_isin = st.text_input("ISIN", value=def_isin, key="sr_f_isin")
            with fc3:
                sf_ac = st.selectbox("Maps to asset class", ac_opts_with_blank, index=sug_idx, key="sr_f_ac")
            with fc4:
                # Use counter in key so value resets to 0 after each add
                _add_cnt = st.session_state.get("etf_add_count", 0)
                sf_val = st.number_input("Value (EUR)", min_value=0.0, value=0.0,
                                          step=100.0, key=f"sr_f_val_{_add_cnt}")

            if not last_result.get("success"):
                st.caption(f"[View on justETF]({last_result.get('url', '')})")

            if st.button("✅ Add to holdings", key="sr_add_btn"):
                if sf_name and sf_val > 0 and sf_ac not in ["— select asset class —", ""]:
                    # Aggregate: if same Name+AssetClass already exists, add to value
                    merged = False
                    for h in st.session_state["etf_holdings"]:
                        if h["Name"].strip().lower() == sf_name.strip().lower() and                            h["Asset Class"] == sf_ac:
                            h["Value (EUR)"] = float(h["Value (EUR)"]) + sf_val
                            merged = True
                            break
                    if not merged:
                        st.session_state["etf_holdings"].append({
                            "Name": sf_name, "ISIN": sf_isin,
                            "Asset Class": sf_ac, "Value (EUR)": sf_val
                        })
                    # Keep result visible so user can add another amount
                    # Increment a counter to reset only the value field
                    st.session_state["etf_add_count"] = st.session_state.get("etf_add_count", 0) + 1
                    action = "Updated" if merged else "Added"
                    st.success(f"✅ **{action}: {sf_name}** (€{sf_val:,.0f}) — see Your Holdings below.")
                elif sf_ac in ["— select asset class —", ""]:
                    st.warning("Please select an asset class before adding.")
                else:
                    st.warning("Enter a value greater than 0 before adding.")

        # Quick-add without searching — for when you know your ETF
        with st.expander("➕ Add ETF manually (without searching)", expanded=False):
            qa1, qa2, qa3, qa4 = st.columns([3, 2, 2, 2])
            with qa1:
                qa_name = st.text_input("Name / Ticker", key="qa_name", placeholder="e.g. VWCE")
            with qa2:
                qa_isin = st.text_input("ISIN", key="qa_isin", placeholder="e.g. IE00BK5BQT80")
            with qa3:
                qa_ac = st.selectbox("Asset class", list(ASSETS.keys()), key="qa_ac")
            with qa4:
                qa_val = st.number_input("Value (EUR)", min_value=0.0, value=0.0,
                                          step=100.0, key="qa_val")
            if st.button("Add", key="qa_add_btn") and qa_name:
                if qa_val > 0:
                    merged = False
                    for h in st.session_state["etf_holdings"]:
                        if h["Name"].strip().lower() == qa_name.strip().lower() and                            h["Asset Class"] == qa_ac:
                            h["Value (EUR)"] = float(h["Value (EUR)"]) + qa_val
                            merged = True
                            break
                    if not merged:
                        st.session_state["etf_holdings"].append({
                            "Name": qa_name, "ISIN": qa_isin,
                            "Asset Class": qa_ac, "Value (EUR)": qa_val
                        })
                    action = "Updated" if merged else "Added"
                    st.success(f"✅ **{action} {qa_name}** — scroll down to Your Holdings, then go to Rebalance tab.")
                else:
                    st.warning("Enter a value greater than 0.")

        # ── Holdings — always visible below search ───────────────
        st.divider()
        st.markdown("### Your Holdings")
        if st.session_state["etf_holdings"]:
            holdings_df = pd.DataFrame(st.session_state["etf_holdings"])
            total_h = holdings_df["Value (EUR)"].sum()
            st.caption(
                f"**{len(st.session_state['etf_holdings'])} ETFs added** · "
                f"Total: **EUR {total_h:,.0f}** · "
                f"Go to the **Rebalance** tab to compare against your model target."
            )
            # Editable table — change asset class mapping or values inline
            edited_holdings = st.data_editor(
                holdings_df,
                column_config={
                    "Name":        st.column_config.TextColumn("ETF Name"),
                    "ISIN":        st.column_config.TextColumn("ISIN"),
                    "Asset Class": st.column_config.SelectboxColumn(
                        "Asset Class", options=list(ASSETS.keys())
                    ),
                    "Value (EUR)": st.column_config.NumberColumn(
                        "Value (EUR)", min_value=0, format="EUR %.0f"
                    ),
                },
                use_container_width=True, hide_index=True,
                key="holdings_editor", num_rows="dynamic",
            )
            if edited_holdings is not None:
                st.session_state["etf_holdings"] = edited_holdings.to_dict("records")

            # Quick drift summary vs active portfolio
            agg     = holdings_df.groupby("Asset Class")["Value (EUR)"].sum()
            agg_pct = agg / total_h if total_h > 0 else agg
            target_w_map = dict(zip(assets, w))
            all_classes  = sorted(set(list(agg_pct.index) + [
                a for a, wi in target_w_map.items() if wi > 0.005
            ]))
            drift_rows = []
            for ac in all_classes:
                actual    = float(agg_pct.get(ac, 0.0))
                target_wt = float(target_w_map.get(ac, 0.0)) if target_w_map.get(ac, 0.0) > 0.005 else 0.0
                drift_rows.append({
                    "Asset Class":  ac,
                    "You hold":     actual,
                    "Model target": target_wt,
                    "Drift":        actual - target_wt,
                })
            drift_df = pd.DataFrame(drift_rows)
            drift_df = drift_df[drift_df[["You hold","Model target"]].max(axis=1) > 0.001]
            st.markdown("**Quick drift vs model target:**")
            st.dataframe(
                drift_df.style
                    .format({"You hold": "{:.1%}", "Model target": "{:.1%}", "Drift": "{:+.1%}"})
                    .bar(subset=["Drift"], align="zero", color=["#7f1d1d","#14532d"]),
                use_container_width=True, hide_index=True
            )
            st.caption(
                "Drift = actual minus target. Green = underweight. Red = overweight. "
                "Go to **Rebalance** tab for full buy/sell signals."
            )
        else:
            st.info(
                "No holdings yet — search for an ETF above and click **Add to holdings**. "
                "Your list will appear here and automatically populate the Rebalance tab."
            )


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
        # Enforce symmetry + diagonal + valid range
        edited_corr = edited_corr.clip(-1.0, 1.0)
        edited_corr = (edited_corr + edited_corr.T) / 2
        np.fill_diagonal(edited_corr.values, 1.0)
        extreme = ((edited_corr.abs() > 0.95) & (edited_corr != 1.0)).any().any()
        if extreme:
            st.warning("⚠️ Some correlations exceed ±0.95 — this may make the covariance matrix near-singular and destabilise the optimiser.")
        st.session_state["corr_override"] = edited_corr

        st.divider()
        st.subheader("Asset Utility Analysis")
        st.caption("Assets the optimizer never selects at your current target — consider removing or adjusting their return assumptions.")
        if "results" in st.session_state:
            res_w  = st.session_state["results"]["w"]
            res_as = st.session_state["results"]["assets"]
            dead   = [(a, w_i) for a, w_i in zip(res_as, res_w) if w_i < 0.005]
            active = [(a, w_i) for a, w_i in zip(res_as, res_w) if w_i >= 0.005]
            if dead:
                dead_df = pd.DataFrame([{
                    "Asset": a,
                    "Category": ASSETS[a]["cat"],
                    "Base Return": f"{ASSETS[a]['return']*100:.1f}%",
                    "Volatility": f"{ASSETS[a]['vol']*100:.1f}%",
                    "Note": ASSETS[a].get("note",""),
                    "Status": "❌ Not selected by optimizer"
                } for a, _ in dead])
                st.dataframe(dead_df, use_container_width=True, hide_index=True)
                st.caption(
                    "These assets were available but not chosen. Either their risk-adjusted return "
                    "is dominated by other assets at this target, or the category cap is already full. "
                    "Try: raising their return assumption in the table above, or excluding competing assets."
                )
            else:
                st.success("All selected assets appear in the optimal portfolio.")
        else:
            st.info("Run Build Plan first to see asset utility analysis.")

        st.divider()
        st.subheader("Optimizer & Simulation Config")
        st.json(ASSUMPTIONS["optimizer"] | ASSUMPTIONS["simulation"] |
                {"safe_withdrawal_rate": ASSUMPTIONS["safe_withdrawal_rate"],
                 "rebalance_drift_threshold": ASSUMPTIONS["rebalance_drift_threshold"],
                 "return_scenarios": ASSUMPTIONS["return_scenarios"]})
