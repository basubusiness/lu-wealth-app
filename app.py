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
    },
    # ── US ────────────────────────────────────────────────────────────────
    "US Equity":             {
        "return": 0.078, "vol": 0.17, "cat": "Equity",
        "note": "S&P 500 or total US market. Overlaps with World Equity.",
        "overlap": ["World Equity"],
    },
    # ── Non-US Developed ──────────────────────────────────────────────────
    # Clean complement to US Equity — no overlap with World Equity if used
    # in place of it, or a useful explicit ex-US tilt alongside World.
    "Europe Equity":         {
        "return": 0.070, "vol": 0.16, "cat": "Equity",
        "note": "Developed Europe (STOXX 600 / MSCI Europe). No US overlap.",
        "overlap": [],
    },
    "Japan / Pacific":       {
        "return": 0.065, "vol": 0.17, "cat": "Equity",
        "note": "Japan, Australia, Singapore. Low US correlation.",
        "overlap": [],
    },
    # ── Emerging & Frontier ───────────────────────────────────────────────
    "Emerging Markets":      {
        "return": 0.080, "vol": 0.22, "cat": "Equity",
        "note": "MSCI EM — China, India, Brazil, Taiwan, Korea, etc.",
        "overlap": [],
    },
    "Frontier Markets":      {
        "return": 0.085, "vol": 0.28, "cat": "Alt",
        "note": "Pre-EM countries. Higher growth potential, lower liquidity.",
        "overlap": [],
    },
    # ── Size factor ───────────────────────────────────────────────────────
    "Global Small Cap":      {
        "return": 0.082, "vol": 0.19, "cat": "Equity",
        "note": "Size premium tilt. Overlaps with World/US if both held.",
        "overlap": ["World Equity", "US Equity"],
    },
    # ── Real Assets ───────────────────────────────────────────────────────
    "Global REIT":           {
        "return": 0.060, "vol": 0.19, "cat": "Real",
        "note": "Listed real estate. Rate-sensitive; correlates with equity in stress.",
        "overlap": [],
    },
    "Gold":                  {
        "return": 0.050, "vol": 0.17, "cat": "Real",
        "note": "Crisis hedge, low long-run real return. Useful as ballast.",
        "overlap": [],
    },
    "Broad Commodities":     {
        "return": 0.040, "vol": 0.20, "cat": "Real",
        "note": "Energy, metals, agri basket. Inflation hedge.",
        "overlap": [],
    },
    # ── Bonds ─────────────────────────────────────────────────────────────
    "Euro Gov Bonds":        {
        "return": 0.030, "vol": 0.06, "cat": "Bond",
        "note": "EUR sovereign debt. Capital preservation, low return.",
        "overlap": [],
    },
    "Corp Bonds":            {
        "return": 0.035, "vol": 0.07, "cat": "Bond",
        "note": "Investment-grade corporate bonds. Slight credit premium.",
        "overlap": [],
    },
    "Global Inflation Bonds":{
        "return": 0.032, "vol": 0.05, "cat": "Bond",
        "note": "TIPS / linkers. Real return protection.",
        "overlap": [],
    },
    # ── Cash ──────────────────────────────────────────────────────────────
    "Cash":                  {
        "return": 0.025, "vol": 0.01, "cat": "Cash",
        "note": "Money market / short-term deposits. Liquidity buffer.",
        "overlap": [],
    },
    # ── Satellite / Speculative ───────────────────────────────────────────
    "Semiconductors":        {
        "return": 0.090, "vol": 0.30, "cat": "Alt",
        "note": "AI & chip supply chain thematic. High vol, high US concentration.",
        "overlap": ["World Equity", "US Equity"],
    },
    "Uranium / Nuclear":     {
        "return": 0.090, "vol": 0.38, "cat": "Alt",
        "note": "Nuclear energy renaissance theme. Illiquid, policy-sensitive. Hard-capped at 5%.",
        "overlap": [],
        "max_w": 0.05,   # per-asset hard cap — overrides global max
    },
    "Crypto":                {
        "return": 0.100, "vol": 0.75, "cat": "Alt",
        "note": "Bitcoin / broad crypto. Extreme vol. Treat as speculative satellite only.",
        "overlap": [],
        "max_w": 0.05,   # hard cap
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
            "name": "max_diversified",
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
        if min_dist > 0.10:
            kept.append(c)

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
                overlaps = ASSETS[a].get("overlap", [])
                active_overlaps = [o for o in overlaps if st.session_state.get(f"asset_{o}", False)]
                label = a
                st.checkbox(label, key=f"asset_{a}",
                            help=ASSETS[a].get("note", "") +
                            (f" ⚠️ Overlaps with: {', '.join(active_overlaps)}" if active_overlaps else ""))

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
        pairs_str = ", ".join(f"**{p[0]}** + **{p[1]}**" for p in overlap_pairs)
        st.warning(
            f"⚠️ **Overlap detected:** {pairs_str}. "
            f"These assets share underlying holdings (e.g. World Equity contains ~65% US stocks). "
            f"Your effective exposure to the overlapping region is higher than the weights suggest. "
            f"This is fine if intentional — just be aware."
        )

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

        st.session_state["results"] = {
            "w": w, "port_r": port_r, "port_v": port_v,
            "paths_base": paths_base, "paths_crisis": paths_crisis,
            "assets": selected_assets, "sens": sens,
            "alternatives": alternatives,
            "shared_shocks": shared_shocks,  # passed to alternatives for fair comparison
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
    all_opts_early = [{"label": "★ Optimal", "weights": w,
                        "port_r": port_r, "port_v": port_v,
                        "sharpe": (port_r - 0.02) / port_v if port_v > 0 else 0}] + alts_early

    # Portfolio selector — lives above everything, drives all downstream metrics
    sel_labels = [o["label"] for o in all_opts_early]
    active_key = st.session_state.get("active_portfolio_label", "★ Optimal")
    if active_key not in sel_labels:
        active_key = sel_labels[0]

    st.markdown("### Active Portfolio")
    chosen_label_top = st.radio(
        "Select portfolio to view across all tabs:",
        sel_labels,
        index=sel_labels.index(active_key),
        horizontal=True,
        key="active_portfolio_radio",
        help="Switching here updates the plan table, metrics, projection and risk tabs to reflect the selected portfolio."
    )
    st.session_state["active_portfolio_label"] = chosen_label_top
    active = next(o for o in all_opts_early if o["label"] == chosen_label_top)

    # Override active portfolio variables — everything downstream uses these
    w_active      = active["weights"]
    port_r_active = active["port_r"]
    port_v_active = active["port_v"]
    sharpe_active = active["sharpe"]

    if chosen_label_top != "★ Optimal":
        st.info(
            f"Viewing **{chosen_label_top}** — all metrics, projections and risk figures "
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
    monthly_income = med_final * swr / 12
    tipping        = next((i for i in range(1, years + 1)
                           if p50_b[i] * swr / 12 >= monthly), None)
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
                    f"Safe withdrawal income ({int(swr*100)}% SWR)",
                    "",
                    f"SWR = Safe Withdrawal Rate. If your portfolio is worth X at retirement, you can withdraw {int(swr*100)}% per year (X × 0.04 ÷ 12 monthly) without depleting it over a 30-year horizon — based on Bengen (1994). You keep the rest invested so it keeps growing. Does NOT include taxes or withdrawal-phase sequence risk. Many planners use 3–3.5% for extra safety."
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
            all_opts = all_opts_early  # already built above
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
                    summary_rows.append({
                        "Portfolio":          o["label"],
                        "Return":             f"{o['port_r']*100:.2f}%",
                        "Vol":                f"{o['port_v']*100:.1f}%",
                        "Sharpe":             f"{o['sharpe']:.2f}",
                        f"Median{label_sfx}": f"EUR {p50_val:,.0f}",
                        f"Worst 10%{label_sfx}": f"EUR {p10_val:,.0f}",
                        "Worst single year":  f"{wy_loss*100:.1f}%",
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
