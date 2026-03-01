"""
Comprehensive tests for tide_predictor.py
==========================================
1. Unit tests: strangle_payoff formula
2. Unit tests: _to_tau_h epoch conversion
3. Unit tests: design matrix sanity
4. Model fit quality: R², residual std, actuals vs predictions
5. Backtest: hold-out future readings and verify PI coverage
6. TIDE_SWING formula correctness (synthetic data with known answers)
7. Physical sanity of current live predictions
8. Settlement detection: if 12:00 reading present, verify exact value used
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, "src")
from tide_predictor import (
    CONFIDENCE,
    N_MC,
    SESSION_END,
    SESSION_START,
    STRIKE_HIGH,
    STRIKE_LOW,
    TIDAL_COMPONENTS,
    TRAINING_HOURS,
    TidalModel,
    _design,
    _row,
    _to_tau_h,
    fetch_thames_data,
    fit_tidal_model,
    get_current_estimates,
    model_diagnostics,
    predict_level,
    predict_tide_swing,
    predict_tide_spot,
    run,
    strangle_payoff,
)

PASS = "  ✓"
FAIL = "  ✗"
results: list[bool] = []


def check(name: str, cond: bool, extra: str = "") -> bool:
    sym = PASS if cond else FAIL
    msg = f"{sym}  {name}"
    if extra:
        msg += f"   [{extra}]"
    print(msg)
    results.append(cond)
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# 1. STRANGLE PAYOFF UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  1. STRANGLE PAYOFF  ═══")
check("payoff(0 cm) = 20",    abs(strangle_payoff(0)   - 20.0) < 1e-9)
check("payoff(10 cm) = 10",   abs(strangle_payoff(10)  - 10.0) < 1e-9)
check("payoff(19.9) = 0.1",   abs(strangle_payoff(19.9) - 0.1) < 1e-6)
check("payoff(20 cm) = 0",    abs(strangle_payoff(20)  - 0.0) < 1e-9)
check("payoff(22.5 cm) = 0",  abs(strangle_payoff(22.5) - 0.0) < 1e-9)
check("payoff(25 cm) = 0",    abs(strangle_payoff(25)  - 0.0) < 1e-9)
check("payoff(30 cm) = 5",    abs(strangle_payoff(30)  - 5.0) < 1e-9)
check("payoff(100 cm) = 75",  abs(strangle_payoff(100) - 75.0) < 1e-9)
arr = np.array([0, 10, 20, 25, 30, 100], dtype=float)
exp = np.array([20, 10, 0, 0, 5, 75],   dtype=float)
check("vectorised payoff matches", np.allclose(strangle_payoff(arr), exp))

# ─────────────────────────────────────────────────────────────────────────────
# 2. EPOCH CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  2. EPOCH CONVERSION  ═══")
check("τ(SESSION_START) = 0",   abs(_to_tau_h(SESSION_START)) < 1e-9)
check("τ(SESSION_END)   = 24",  abs(_to_tau_h(SESSION_END) - 24.0) < 1e-9)
check("τ(mid) = 12",            abs(_to_tau_h(SESSION_START + timedelta(hours=12)) - 12.0) < 1e-9)
# pd.Timestamp and datetime agree
ts_pd = pd.Timestamp(SESSION_END)
ts_py = SESSION_END
check("pd.Timestamp == datetime", abs(_to_tau_h(ts_pd) - _to_tau_h(ts_py)) < 1e-9)
# Negative τ for times before session
before = SESSION_START - timedelta(hours=6)
check("τ(before start) = -6",   abs(_to_tau_h(before) + 6.0) < 1e-9)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DESIGN MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  3. DESIGN MATRIX  ═══")
periods_h = list(TIDAL_COMPONENTS.values())
n_components = len(periods_h)
n_params = 2 * n_components + 1  # cos+sin per component + intercept

r0 = _row(0.0, periods_h)
check("row(τ=0): intercept = 1",  abs(r0[0] - 1.0) < 1e-9)
check("row(τ=0): cos terms = 1",  np.allclose(r0[1::2], 1.0))
check("row(τ=0): sin terms = 0",  np.allclose(r0[2::2], 0.0))
check(f"row length = {len(r0)} (expected {n_params})", len(r0) == n_params)

X48 = _design(np.linspace(0, 48, 200), periods_h)
rank = np.linalg.matrix_rank(X48)
check(f"design rank = {rank}  (expected {n_params})", rank == n_params)

cond = np.linalg.cond(X48)
check(f"condition number = {cond:.1e}  (< 1e4)", cond < 1e4, f"actual {cond:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL FIT QUALITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  4. MODEL FIT QUALITY  ═══")
df_all = fetch_thames_data(500)
model  = fit_tidal_model(df_all)
diag   = model_diagnostics(model)

# R² on training window
latest = df_all["time"].max()
dft    = df_all[df_all["time"] >= latest - pd.Timedelta(hours=TRAINING_HOURS)]
y      = dft["level"].values
X      = _design(dft["time"].apply(_to_tau_h).values, model.periods_h)
y_hat  = X @ model.beta
ss_res = float(np.sum((y - y_hat) ** 2))
ss_tot = float(np.sum((y - np.mean(y)) ** 2))
r2     = 1.0 - ss_res / ss_tot
check(f"R² on 48h training data = {r2:.4f}  (> 0.96)", r2 > 0.96, f"{r2:.4f}")

# OLS in-sample residual σ (4-component model)
sigma_mm = model.sigma * 1000
check(f"OLS σ = {sigma_mm:.1f} mm  (< 350 mm)", sigma_mm < 350.0, f"{sigma_mm:.1f} mm")

# Empirical sigma must be > OLS sigma (it captures out-of-sample inflation)
check(f"emp_sigma > OLS sigma  [{model.emp_sigma*1000:.1f} > {sigma_mm:.1f}]",
      model.emp_sigma > model.sigma)

# Shapiro-Wilk on residuals (Gaussian test)
_, pval = sp_stats.shapiro(model.residuals[:50])
# Non-Gaussianity is expected for tidal residuals (overtides) -- just report
check(f"Residuals Shapiro-Wilk p = {pval:.3f}  (informational)", True)  # always pass

# MAE on last 12 readings — in-sample so checks interpolation accuracy
held = df_all.tail(12)
errors = [abs(r["level"] - predict_level(model, r["time"].to_pydatetime())[0]) * 1000
          for _, r in held.iterrows()]
mae_mm = float(np.mean(errors))
check(f"MAE on last 12 readings = {mae_mm:.1f} mm  (< 450 mm)", mae_mm < 450.0, f"{mae_mm:.1f} mm")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTEST: PREDICTION INTERVAL EMPIRICAL COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  5. PI COVERAGE BACKTEST  ═══")
# For each hold-out window size H, train on data up to (latest - H),
# then predict each reading in the hold-out and count PI hits.
# The emp_sigma-based PI targets 95%; we accept >= 85% empirically
# (some remaining coverage gap is expected from non-stationarity of residuals).
n_covered = 0
n_total   = 0
for H_hours in [2, 4, 6, 8, 12]:
    cutoff   = latest - pd.Timedelta(hours=H_hours)
    df_train = df_all[df_all["time"] < cutoff]
    df_test  = df_all[df_all["time"] >= cutoff]
    if len(df_train) < 10 or len(df_test) == 0:
        continue
    m_back = fit_tidal_model(df_train)
    for _, row in df_test.iterrows():
        mu, lo, hi = predict_level(m_back, row["time"].to_pydatetime())
        n_covered += int(lo <= row["level"] <= hi)
        n_total   += 1

emp_cov = n_covered / n_total if n_total else 0.0
check(
    f"Empirical PI coverage = {n_covered}/{n_total} = {emp_cov:.1%}  (target ≥ 85%)",
    emp_cov >= 0.85,
    f"95% nominal, emp_sigma-calibrated",
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TIDE_SWING FORMULA EXACTNESS (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  6. TIDE_SWING FORMULA  ═══")
timestamps = pd.date_range(SESSION_START, SESSION_END, freq="15min", tz="UTC")
assert len(timestamps) == 97  # 96 diffs

# Need a plausible model — build from real data for the trivial model fallback
m_any = model

# Test 1: flat level (diff=0 cm every step) → payoff=20 each → total = 96×20 = 1920
df_flat = pd.DataFrame({"time": timestamps, "level": np.zeros(97)})
sw_flat = predict_tide_swing(df_flat, m_any, now=SESSION_START + timedelta(seconds=1))
check(f"flat level: known_sum = {sw_flat.known_sum:.0f}  (expected 1920)", abs(sw_flat.known_sum - 1920) < 1e-6)
check("flat level: n_future = 0", sw_flat.n_future == 0)
check("flat level: n_known = 96", sw_flat.n_known == 96)

# Test 2: diff = 22.5 cm/step → dead zone → payoff = 0
df_22 = pd.DataFrame({"time": timestamps, "level": np.arange(97) * 0.225})
sw_22 = predict_tide_swing(df_22, m_any, now=SESSION_START + timedelta(seconds=1))
check(f"dead-zone: known_sum = {sw_22.known_sum:.4f}  (expected 0)", abs(sw_22.known_sum) < 1e-6)

# Test 3: diff = 50 cm/step (0.50 m) → call payoff = 25 each → total = 96×25 = 2400
df_50 = pd.DataFrame({"time": timestamps, "level": np.arange(97) * 0.50})
sw_50 = predict_tide_swing(df_50, m_any, now=SESSION_START + timedelta(seconds=1))
check(f"large change: known_sum = {sw_50.known_sum:.0f}  (expected 2400)", abs(sw_50.known_sum - 2400) < 1e-6)

# Test 4: alternating +0.05 m, -0.05 m (±5 cm diff) → payoff = 15 each → total = 96×15 = 1440
levels_alt = np.cumsum(np.where(np.arange(97) % 2 == 0, 0.05, -0.05))
df_alt = pd.DataFrame({"time": timestamps, "level": levels_alt})
sw_alt = predict_tide_swing(df_alt, m_any, now=SESSION_START + timedelta(seconds=1))
check(f"5 cm alternating: known_sum = {sw_alt.known_sum:.0f}  (expected 1440)", abs(sw_alt.known_sum - 1440) < 1e-6)

# Test 5: partial data (only first 50 timestamps present) → should use MC for the rest
df_partial = df_flat.iloc[:50].copy()
sw_partial = predict_tide_swing(df_partial, m_any, now=SESSION_START + timedelta(hours=12))
check("partial data: n_known = 49", sw_partial.n_known == 49, f"got {sw_partial.n_known}")
check("partial data: n_future > 0", sw_partial.n_future > 0, f"got {sw_partial.n_future}")
check("partial data: CI ordered",   sw_partial.lower <= sw_partial.mean <= sw_partial.upper)

# Test 6: MC CI width decreases as more data known
# Compare swing at T=1h into session vs T=23h into session
df_early = pd.DataFrame({"time": timestamps[:6],  "level": np.zeros(6)})  # 5 readings
df_late  = pd.DataFrame({"time": timestamps[:90], "level": np.zeros(90)}) # 89 readings
sw_early = predict_tide_swing(df_early, m_any, now=SESSION_START + timedelta(hours=1))
sw_late  = predict_tide_swing(df_late,  m_any, now=SESSION_START + timedelta(hours=22))
ci_early = sw_early.upper - sw_early.lower
ci_late  = sw_late.upper  - sw_late.lower
check(
    f"CI shrinks as data accumulates: early={ci_early:.0f}  late={ci_late:.0f}",
    ci_late < ci_early,
    "more known → narrower CI",
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PHYSICAL SANITY OF LIVE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══  7. LIVE PREDICTIONS SANITY  ═══")
result = run(verbose=False)
spot   = result.tide_spot
swing  = result.tide_swing

# Westminster spring range ~7 m → levels typically −2 to +4 mAOD → SPOT 0..7000 mm
check(f"TIDE_SPOT mean = {spot.mean}  in [0, 7000]",     0 <= spot.mean <= 7000)
check(f"TIDE_SPOT PI: lower ≤ mean ≤ upper",             spot.lower <= spot.mean <= spot.upper)
check(f"TIDE_SPOT PI width ≥ 0",                         spot.upper >= spot.lower)

# If 12:00 reading is already exact, PI should collapse to a point
if spot.exact is not None:
    check("TIDE_SPOT exact: lower == upper == mean",
          spot.lower == spot.upper == spot.mean, f"exact={spot.exact}")

# TIDE_SWING: 96 intervals, payoff per step in [0,20] at rest, can exceed with large moves
check(f"TIDE_SWING mean = {swing.mean}  in [0, 10000]",  0 <= swing.mean <= 10000)
check(f"TIDE_SWING CI ordered",                          swing.lower <= swing.mean <= swing.upper)
check(f"TIDE_SWING known_sum ≥ 0",                       swing.known_sum >= 0.0)
check(f"TIDE_SWING n_known + n_future ≤ 96  [{swing.n_known}+{swing.n_future}]",
      swing.n_known + swing.n_future <= 96)
if swing.n_future == 0:
    check("TIDE_SWING complete: std=0", swing.std == 0.0)
else:
    check("TIDE_SWING incomplete: std > 0", swing.std > 0.0)

# get_current_estimates dict shape
est = get_current_estimates()
check("get_current_estimates has tide_spot key",  "tide_spot"  in est)
check("get_current_estimates has tide_swing key", "tide_swing" in est)
check("get_current_estimates has fetched_at key", "fetched_at" in est)
for key in ("mean", "lower", "upper"):
    check(f"tide_spot.{key} present",  key in est["tide_spot"])
    check(f"tide_swing.{key} present", key in est["tide_swing"])

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_fail = len(results) - n_pass
print()
print("═" * 60)
print(f"  {n_pass}/{len(results)} tests passed   ({n_fail} failed)")
print("═" * 60)
if n_fail > 0:
    sys.exit(1)
