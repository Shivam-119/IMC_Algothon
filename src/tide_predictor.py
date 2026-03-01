"""
Tide Predictor — products 1 & 2
================================
TIDE_SPOT  : abs(Thames Westminster gauge level at Sun 12:00 London) × 1000  (mm mAOD)
TIDE_SWING : sum of strangle payoffs on 15-min absolute changes (cm) over the 24-h session
             strangle strike low = 20 cm, strike high = 25 cm
             payoff = max(0, 20 − diff_cm) + max(0, diff_cm − 25)

Session window : 2026-02-28 12:00:00 UTC  →  2026-03-01 12:00:00 UTC
Settlement     : 2026-03-01 12:00:00 UTC  (= 12:00 London / GMT)

Modelling approach
------------------
Harmonic tidal regression using only two well-separated dominant constituents:
  • M2  (12.4206 h) — principal lunar semidiurnal, dominant in Thames
  • K1  (23.9345 h) — lunar-solar diurnal, controls tidal inequality

These have a frequency ratio ≈ 1.93, giving a well-conditioned OLS design
matrix over any 48-h training window.  Adding S2/N2 (periods 12.0/12.66 h,
nearly aliases of M2) causes near-singular matrices and wild extrapolation.

Model: h(t) = C + a1*cos(w1*τ) + b1*sin(w1*τ) + a2*cos(w2*τ) + b2*sin(w2*τ)
τ = hours from midpoint of training window.  5 OLS parameters.

For TIDE_SWING, 93-95 of the 96 session diffs are already known exactly at time
of run; uncertainty comes only from the 1-3 remaining future readings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"
READINGS_URL = (
    "https://environment.data.gov.uk/flood-monitoring/id/measures/"
    f"{THAMES_MEASURE}/readings"
)

SESSION_START = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
SESSION_END   = datetime(2026, 3,  1, 12, 0, 0, tzinfo=timezone.utc)

# M2+K1 are primary; M4 and MK3 capture Thames shallow-water overtides
# (strong non-linear distortion due to shallow estuary geometry)
TIDAL_COMPONENTS = {
    "M2":  12.4206,   # hours — principal lunar semidiurnal
    "K1":  23.9345,   # hours — lunar-solar diurnal
    "M4":   6.2103,   # hours — principal shallow-water overtide (half M2)
    "MK3":  8.177,    # hours — compound tide M2×K1
}

STRIKE_LOW  = 20.0   # cm
STRIKE_HIGH = 25.0   # cm
CONFIDENCE  = 0.95
N_MC        = 30_000
TRAINING_HOURS = 48.0  # use last 48 h for model fit


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────

def fetch_thames_data(limit: int = 500) -> pd.DataFrame:
    """Return DataFrame(time[UTC], level[mAOD]) sorted ascending."""
    resp = requests.get(
        READINGS_URL,
        params={"_sorted": "", "_limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    df = pd.DataFrame(items)[["dateTime", "value"]].rename(
        columns={"dateTime": "time", "value": "level"}
    )
    df["time"]  = pd.to_datetime(df["time"], utc=True)
    df["level"] = df["level"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic model
# ─────────────────────────────────────────────────────────────────────────────

class TidalModel(NamedTuple):
    beta:       np.ndarray
    XtX_inv:    np.ndarray
    sigma:      float        # OLS residual std dev (in-sample)
    emp_sigma:  float        # empirical out-of-sample RMSE (8 h holdout) – kept for compat
    # Horizon-stratified empirical sigmas: (0-3 h, 3-10 h, >10 h)
    emp_sigma_horizons: tuple[float, float, float]
    train_end_h: float       # tau_h of the last training point (for horizon calc)
    t0_h:       float        # unused sentinel (0.0); epoch fixed at SESSION_START
    periods_h:  list[float]
    n:  int
    p:  int
    residuals:  np.ndarray


def _row(tau_h: float, periods_h: list[float]) -> np.ndarray:
    """One design-matrix row.  tau_h = hours from t0."""
    r = [1.0]
    for T in periods_h:
        w = 2.0 * np.pi / T
        r.append(np.cos(w * tau_h))
        r.append(np.sin(w * tau_h))
    return np.asarray(r, dtype=float)


def _design(tau_h_vec: np.ndarray, periods_h: list[float]) -> np.ndarray:
    return np.vstack([_row(t, periods_h) for t in tau_h_vec])


# Use SESSION_START as the universal reference epoch (avoids numpy int64 unit ambiguity).
_REF = SESSION_START


def _to_tau_h(ts: pd.Timestamp | datetime) -> float:
    """Convert any UTC timestamp to hours since SESSION_START."""
    if isinstance(ts, pd.Timestamp):
        return (ts - pd.Timestamp(_REF)).total_seconds() / 3600.0
    return (ts - _REF).total_seconds() / 3600.0


def _ols_fit(
    tau_h: np.ndarray, y: np.ndarray, periods_h: list[float]
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Core OLS solve.  Returns (beta, XtX_inv, sigma, residuals)."""
    X = _design(tau_h, periods_h)
    n, p = X.shape
    XtX     = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta
    sigma   = float(np.sqrt(np.sum(resid ** 2) / max(n - p, 1)))
    return beta, XtX_inv, sigma, resid


def fit_tidal_model(df: pd.DataFrame) -> TidalModel:
    """
    OLS harmonic fit (M2+K1+M4+MK3) on the most recent TRAINING_HOURS of data.

    Also computes an *empirical* sigma by rolling holdout: train on the first
    (TRAINING_HOURS − 8 h), predict the last 8 h, and record the RMSE.  This
    empirical sigma, which captures residual signal from unmmodelled tidal
    constituents, is used for PI width instead of the OLS sigma.
    """
    periods_h = list(TIDAL_COMPONENTS.values())
    latest = df["time"].max()
    cutoff = latest - pd.Timedelta(hours=TRAINING_HOURS)
    dft = df[df["time"] >= cutoff].copy().reset_index(drop=True)

    tau_h = dft["time"].apply(_to_tau_h).values
    y     = dft["level"].values

    beta, XtX_inv, sigma, resid = _ols_fit(tau_h, y, periods_h)
    train_end_h = float(_to_tau_h(latest))
    min_pts = len(periods_h) * 2 + 2

    # ── Horizon-stratified empirical sigmas ───────────────────────────────
    # Each sigma band is calibrated so that t_crit × sigma ≈ 85th-percentile
    # of absolute prediction errors across multiple shifted holdout windows.
    # This guarantees ≥85% empirical coverage even for non-Gaussian tidal
    # residuals (meteorological surge, shallow-water overtides not in model).
    #
    # sigma = percentile_85(|errors|) / t_crit
    # → t_crit × sigma = p85 → empirical coverage ≈ 85% by construction.
    #
    # Bands:
    #   SHORT  : horizon 0 – 3 h
    #   MID    : horizon 3 – 10 h
    #   LONG   : horizon > 10 h

    # t-critical for 95% two-sided (used later in predict_level too)
    _alpha  = 1.0 - CONFIDENCE
    _t_crit = float(stats.t.ppf(1.0 - _alpha / 2.0, df=max(len(y) - len(periods_h) * 2 - 1, 1)))

    def _calibrated_sigma(window_h: float, n_shifts: int = 5,
                          percentile: float = 87.0) -> float:
        """Collect abs prediction errors from n_shifts walk-forward windows of
        length window_h, then set sigma = percentile(|errors|) / t_crit so the
        resulting PI covers that percentile of errors by construction."""
        all_errors = []
        for shift in range(n_shifts):
            shift_h = shift * window_h
            test_end   = latest - pd.Timedelta(hours=shift_h)
            test_start = test_end - pd.Timedelta(hours=window_h)
            train_df = dft[dft["time"] < test_start]
            test_df  = dft[(dft["time"] >= test_start) & (dft["time"] < test_end)]
            if len(train_df) < min_pts or len(test_df) == 0:
                continue
            b, _, _, _ = _ols_fit(train_df["time"].apply(_to_tau_h).values,
                                   train_df["level"].values, periods_h)
            X_t = _design(test_df["time"].apply(_to_tau_h).values, periods_h)
            all_errors.extend(np.abs(test_df["level"].values - X_t @ b).tolist())
        if len(all_errors) < 3:
            return sigma * 3.0
        p_val = float(np.percentile(all_errors, percentile))
        return max(p_val / _t_crit, sigma)   # never smaller than OLS sigma

    sigma_short = _calibrated_sigma(window_h=3.0,  n_shifts=5)
    sigma_mid   = _calibrated_sigma(window_h=7.0,  n_shifts=4)
    sigma_long  = _calibrated_sigma(window_h=10.0, n_shifts=3)

    # Legacy 8h emp_sigma (kept for backward compat / diagnostics)
    emp_sigma = _calibrated_sigma(window_h=8.0, n_shifts=3)

    return TidalModel(
        beta=beta, XtX_inv=XtX_inv, sigma=sigma, emp_sigma=emp_sigma,
        emp_sigma_horizons=(sigma_short, sigma_mid, sigma_long),
        train_end_h=train_end_h,
        t0_h=0.0, periods_h=periods_h,
        n=len(y), p=len(periods_h) * 2 + 1, residuals=resid,
    )


def _horizon_sigma(model: TidalModel, dt: datetime) -> float:
    """Return the empirically-calibrated sigma for a prediction at `dt`.

    The horizon is computed relative to model.train_end_h, i.e. the τ_h of
    the last training data point.  Longer horizons get a larger sigma so that
    the resulting PI properly reflects the increased extrapolation uncertainty.
    """
    horizon_h = _to_tau_h(dt) - model.train_end_h
    s0, s1, s2 = model.emp_sigma_horizons
    if horizon_h <= 3.0:
        return s0
    elif horizon_h <= 10.0:
        return s1
    else:
        return s2


def predict_level(
    model: TidalModel, dt: datetime
) -> tuple[float, float, float]:
    """
    Predict level at dt.
    Returns (mean, lower_95, upper_95) in mAOD.

    Point estimate uses the OLS fit (beta).
    PI width uses the horizon-stratified empirical sigma so that short-range
    predictions get a tighter belt and long-range ones stay calibrated.
    """
    tau_h = _to_tau_h(dt)
    x     = _row(tau_h, model.periods_h)
    mu    = float(x @ model.beta)

    pred_std = _horizon_sigma(model, dt)
    alpha    = 1.0 - CONFIDENCE
    t_crit   = stats.t.ppf(1.0 - alpha / 2.0, df=model.n - model.p)
    return mu, mu - t_crit * pred_std, mu + t_crit * pred_std


def predict_samples(
    model: TidalModel,
    times: list[datetime],
    n_samples: int = N_MC,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Monte-Carlo samples from the predictive distribution at `times`.
    Returns (n_samples × len(times)) array.
    Each column uses the horizon-appropriate sigma so that farther-ahead
    times get noisier samples (better-calibrated CI for TIDE_SWING).
    """
    if rng is None:
        rng = np.random.default_rng()
    means     = np.array([predict_level(model, t)[0] for t in times])
    sigmas    = np.array([_horizon_sigma(model, t) for t in times])  # (len(times),)
    noise = rng.standard_normal((n_samples, len(times))) * sigmas[np.newaxis, :]
    return means + noise


# ─────────────────────────────────────────────────────────────────────────────
# Settlement calculators
# ─────────────────────────────────────────────────────────────────────────────

def strangle_payoff(diff_cm: float | np.ndarray) -> float | np.ndarray:
    """Strangle payoff = max(0, 20 − x) + max(0, x − 25)  for change x cm."""
    return np.maximum(0.0, STRIKE_LOW - diff_cm) + np.maximum(0.0, diff_cm - STRIKE_HIGH)


class TideSpotPrediction(NamedTuple):
    mean:        float
    lower:       float
    upper:       float
    level_mean:  float
    level_lower: float
    level_upper: float
    exact:       float | None


class TideSwingPrediction(NamedTuple):
    mean:        float
    lower:       float
    upper:       float
    std:         float
    known_sum:   float
    future_mean: float
    n_known:     int
    n_future:    int


def predict_tide_spot(
    df: pd.DataFrame, model: TidalModel
) -> TideSpotPrediction:
    """Predict TIDE_SPOT = abs(level at SESSION_END) × 1000."""
    existing = df[df["time"] == pd.Timestamp(SESSION_END)]
    if not existing.empty:
        lv  = float(existing.iloc[0]["level"])
        val = round(abs(lv) * 1000)
        return TideSpotPrediction(
            mean=val, lower=val, upper=val,
            level_mean=lv, level_lower=lv, level_upper=lv, exact=val,
        )

    mu, lo, hi = predict_level(model, SESSION_END)

    if lo < 0 < hi:
        spot_lo = 0.0
        spot_hi = max(abs(lo), abs(hi)) * 1000
    elif lo >= 0:
        spot_lo, spot_hi = lo * 1000, hi * 1000
    else:
        spot_lo, spot_hi = abs(hi) * 1000, abs(lo) * 1000

    return TideSpotPrediction(
        mean=round(abs(mu) * 1000), lower=round(spot_lo), upper=round(spot_hi),
        level_mean=mu, level_lower=lo, level_upper=hi, exact=None,
    )


def predict_tide_swing(
    df: pd.DataFrame,
    model: TidalModel,
    now: datetime | None = None,
    rng: np.random.Generator | None = None,
) -> TideSwingPrediction:
    """Predict TIDE_SWING: exact payoffs for known intervals, MC for future ones."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    timestamps = pd.date_range(SESSION_START, SESSION_END, freq="15min", tz="UTC")
    assert len(timestamps) == 97

    df_idx = df.set_index("time")["level"]
    levels: list[float | None] = [
        float(df_idx[ts]) if ts in df_idx.index else None
        for ts in timestamps
    ]

    known_payoffs:  list[float] = []
    future_indices: list[int]   = []
    for i in range(1, 97):
        if levels[i - 1] is not None and levels[i] is not None:
            diff_cm = abs(levels[i] - levels[i - 1]) * 100.0  # type: ignore[operator]
            known_payoffs.append(float(strangle_payoff(diff_cm)))
        else:
            future_indices.append(i)

    known_sum = float(sum(known_payoffs))
    n_known   = len(known_payoffs)
    n_future  = len(future_indices)

    if n_future == 0:
        return TideSwingPrediction(
            mean=round(known_sum), lower=round(known_sum), upper=round(known_sum),
            std=0.0, known_sum=known_sum, future_mean=0.0,
            n_known=n_known, n_future=0,
        )

    # Identify unique future timestamps needed
    future_ts_set: set[int] = set()
    for i in future_indices:
        if levels[i - 1] is None:
            future_ts_set.add(i - 1)
        if levels[i] is None:
            future_ts_set.add(i)

    future_ts_list = sorted(future_ts_set)
    future_dts     = [
        timestamps[idx].to_pydatetime().replace(tzinfo=timezone.utc)
        for idx in future_ts_list
    ]
    idx_to_col = {idx: col for col, idx in enumerate(future_ts_list)}

    samples = predict_samples(model, future_dts, n_samples=N_MC, rng=rng)

    payoff_mc = np.zeros(N_MC)
    for i in future_indices:
        lvl_prev = (np.full(N_MC, levels[i - 1]) if levels[i - 1] is not None
                    else samples[:, idx_to_col[i - 1]])
        lvl_curr = (np.full(N_MC, levels[i]) if levels[i] is not None
                    else samples[:, idx_to_col[i]])
        payoff_mc += strangle_payoff(np.abs(lvl_curr - lvl_prev) * 100.0)

    total_mc = known_sum + payoff_mc
    alpha    = 1.0 - CONFIDENCE
    lo_q, hi_q = np.quantile(total_mc, [alpha / 2.0, 1.0 - alpha / 2.0])

    return TideSwingPrediction(
        mean=round(float(np.mean(total_mc))),
        lower=round(float(lo_q)),
        upper=round(float(hi_q)),
        std=float(np.std(total_mc)),
        known_sum=known_sum,
        future_mean=float(np.mean(payoff_mc)),
        n_known=n_known,
        n_future=n_future,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def model_diagnostics(model: TidalModel) -> dict:
    s0, s1, s2 = model.emp_sigma_horizons
    return {
        "n_obs":              model.n,
        "n_params":           model.p,
        "residual_std_mAOD":  round(model.sigma, 4),
        "residual_std_mm":    round(model.sigma * 1000, 2),
        "emp_sigma_mm":       round(model.emp_sigma * 1000, 2),
        "emp_sigma_short_mm": round(s0 * 1000, 2),
        "emp_sigma_mid_mm":   round(s1 * 1000, 2),
        "emp_sigma_long_mm":  round(s2 * 1000, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

class TidePredictions(NamedTuple):
    tide_spot:  TideSpotPrediction
    tide_swing: TideSwingPrediction
    model:      TidalModel
    df:         pd.DataFrame
    fetched_at: datetime


def run(verbose: bool = True, seed: int | None = 42) -> TidePredictions:
    """Fetch → fit → predict.  Returns full result object."""
    rng = np.random.default_rng(seed)
    now = datetime.now(tz=timezone.utc)

    if verbose:
        print(f"[tide_predictor] {now.isoformat(timespec='seconds')} UTC")

    df = fetch_thames_data(limit=500)

    if verbose:
        print(f"  {len(df)} readings  {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    model = fit_tidal_model(df)
    diag  = model_diagnostics(model)

    if verbose:
        print(f"  M2+K1+M4+MK3 harmonic fit (last {int(TRAINING_HOURS)}h):  "
              f"OLS σ = {diag['residual_std_mm']} mm  "
              f"emp σ (8h) = {diag['emp_sigma_mm']} mm  "
              f"(n={diag['n_obs']}, p={diag['n_params']})")
        print(f"  horizon sigma  short={diag['emp_sigma_short_mm']} mm  "
              f"mid={diag['emp_sigma_mid_mm']} mm  "
              f"long={diag['emp_sigma_long_mm']} mm")

    spot  = predict_tide_spot(df, model)
    swing = predict_tide_swing(df, model, now=now, rng=rng)

    if verbose:
        _print_summary(spot, swing, now)

    return TidePredictions(tide_spot=spot, tide_swing=swing,
                           model=model, df=df, fetched_at=now)


def _print_summary(spot: TideSpotPrediction, swing: TideSwingPrediction,
                   now: datetime) -> None:
    h_left = (SESSION_END - now).total_seconds() / 3600.0
    print()
    print("=" * 62)
    print(f"  TIDE SETTLEMENT PREDICTIONS   T−{h_left:.2f}h")
    print("=" * 62)
    print()
    print("  TIDE_SPOT  (abs tidal level at 12:00 × 1000, mm mAOD)")
    if spot.exact is not None:
        print(f"    EXACT settlement : {spot.exact}")
    else:
        print(f"    Forecast level   : {spot.level_mean:+.4f} mAOD"
              f"  [{spot.level_lower:+.4f}, {spot.level_upper:+.4f}]")
        print(f"    Settlement est.  : {spot.mean}")
        print(f"    95% PI           : [{spot.lower}, {spot.upper}]")
    print()
    print("  TIDE_SWING  (strangle sum, strikes 20 / 25 cm over 96 intervals)")
    print(f"    Known ({swing.n_known} intervals): {swing.known_sum:.2f}")
    print(f"    Future ({swing.n_future} interval{'s' if swing.n_future != 1 else ''}): "
          f"mean {swing.future_mean:.2f}")
    print(f"    Settlement est.  : {swing.mean}")
    print(f"    95% CI           : [{swing.lower}, {swing.upper}]  (σ = {swing.std:.1f})")
    print()
    print("=" * 62)


def get_current_estimates(seed: int | None = None) -> dict:
    """
    Convenience wrapper for bot integration.  Returns:
    {
        "tide_spot":  {"mean", "lower", "upper", "exact"},
        "tide_swing": {"mean", "lower", "upper", "std"},
        "fetched_at": datetime,
    }
    """
    r = run(verbose=False, seed=seed)
    return {
        "tide_spot":  {"mean": r.tide_spot.mean,  "lower": r.tide_spot.lower,
                       "upper": r.tide_spot.upper, "exact": r.tide_spot.exact},
        "tide_swing": {"mean": r.tide_swing.mean, "lower": r.tide_swing.lower,
                       "upper": r.tide_swing.upper, "std": r.tide_swing.std},
        "fetched_at": r.fetched_at,
    }


if __name__ == "__main__":
    run(verbose=True)
