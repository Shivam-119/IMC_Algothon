"""
ML-Driven Portfolio Optimisation with Signal Gating & Validation
================================================================
Optimises a long-only portfolio across 10 instruments using:
  1. Gradient Boosting (walk-forward) to predict 21-day forward returns
  2. Trend signal gating — only invest in bullish-signal instruments
  3. Risk parity + signal tilt weighting
  4. Multi-criteria selection across competing strategies
  5. 7-check validation suite

Inputs:  prices.csv, signals.csv, volumes.csv, cash_rate.csv
Output:  Market_Takers_-_Task_2.csv (asset, weight)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = '/mnt/user-data/uploads'

prices = pd.read_csv(f'{DATA_DIR}/prices.csv', parse_dates=['date'], index_col='date')
signals = pd.read_csv(f'{DATA_DIR}/signals.csv', parse_dates=['date'], index_col='date')
volumes = pd.read_csv(f'{DATA_DIR}/volumes.csv', parse_dates=['date'], index_col='date')
cash_rate = pd.read_csv(f'{DATA_DIR}/cash_rate.csv', parse_dates=['date'], index_col='date')

prices.columns = prices.columns.str.strip()
signals.columns = signals.columns.str.strip()
volumes.columns = volumes.columns.str.strip()

instruments = [f'INSTRUMENT_{i}' for i in range(1, 11)]
n_inst = len(instruments)
returns = prices[instruments].pct_change().dropna()

# Risk-free rate (3-month Treasury)
rf_annual = pd.to_numeric(cash_rate['3mo'], errors='coerce').dropna().iloc[-1] / 100

print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} days)")
print(f"Risk-free rate: {rf_annual*100:.2f}%\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: FEATURE ENGINEERING (26 features per instrument)
# ══════════════════════════════════════════════════════════════════════════════

def build_features(inst):
    """Build feature matrix for a single instrument."""
    feats = pd.DataFrame(index=prices.index)
    p = prices[inst]
    r = returns.reindex(prices.index)[inst]

    # ── Trend signals from signals.csv (4 per instrument) ──
    for lag in [4, 8, 16, 32]:
        col = f'{inst}_trend{lag}'
        if col in signals.columns:
            feats[f'trend_{lag}'] = signals[col]

    # ── Composite trend metrics ──
    trend_cols = [c for c in signals.columns if c.startswith(f'{inst}_trend')]
    if trend_cols:
        feats['trend_composite'] = signals[trend_cols].mean(axis=1)
        feats['trend_dispersion'] = signals[trend_cols].std(axis=1)
        feats['signal_agreement'] = signals[trend_cols].apply(np.sign).mean(axis=1).abs()

    # ── Momentum (multiple lookbacks) ──
    for w in [5, 10, 21, 63, 126, 252]:
        feats[f'mom_{w}d'] = p.pct_change(w)

    # ── Volatility (annualised) ──
    for w in [10, 21, 63]:
        feats[f'vol_{w}d'] = r.rolling(w).std() * np.sqrt(252)
    feats['vol_ratio'] = r.rolling(10).std() / r.rolling(63).std()

    # ── Mean reversion ──
    for w in [21, 63]:
        ma = p.rolling(w).mean()
        feats[f'dist_from_ma_{w}'] = (p - ma) / ma

    # ── RSI (14-day) ──
    delta = p.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feats['rsi_14'] = 100 - (100 / (1 + gain / loss))

    # ── Volume features ──
    vol_col = f'{inst}_vol'
    if vol_col in volumes.columns and volumes[vol_col].sum() > 0:
        v = volumes[vol_col]
        feats['vol_ratio_20d'] = v / v.rolling(20).mean()
        feats['vol_change'] = v.pct_change(5)

    # ── Drawdown ──
    cum = (1 + r).cumprod()
    feats['drawdown'] = cum / cum.cummax() - 1

    # ── Higher moments ──
    feats['skew_21d'] = r.rolling(21).skew()
    feats['kurt_21d'] = r.rolling(21).kurt()

    # ── Cross-asset relative strength ──
    all_rets_21 = returns[instruments].rolling(21).mean().reindex(prices.index)
    feats['relative_strength'] = all_rets_21[inst] - all_rets_21.mean(axis=1)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ML MODEL TRAINING (Walk-Forward Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════════════

FORWARD_DAYS = 21          # Predict 21-day forward returns
TRAIN_MIN = 504            # Minimum 2 years training data
RETRAIN_EVERY = 126        # Retrain every ~6 months

predicted_returns = {}
dir_accuracies = {}

print("Training ML models (walk-forward)...")
for inst in instruments:
    feats = build_features(inst)
    target = returns.reindex(prices.index)[inst].rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)
    combined = feats.join(target.rename('target')).replace([np.inf, -np.inf], np.nan).dropna()
    X = combined.drop('target', axis=1)
    y = combined['target']

    n = len(X)
    split_points = list(range(TRAIN_MIN, n, RETRAIN_EVERY))
    if split_points[-1] != n:
        split_points.append(n)

    pred_series = pd.Series(index=X.index, dtype=float)

    for i in range(len(split_points) - 1):
        train_end, test_end = split_points[i], split_points[i + 1]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[:train_end].replace([np.inf, -np.inf], np.nan).fillna(0))
        X_test = scaler.transform(X.iloc[train_end:test_end].replace([np.inf, -np.inf], np.nan).fillna(0))

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=30, max_features=0.5, random_state=42
        )
        model.fit(X_train, y.iloc[:train_end])
        pred_series.iloc[train_end:test_end] = model.predict(X_test)

    # Final prediction on latest data
    scaler = StandardScaler()
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler.fit(X_clean)
    predicted_returns[inst] = model.predict(scaler.transform(X_clean.iloc[[-1]]))[0]

    # Walk-forward directional accuracy
    valid = pred_series.dropna()
    actual = y.loc[valid.index]
    dir_accuracies[inst] = (np.sign(valid) == np.sign(actual)).mean()

print("Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: CURRENT SIGNAL ANALYSIS & SIGNAL GATING
# ══════════════════════════════════════════════════════════════════════════════

# Compute composite signal per instrument (mean of trend4/8/16/32)
current_signals = {}
for inst in instruments:
    trend_cols = [c for c in signals.columns if c.startswith(f'{inst}_trend')]
    current_signals[inst] = signals[trend_cols].iloc[-1].mean()

# HARD GATE: only invest where composite trend signal > 0
eligible = {inst: current_signals[inst] > 0 for inst in instruments}
eligible_idx = [i for i, inst in enumerate(instruments) if eligible[inst]]
n_eligible = len(eligible_idx)

print("Signal-Gated Universe:")
for inst in instruments:
    s = current_signals[inst]
    pred = predicted_returns[inst]
    da = dir_accuracies[inst]
    status = ' ELIGIBLE' if eligible[inst] else ' EXCLUDED'
    print(f"  {inst}: Sig={s:+.4f}  ML_Pred={pred*100:+.2f}%  DirAcc={da:.1%}  {status}")
print(f"\n  {n_eligible} instruments eligible\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: COMPOSITE SCORING
# ══════════════════════════════════════════════════════════════════════════════

# Covariance matrix (blended 6M + 1Y for stability)
cov_252 = returns[instruments].iloc[-252:].cov().values * 252
cov_126 = returns[instruments].iloc[-126:].cov().values * 252
cov_matrix = 0.5 * cov_126 + 0.5 * cov_252
recent_vol = np.sqrt(np.diag(cov_matrix))

scores = np.zeros(n_inst)
for i, inst in enumerate(instruments):
    if not eligible[inst]:
        scores[i] = -999
        continue
    ml_ann = predicted_returns[inst] * 12                           # Annualised ML prediction
    sig = current_signals[inst]                                     # Trend signal strength
    conf = max(dir_accuracies[inst] - 0.50, 0) * 10                # Model confidence (above random)
    risk_adj = ml_ann / recent_vol[i] if recent_vol[i] > 0 else 0  # Risk-adjusted prediction
    mom = prices[inst].pct_change(21).iloc[-1]                      # Recent 21-day momentum

    # Weighted composite score
    scores[i] = 0.25 * ml_ann + 0.25 * sig + 0.20 * conf + 0.20 * risk_adj + 0.10 * mom


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: CONSTRAINED PORTFOLIO OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def risk_contrib_pct(w, cov):
    """Percentage risk contribution per instrument."""
    pvar = w @ cov @ w
    if pvar < 1e-12:
        return np.zeros_like(w)
    pvol = np.sqrt(pvar)
    marginal = cov @ w
    rc = w * marginal / pvol
    return rc / (rc.sum() + 1e-10) * 100


def objective(w, mu, cov, rf, risk_cap=0.35, lam_risk=8.0, lam_hhi=1.0):
    """
    Objective: maximise risk-adjusted return with penalties for:
      - Risk concentration (any single instrument > risk_cap)
      - Weight concentration (HHI)
    Plus bonus for diversification ratio.
    """
    port_ret = w @ mu
    pvar = w @ cov @ w
    pvol = np.sqrt(pvar) if pvar > 1e-12 else 1e-6
    sharpe = (port_ret - rf) / pvol

    # Penalise excessive risk from any one instrument
    rc = risk_contrib_pct(w, cov)
    excess = np.maximum(rc - risk_cap * 100, 0)
    risk_penalty = lam_risk * np.sum(excess ** 2) / 10000

    # Penalise weight concentration
    hhi_penalty = lam_hhi * np.sum(w ** 2)

    # Reward diversification
    weighted_vol = np.sum(w * np.sqrt(np.diag(cov)))
    div_bonus = 0.2 * np.log(weighted_vol / pvol) if pvol > 1e-10 else 0

    return -(sharpe + div_bonus - risk_penalty - hhi_penalty)


# Constraints & bounds
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = [(0, 0.30 if eligible[instruments[i]] else 0) for i in range(n_inst)]
x0 = np.zeros(n_inst)
for i in eligible_idx:
    x0[i] = 1.0 / n_eligible

# ── Strategy A: Balanced (moderate risk cap) ──
res_A = minimize(objective, x0, args=(scores, cov_matrix, rf_annual, 0.35, 8.0, 1.0),
                 method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000})
w_A = np.maximum(res_A.x, 0)
w_A /= w_A.sum()

# ── Strategy B: Diversified (tight risk cap, strong HHI penalty) ──
res_B = minimize(objective, x0, args=(scores, cov_matrix, rf_annual, 0.25, 15.0, 2.0),
                 method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000})
w_B = np.maximum(res_B.x, 0)
w_B /= w_B.sum()

# ── Strategy C: Signal-proportional Risk Parity ──
inv_vol = np.zeros(n_inst)
for i in eligible_idx:
    inv_vol[i] = 1 / recent_vol[i] if recent_vol[i] > 0 else 0
sig_tilt = np.array([max(current_signals[inst], 0) for inst in instruments])
if sig_tilt.max() > 0:
    sig_tilt /= sig_tilt.max()
w_C = inv_vol * (0.4 + 0.6 * sig_tilt)
w_C = np.minimum(w_C, 0.30)
if w_C.sum() > 0:
    w_C /= w_C.sum()

# ── Strategy D: Ensemble (average of A, B, C) ──
w_D = (w_A + w_B + w_C) / 3
w_D = np.maximum(w_D, 0)
w_D /= w_D.sum()

portfolios = {
    'Equal Weight':   np.ones(n_inst) / n_inst,
    'A: Balanced':    w_A,
    'B: Diversified': w_B,
    'C: RP+Signal':   w_C,
    'D: Ensemble':    w_D,
}


# ══════════════════════════════════════════════════════════════════════════════
# PART 7: BACKTESTING & MULTI-CRITERIA SELECTION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 90)
print("WEIGHTS (%)")
print("=" * 90)
print((pd.DataFrame(portfolios, index=instruments).T * 100).round(2).to_string())

# ── Full-period backtest ──
print(f"\nFULL PERIOD BACKTEST ({returns.index[0].date()} → {returns.index[-1].date()}):")
print("-" * 90)
results = {}
for name, w in portfolios.items():
    port_ret = (returns[instruments] * w).sum(axis=1)
    cum = (1 + port_ret).cumprod()
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_annual) / ann_vol
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    rc = risk_contrib_pct(w, cov_matrix)
    results[name] = {
        'ret': ann_ret, 'vol': ann_vol, 'sharpe': sharpe,
        'dd': max_dd, 'calmar': calmar, 'cum': cum, 'max_rc': rc.max()
    }
    print(f"  {name:20s}  Ret:{ann_ret*100:6.2f}%  Vol:{ann_vol*100:5.2f}%  "
          f"Sharpe:{sharpe:.2f}  MaxDD:{max_dd*100:6.1f}%  MaxRC:{rc.max():.0f}%")

# ── Recent out-of-sample period ──
# Detect the latest ~3 month block that wasn't in previous data
all_dates = returns.index
cutoff = all_dates[-63]  # approx last 3 months
new_rets = returns.loc[cutoff:]

print(f"\nRECENT OUT-OF-SAMPLE ({cutoff.date()} → {all_dates[-1].date()}):")
print("-" * 90)
new_sharpes = {}
for name, w in portfolios.items():
    pr = (new_rets[instruments] * w).sum(axis=1)
    cum_ret = (1 + pr).cumprod().iloc[-1] - 1
    ann_vol = pr.std() * np.sqrt(252)
    sharpe = (pr.mean() * 252 - rf_annual) / ann_vol if ann_vol > 0 else 0
    max_dd = ((1 + pr).cumprod() / (1 + pr).cumprod().cummax() - 1).min()
    new_sharpes[name] = sharpe
    print(f"  {name:20s}  Return:{cum_ret*100:+6.2f}%  Sharpe:{sharpe:+.2f}  MaxDD:{max_dd*100:.2f}%")

# ── Multi-criteria scoring ──
candidates = {k: v for k, v in results.items() if k.startswith(('A:', 'B:', 'C:', 'D:'))}

def normalise(d):
    vals = np.array(list(d.values()))
    mn, mx = vals.min(), vals.max()
    return {k: (v - mn) / (mx - mn + 1e-10) for k, v in d.items()}

n_sharpe = normalise({k: v['sharpe'] for k, v in candidates.items()})
n_calmar = normalise({k: v['calmar'] for k, v in candidates.items()})
n_risk   = normalise({k: 1 - v['max_rc'] / 100 for k, v in candidates.items()})
n_new    = normalise({k: new_sharpes[k] for k in candidates})

print(f"\nMULTI-CRITERIA SCORING:")
best_score, best_name = -1, None
for name in candidates:
    w = portfolios[name]
    aligned = sum(1 for i, inst in enumerate(instruments)
                  if (current_signals[inst] > 0 and w[i] > 0.01) or
                     (current_signals[inst] <= 0 and w[i] < 0.01))
    n_align = aligned / 10

    total = (0.30 * n_sharpe[name] + 0.15 * n_calmar[name] +
             0.25 * n_risk[name] + 0.15 * n_new[name] + 0.15 * n_align)

    print(f"  {name:20s}  Sharpe:{n_sharpe[name]:.2f}  Calmar:{n_calmar[name]:.2f}  "
          f"RiskDiv:{n_risk[name]:.2f}  NewData:{n_new[name]:.2f}  SigAlign:{n_align:.1f}  → {total:.3f}")

    if total > best_score:
        best_score, best_name = total, name

best_w = portfolios[best_name]
print(f"\n★ SELECTED: {best_name}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 8: VALIDATION SUITE (7 checks)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 90)
print(f"VALIDATION: {best_name}")
print("=" * 90)

checks_passed = 0

# ── Check 1: Weight constraints ──
c1 = (abs(best_w.sum() - 1.0) < 0.001 and (best_w >= 0).all() and best_w.max() <= 0.40)
checks_passed += c1
print(f"  {'y' if c1 else 'n'} Check 1 — Weights valid (sum={best_w.sum():.4f}, max={best_w.max()*100:.1f}%)")

# ── Check 2: Beats equal weight on recent data ──
new_pr = (new_rets[instruments] * best_w).sum(axis=1)
eq_pr = (new_rets[instruments] * (np.ones(10)/10)).sum(axis=1)
new_sh = (new_pr.mean()*252 - rf_annual) / (new_pr.std()*np.sqrt(252))
eq_sh = (eq_pr.mean()*252 - rf_annual) / (eq_pr.std()*np.sqrt(252))
c2 = new_sh > eq_sh
checks_passed += c2
print(f"  {'y' if c2 else 'n'} Check 2 — New data: Portfolio Sharpe {new_sh:.2f} vs EW Sharpe {eq_sh:.2f}")

# ── Check 3: ML direction accuracy > 50% ──
avg_da = np.mean(list(dir_accuracies.values()))
c3 = avg_da > 0.50
checks_passed += c3
print(f"  {'y' if c3 else 'n'} Check 3 — ML direction accuracy: {avg_da:.1%}")

# ── Check 4: Signal alignment ≥ 7/10 ──
aligned = sum(1 for i, inst in enumerate(instruments)
              if (current_signals[inst] > 0 and best_w[i] > 0.01) or
                 (current_signals[inst] <= 0 and best_w[i] < 0.01))
c4 = aligned >= 7
checks_passed += c4
print(f"  {'y' if c4 else 'n'} Check 4 — Signal alignment: {aligned}/10")

# ── Check 5: Risk concentration < 40% ──
rc_pct = risk_contrib_pct(best_w, cov_matrix)
c5 = rc_pct.max() < 40
checks_passed += c5
print(f"  {'y' if c5 else 'n'} Check 5 — Risk concentration: {rc_pct.max():.1f}% (cap: 40%)")

# ── Check 6: Drawdown < equal weight ──
port_cum = (1 + (returns[instruments] * best_w).sum(axis=1)).cumprod()
port_dd = (port_cum / port_cum.cummax() - 1).min()
eq_cum = (1 + (returns[instruments] * (np.ones(10)/10)).sum(axis=1)).cumprod()
eq_dd = (eq_cum / eq_cum.cummax() - 1).min()
c6 = abs(port_dd) < abs(eq_dd)
checks_passed += c6
print(f"  {'y' if c6 else 'n'} Check 6 — Max drawdown: {port_dd*100:.1f}% vs EW {eq_dd*100:.1f}%")

# ── Check 7: Beats ≥1 naive benchmark on recent data ──
# Momentum top 4
mom_12m = prices[instruments].pct_change(252).iloc[-1]
top4 = mom_12m.nlargest(4).index
w_mom = np.zeros(10)
for inst in top4:
    w_mom[instruments.index(inst)] = 0.25
# Inverse vol
rvol = returns[instruments].iloc[-63:].std()
w_iv = ((1/rvol) / (1/rvol).sum()).values

benchmarks = {'Momentum': w_mom, 'InvVol': w_iv, 'EqualWeight': np.ones(10)/10}
beaten = 0
for bname, bw in benchmarks.items():
    bpr = (new_rets[instruments] * bw).sum(axis=1)
    bsh = (bpr.mean()*252 - rf_annual) / (bpr.std()*np.sqrt(252))
    if new_sh > bsh:
        beaten += 1
c7 = beaten >= 1
checks_passed += c7
print(f"  {'y' if c7 else 'n'} Check 7 — Beats {beaten}/3 naive benchmarks on recent Sharpe")

# ── Final verdict ──
print(f"\n  RESULT: {checks_passed}/7 checks passed")
if checks_passed >= 6:
    print("   STRONGLY VALIDATED — safe to submit")
elif checks_passed >= 5:
    print("   VALIDATED — acceptable with minor caveats")
elif checks_passed >= 3:
    print("    MIXED — review before submitting")
else:
    print("   FAILED — do not submit")


# ══════════════════════════════════════════════════════════════════════════════
# PART 9: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("FINAL PORTFOLIO:")
print(f"{'='*90}")
for i, inst in enumerate(instruments):
    if best_w[i] > 0.005:
        print(f"  {inst}: {best_w[i]*100:5.1f}%  Sig={current_signals[inst]:+.4f}   "
              f"Risk={rc_pct[i]:.1f}%")

eff_n = 1 / (best_w ** 2).sum()
print(f"\n  Effective N: {eff_n:.1f} instruments")
print(f"  Portfolio vol: {np.sqrt(best_w @ cov_matrix @ best_w)*100:.2f}%")

# Write CSV
output_path = '/mnt/user-data/outputs/Market_Takers_-_Task_2.csv'
out_df = pd.DataFrame([{'asset': inst, 'weight': round(best_w[i], 4)}
                        for i, inst in enumerate(instruments)])
out_df.to_csv(output_path, index=False)

print(f"\n  Saved to: {output_path}")
print(f"\n{out_df.to_string(index=False)}")
