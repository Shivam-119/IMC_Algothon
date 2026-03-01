#!/usr/bin/env python3
"""
IMCity Trading Bot — market-making + directional alpha on all 8 products.

Strategy
--------
* Maintain fair-value (FV) estimates for every product, refreshed every
  REFRESH_INTERVAL seconds from real data (tides, weather, flights).
* On each SSE orderbook event, either:
    1. TAKE  — send an IOC order if the market is > TAKE_THRESHOLD off our FV.
    2. QUOTE — cancel-and-replace bid/ask around FV (only when mid shifts
               >= REQUOTE_THRESHOLD ticks, to preserve queue priority).
* Position risk: stop quoting into a side once |position| >= MAX_POSITION.

Fair-value sources
------------------
* TIDE_SPOT / TIDE_SWING : tide_predictor.py  (M2+K1+M4+MK3 harmonic OLS)
* WX_SPOT   / WX_SUM     : Open-Meteo 15-min weather forecast (free, no key)
* LHR_COUNT / LHR_INDEX  : AeroDataBox via RapidAPI (optional; falls back to
                            starting price if RAPIDAPI_KEY is blank)
* LON_ETF                : TIDE_SPOT + WX_SPOT + LHR_COUNT  (derived)
* LON_FLY                : E[fly payoff(ETF)] via Monte Carlo

Credentials
-----------
Set via environment variables (recommended) or edit the CONFIG block below:
    CMI_URL   — exchange base URL
    CMI_USER  — your username
    CMI_PASS  — your password
    RAPIDAPI_KEY — AeroDataBox key (optional; leave blank to skip LHR products)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_TMPL = os.path.join(_SRC, "..", "algothon-templates")
for _p in (_SRC, _TMPL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bot_template import (           # type: ignore[import]
    BaseBot, OrderBook, Order, OrderRequest, Trade, Side, Product)
from tide_predictor import (
    fetch_thames_data, fit_tidal_model,
    predict_tide_spot, predict_tide_swing,
    SESSION_START, SESSION_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit here or set env vars
# ─────────────────────────────────────────────────────────────────────────────
EXCHANGE_URL   = os.getenv("CMI_URL",       "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/")
USERNAME       = os.getenv("CMI_USER",      "REPLACE_WITH_USERNAME")
PASSWORD       = os.getenv("CMI_PASS",      "REPLACE_WITH_PASSWORD")
RAPIDAPI_KEY   = os.getenv("RAPIDAPI_KEY",  "")   # leave blank to skip LHR data

# ── Trading parameters ────────────────────────────────────────────────────────
REFRESH_INTERVAL   = 60     # seconds between full FV refreshes
QUOTE_VOLUME       = 3      # contracts per quote side
MAX_POSITION       = 12     # absolute per-product position cap
MIN_QUOTE_TICKS    = 3      # minimum bid-ask half-spread (ticks)
QUOTE_WIDTH_FRAC   = 6.0    # PI_half / this = quote half-width (higher → tighter)
TAKE_THRESHOLD     = 15     # ticks edge required to send an IOC taker order
REQUOTE_THRESHOLD  = 1      # only refresh quotes when mid moves ≥ this many ticks
N_MC               = 4_000  # MC samples used for ETF-derived products

# ─────────────────────────────────────────────────────────────────────────────
# Fair-value container
# ─────────────────────────────────────────────────────────────────────────────

class FV(NamedTuple):
    """Fair-value estimate for a single product."""
    mean:  float
    lower: float   # 95 % CI lower
    upper: float   # 95 % CI upper
    tick:  float = 1.0

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def quote_half(self) -> float:
        """Half-spread to use when quoting."""
        return max(self.half_width / QUOTE_WIDTH_FRAC, self.tick * MIN_QUOTE_TICKS)

    def bid_px(self) -> float:
        return math.floor((self.mean - self.quote_half()) / self.tick) * self.tick

    def ask_px(self) -> float:
        return math.ceil((self.mean + self.quote_half()) / self.tick) * self.tick


# ─────────────────────────────────────────────────────────────────────────────
# Weather fair-value helpers  (WX_SPOT, WX_SUM)
# ─────────────────────────────────────────────────────────────────────────────
_LONDON_LAT, _LONDON_LON = 51.5074, -0.1278
_SETTLE_TIME = SESSION_END   # 12:00 UTC = 12:00 London (UTC March 1)


def _fetch_weather() -> pd.DataFrame:
    """15-min weather for London from Open-Meteo. Returns temp_C and humidity."""
    variables = "temperature_2m,relative_humidity_2m"
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":           _LONDON_LAT,
            "longitude":          _LONDON_LON,
            "minutely_15":        variables,
            "past_minutely_15":   97,
            "forecast_minutely_15": 97,
            "timezone":           "Europe/London",
        },
        timeout=15,
    )
    resp.raise_for_status()
    m = resp.json()["minutely_15"]
    df = pd.DataFrame({
        "time":        pd.to_datetime(m["time"]).tz_localize("Europe/London"),
        "temp_c":      m["temperature_2m"],
        "humidity":    m["relative_humidity_2m"],
    })
    df["temp_f"] = df["temp_c"] * 9.0 / 5.0 + 32.0
    df["wx"]     = df["temp_f"] * df["humidity"]
    return df


def _wx_fair_values(df_weather: pd.DataFrame) -> tuple[FV, FV]:
    """Compute FV for WX_SPOT and WX_SUM from Open-Meteo data."""
    # Align session window: 12:00 London Feb 28 → 12:00 London Mar 1
    sess_start_local = SESSION_START.astimezone(tz=None)  # convert to local
    sess_end_local   = SESSION_END.astimezone(tz=None)

    # Cast to a timezone-aware London time for comparison
    try:
        import zoneinfo
        london = zoneinfo.ZoneInfo("Europe/London")
    except Exception:
        london = None

    # Convert session boundaries to pandas Timestamps we can compare
    t_start = pd.Timestamp(SESSION_START).tz_convert("Europe/London")
    t_end   = pd.Timestamp(SESSION_END).tz_convert("Europe/London")

    # Rows in the 24h session window (use as many real observations as available;
    # fill forward with forecast for future rows)
    session_df = df_weather[
        (df_weather["time"] >= t_start) & (df_weather["time"] <= t_end)
    ].copy()

    # ── WX_SPOT: temp_F × humidity at settlement time ──────────────────────
    # Find the row closest to SESSION_END
    if len(session_df):
        row_spot = session_df.iloc[-1]  # latest available (≈ settlement)
        wx_spot_mean  = float(row_spot["wx"])
        # Estimate uncertainty: std of last-4 observations or 10% rel
        if len(session_df) >= 4:
            wx_std = float(session_df["wx"].tail(4).std())
        else:
            wx_std = wx_spot_mean * 0.10
        wx_std = max(wx_std, 1.0)
    else:
        wx_spot_mean = 3000.0   # fallback reasonable value
        wx_std       = 300.0

    fv_wx_spot = FV(
        mean=round(wx_spot_mean),
        lower=round(wx_spot_mean - 1.96 * wx_std),
        upper=round(wx_spot_mean + 1.96 * wx_std),
    )

    # ── WX_SUM: sum(temp_F × humidity / 100) over 96 15-min intervals ─────
    if len(session_df) >= 2:
        diffs = session_df["wx"].diff().dropna()
        # Payoff uses value (not diff) — reread spec:
        # WX_SUM = sum(temp_F × humidity_%) / 100 per interval
        session_vals = session_df["wx"].values
        wx_sum_mean  = float(session_vals.sum() / 100.0)
        # Conservative uncertainty: ±5 % of sum
        wx_sum_std   = wx_sum_mean * 0.05
        wx_sum_std   = max(wx_sum_std, 5.0)
    else:
        wx_sum_mean = 3000.0
        wx_sum_std  = 300.0

    fv_wx_sum = FV(
        mean=round(wx_sum_mean),
        lower=round(wx_sum_mean - 1.96 * wx_sum_std),
        upper=round(wx_sum_mean + 1.96 * wx_sum_std),
    )
    return fv_wx_spot, fv_wx_sum


# ─────────────────────────────────────────────────────────────────────────────
# Flights fair-value helpers  (LHR_COUNT, LHR_INDEX)
# ─────────────────────────────────────────────────────────────────────────────
_AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
_AIRPORT = "LHR"


def _fetch_flights_half(from_local: str, to_local: str) -> dict:
    url = (
        f"https://{_AERODATABOX_HOST}/flights/airports/iata/"
        f"{_AIRPORT}/{from_local}/{to_local}?direction=Both"
    )
    resp = requests.get(
        url,
        headers={
            "x-rapidapi-host": _AERODATABOX_HOST,
            "x-rapidapi-key":  RAPIDAPI_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _lhr_fair_values(starting_prices: dict[str, int]) -> tuple[FV, FV]:
    """Return FV for LHR_COUNT and LHR_INDEX.  Uses AeroDataBox if key present."""
    if not RAPIDAPI_KEY:
        # No API key: use starting price with wide CI
        def _wide(sym: str) -> FV:
            m = starting_prices.get(sym, 1000)
            return FV(mean=m, lower=round(m * 0.7), upper=round(m * 1.3))
        return _wide("LHR_COUNT"), _wide("LHR_INDEX")

    try:
        # Fetch both 12h halves of the session
        data1 = _fetch_flights_half("2026-02-28T12:00", "2026-03-01T00:00")
        time.sleep(1)
        data2 = _fetch_flights_half("2026-03-01T00:00", "2026-03-01T12:00")

        arr1 = len(data1.get("arrivals",   []))
        dep1 = len(data1.get("departures", []))
        arr2 = len(data2.get("arrivals",   []))
        dep2 = len(data2.get("departures", []))

        lhr_count = arr1 + dep1 + arr2 + dep2
        lhr_count_std = max(lhr_count * 0.05, 10)

        # LHR_INDEX: |Σ (arr-dep)/(arr+dep)| × 100 per 30-min interval
        # We only have totals for 12h blocks so treat each as one interval
        def _imb(a: int, d: int) -> float:
            return (a - d) / (a + d) if (a + d) > 0 else 0.0
        lhr_index = abs(_imb(arr1, dep1) + _imb(arr2, dep2)) * 100
        lhr_index_std = max(lhr_index * 0.15, 3.0)

        return (
            FV(round(lhr_count),
               round(lhr_count - 1.96 * lhr_count_std),
               round(lhr_count + 1.96 * lhr_count_std)),
            FV(round(lhr_index),
               round(lhr_index - 1.96 * lhr_index_std),
               round(lhr_index + 1.96 * lhr_index_std)),
        )
    except Exception as exc:
        print(f"[fv] LHR fetch failed ({exc}); using starting prices")
        def _wp(sym: str) -> FV:
            m = starting_prices.get(sym, 1000)
            return FV(mean=m, lower=round(m * 0.7), upper=round(m * 1.3))
        return _wp("LHR_COUNT"), _wp("LHR_INDEX")


# ─────────────────────────────────────────────────────────────────────────────
# LON_ETF and LON_FLY fair values (derived)
# ─────────────────────────────────────────────────────────────────────────────

def _lon_fly_payoff(etf: np.ndarray) -> np.ndarray:
    """LON_FLY = 2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)."""
    p6200 = np.maximum(0.0, 6200 - etf)
    c6200 = np.maximum(0.0, etf - 6200)
    c6600 = np.maximum(0.0, etf - 6600)
    c7000 = np.maximum(0.0, etf - 7000)
    return 2 * p6200 + c6200 - 2 * c6600 + 3 * c7000


def _derived_fvs(
    fv_tide: FV, fv_wx_spot: FV, fv_lhr_count: FV,
    rng: np.random.Generator,
) -> tuple[FV, FV]:
    """Compute LON_ETF and LON_FLY from component FVs via Monte Carlo."""
    sigma_tide = fv_tide.half_width     / 1.96
    sigma_wx   = fv_wx_spot.half_width  / 1.96
    sigma_lhr  = fv_lhr_count.half_width / 1.96

    samples_tide = rng.normal(fv_tide.mean,      sigma_tide,  N_MC)
    samples_wx   = rng.normal(fv_wx_spot.mean,   sigma_wx,    N_MC)
    samples_lhr  = rng.normal(fv_lhr_count.mean, sigma_lhr,   N_MC)

    etf_samples  = samples_tide + samples_wx + samples_lhr
    fly_samples  = _lon_fly_payoff(etf_samples)

    p025, p975 = np.percentile(etf_samples, [2.5, 97.5])
    fv_etf = FV(mean=round(float(etf_samples.mean())),
                lower=round(float(p025)), upper=round(float(p975)))

    fp025, fp975 = np.percentile(fly_samples, [2.5, 97.5])
    fv_fly = FV(mean=round(float(fly_samples.mean())),
                lower=round(float(fp025)), upper=round(float(fp975)))

    return fv_etf, fv_fly


# ─────────────────────────────────────────────────────────────────────────────
# Full fair-value engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FairValues:
    """Snapshot of FVs for all 8 products."""
    TIDE_SPOT:  FV
    TIDE_SWING: FV
    WX_SPOT:    FV
    WX_SUM:     FV
    LHR_COUNT:  FV
    LHR_INDEX:  FV
    LON_ETF:    FV
    LON_FLY:    FV
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def get(self, symbol: str) -> FV | None:
        return getattr(self, symbol, None)

    def summary(self) -> str:
        lines = [f"[FV snapshot @ {self.fetched_at.strftime('%H:%M:%S')} UTC]"]
        for sym in ("TIDE_SPOT","TIDE_SWING","WX_SPOT","WX_SUM",
                    "LHR_COUNT","LHR_INDEX","LON_ETF","LON_FLY"):
            fv = self.get(sym)
            if fv:
                lines.append(f"  {sym:<12}  mean={fv.mean:>6}  "
                             f"CI=[{fv.lower}, {fv.upper}]  "
                             f"q=[{fv.bid_px():.0f}, {fv.ask_px():.0f}]")
        return "\n".join(lines)


def refresh_fair_values(
    starting_prices: dict[str, int],
    rng: np.random.Generator,
    verbose: bool = True,
) -> FairValues:
    """Fetch all data sources and compute FVs for all 8 products."""
    now = datetime.now(tz=timezone.utc)

    # ── Tidal products ──────────────────────────────────────────────────────
    try:
        df_thames = fetch_thames_data(limit=500)
        model     = fit_tidal_model(df_thames)
        spot_pred = predict_tide_spot(df_thames, model)
        swing_pred = predict_tide_swing(df_thames, model, now=now, rng=rng)

        fv_tide_spot = FV(
            mean  = spot_pred.mean,
            lower = spot_pred.lower,
            upper = spot_pred.upper,
        )
        fv_tide_swing = FV(
            mean  = swing_pred.mean,
            lower = swing_pred.lower,
            upper = swing_pred.upper,
        )
    except Exception as exc:
        print(f"[fv] Tide fetch failed ({exc}); using starting prices")
        def _tide(sym: str) -> FV:
            m = starting_prices.get(sym, 2000)
            return FV(m, round(m * 0.7), round(m * 1.3))
        fv_tide_spot  = _tide("TIDE_SPOT")
        fv_tide_swing = _tide("TIDE_SWING")

    # ── Weather products ───────────────────────────────────────────────────
    try:
        df_weather    = _fetch_weather()
        fv_wx_spot, fv_wx_sum = _wx_fair_values(df_weather)
    except Exception as exc:
        print(f"[fv] Weather fetch failed ({exc}); using starting prices")
        def _wx(sym: str) -> FV:
            m = starting_prices.get(sym, 3000)
            return FV(m, round(m * 0.7), round(m * 1.3))
        fv_wx_spot = _wx("WX_SPOT")
        fv_wx_sum  = _wx("WX_SUM")

    # ── Flights products ───────────────────────────────────────────────────
    fv_lhr_count, fv_lhr_index = _lhr_fair_values(starting_prices)

    # ── Derived products ───────────────────────────────────────────────────
    fv_etf, fv_fly = _derived_fvs(fv_tide_spot, fv_wx_spot, fv_lhr_count, rng)

    fvs = FairValues(
        TIDE_SPOT  = fv_tide_spot,
        TIDE_SWING = fv_tide_swing,
        WX_SPOT    = fv_wx_spot,
        WX_SUM     = fv_wx_sum,
        LHR_COUNT  = fv_lhr_count,
        LHR_INDEX  = fv_lhr_index,
        LON_ETF    = fv_etf,
        LON_FLY    = fv_fly,
        fetched_at = now,
    )

    if verbose:
        print(fvs.summary())

    return fvs


# ─────────────────────────────────────────────────────────────────────────────
# Trading bot
# ─────────────────────────────────────────────────────────────────────────────

class IMCityBot(BaseBot):
    """
    Market-making + directional-alpha bot for IMCity.

    Lifecycle
    ---------
    1. On start: fetch products, do first FV refresh, start SSE stream, start
       background refresh thread.
    2. Background thread: refresh FVs every REFRESH_INTERVAL seconds.
       After each refresh, cancel-and-replace all quotes.
    3. SSE on_orderbook: for each incoming orderbook event:
       a. IOC-take if edge > TAKE_THRESHOLD ticks.
       b. Requote if market mid has drifted >= REQUOTE_THRESHOLD ticks since
          the last time we placed quotes for this symbol.
    4. on_trades: log fills; update position tracking.
    """

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self._rng            = np.random.default_rng(42)
        self._fvs: FairValues | None = None
        self._fvs_lock       = threading.Lock()

        # {symbol: last mid-price we placed quotes at}
        self._last_quoted_mid: dict[str, float] = {}
        # {symbol: (bid_order_id, ask_order_id)}  — active quotes
        self._active_quotes:  dict[str, tuple[str, str]] = {}
        self._quotes_lock     = threading.Lock()

        self._products: dict[str, Product] = {}
        self._positions: dict[str, int]    = {}
        self._pos_lock   = threading.Lock()

        self._stop_event  = threading.Event()
        self._refresh_thread: threading.Thread | None = None

        # Per-symbol lock: prevents concurrent quote-placement for the same product
        self._sym_locks: dict[str, threading.Lock] = {}
        # Per-symbol last-quoting-time (monotonic): rate-limiting guard
        self._last_quote_time: dict[str, float] = {}
        _QUOTE_COOLDOWN = 2.0   # seconds between quotes for the same product
        self._quote_cooldown = _QUOTE_COOLDOWN

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking entry-point: fetch products → FV → start loops."""
        print(f"[bot] Connecting as '{self.username}' → {self._cmi_url}")
        self._products = {p.symbol: p for p in self.get_products()}
        print(f"[bot] Found {len(self._products)} products: {list(self._products)}")

        starting_prices = {sym: p.startingPrice for sym, p in self._products.items()}

        # Initial FV refresh (blocking)
        with self._fvs_lock:
            self._fvs = refresh_fair_values(starting_prices, self._rng, verbose=True)

        # Sync positions
        self._sync_positions()

        # Start SSE stream (calls on_orderbook / on_trades in bg thread)
        self.start()

        # Background FV-refresh thread
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            args=(starting_prices,),
            daemon=True,
            name="fv-refresh",
        )
        self._refresh_thread.start()

        print("[bot] Running. Ctrl+C to stop.")
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        print("[bot] Shutting down …")
        self._stop_event.set()
        self.cancel_all_orders()
        self.stop()
        print("[bot] Stopped. Final P&L:")
        try:
            pnl = self.get_pnl()
            print(f"  {json.dumps(pnl, indent=2)}")
        except Exception:
            pass

    # ── Background FV-refresh loop ──────────────────────────────────────────

    def _refresh_loop(self, starting_prices: dict[str, int]) -> None:
        while not self._stop_event.is_set():
            time.sleep(REFRESH_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                new_fvs = refresh_fair_values(starting_prices, self._rng, verbose=True)
                with self._fvs_lock:
                    self._fvs = new_fvs
                self._sync_positions()
                # Requote all products after FV update
                self._requote_all()
            except Exception as exc:
                print(f"[bot] FV refresh error: {exc}")

    def _sync_positions(self) -> None:
        try:
            raw = self.get_positions()
            with self._pos_lock:
                self._positions = dict(raw)
        except Exception as exc:
            print(f"[bot] Position sync error: {exc}")

    def _requote_all(self) -> None:
        """Cancel-and-replace quotes for every product after FV refresh."""
        with self._fvs_lock:
            fvs = self._fvs
        if fvs is None:
            return
        for sym in self._products:
            fv = fvs.get(sym)
            if fv is None:
                continue
            try:
                ob = self.get_orderbook(sym)
                self._place_quotes(sym, fv, ob, force=True)
                time.sleep(0.2)   # stay under 1 req/sec rule
            except Exception as exc:
                print(f"[bot] Requote {sym}: {exc}")

    # ── SSE callbacks ────────────────────────────────────────────────────────

    def on_orderbook(self, ob: OrderBook) -> None:
        sym = ob.product
        with self._fvs_lock:
            fvs = self._fvs
        if fvs is None:
            return
        fv = fvs.get(sym)
        if fv is None:
            return

        # Market mid (excluding our own orders)
        bids = [o.price for o in ob.buy_orders  if o.volume - o.own_volume > 0]
        asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
        if not bids or not asks:
            return

        best_bid = max(bids)
        best_ask = min(asks)
        mid      = (best_bid + best_ask) / 2.0
        tick     = ob.tick_size or 1.0

        # ── 1. Directional IOC (take edge) ───────────────────────────────
        pos = self._get_position(sym)
        edge_buy  = fv.mean - best_ask          # positive → ask is cheap
        edge_sell = best_bid - fv.mean          # positive → bid is expensive

        if edge_buy >= TAKE_THRESHOLD and pos < MAX_POSITION:
            vol = min(QUOTE_VOLUME, MAX_POSITION - pos)
            self._send_ioc(sym, best_ask, Side.BUY, vol)
            return   # skip regular quoting this tick

        if edge_sell >= TAKE_THRESHOLD and pos > -MAX_POSITION:
            vol = min(QUOTE_VOLUME, MAX_POSITION + pos)
            self._send_ioc(sym, best_bid, Side.SELL, vol)
            return

        # ── 2. Passive quote (only if mid has moved enough) ──────────────
        last_mid = self._last_quoted_mid.get(sym)
        if last_mid is not None and abs(mid - last_mid) < REQUOTE_THRESHOLD * tick:
            return   # mid hasn't moved; preserve queue position

        self._place_quotes(sym, fv, ob, force=False)

    def on_trades(self, trade: Trade) -> None:
        side = "BOT" if trade.buyer == self.username else "SOLD"
        print(f"[fill] {trade.product:12} {side} {trade.volume}@{trade.price:.0f}  "
              f"ts={trade.timestamp}")
        # Sync positions after fill
        self._sync_positions()

    # ── Quoting helpers ──────────────────────────────────────────────────────

    def _place_quotes(self, sym: str, fv: FV, ob: OrderBook,
                       force: bool = False) -> None:
        """Cancel old quotes for sym, place fresh bid/ask around FV.

        Thread-safe: each symbol gets its own lock.  The `force` flag
        bypasses the per-symbol cooldown (used after a full FV refresh).
        """
        lock = self._sym_locks.setdefault(sym, threading.Lock())
        if not lock.acquire(blocking=False):
            return   # another thread is already quoting this symbol
        try:
            # Rate-limit: don't hammer the exchange
            now_mono = time.monotonic()
            if not force and (
                now_mono - self._last_quote_time.get(sym, 0.0) < self._quote_cooldown
            ):
                return
            self._last_quote_time[sym] = now_mono
            self._place_quotes_inner(sym, fv, ob)
        finally:
            lock.release()

    def _place_quotes_inner(self, sym: str, fv: FV, ob: OrderBook) -> None:
        """Internal: assume sym-lock held."""
        tick = ob.tick_size or 1.0
        pos  = self._get_position(sym)

        bid_px = fv.bid_px()
        ask_px = fv.ask_px()

        # Safety: never quote negative price
        bid_px = max(bid_px, tick)
        # Crossed-quote guard
        if bid_px >= ask_px:
            ask_px = bid_px + tick

        # Cancel any existing quotes
        self._cancel_quotes(sym)

        orders: list[OrderRequest] = []

        # Only quote bid if not at long limit
        if pos < MAX_POSITION:
            orders.append(OrderRequest(sym, bid_px, Side.BUY, QUOTE_VOLUME))

        # Only quote ask if not at short limit
        if pos > -MAX_POSITION:
            orders.append(OrderRequest(sym, ask_px, Side.SELL, QUOTE_VOLUME))

        if not orders:
            return

        responses = self.send_orders(orders)

        # Track the new order IDs so we can cancel them next time
        bid_id = ask_id = None
        for resp in responses:
            if resp and resp.side == Side.BUY:
                bid_id = resp.id
            elif resp and resp.side == Side.SELL:
                ask_id = resp.id

        if bid_id or ask_id:
            with self._quotes_lock:
                self._active_quotes[sym] = (bid_id, ask_id)
            # Record the mid when we quoted
            bids = [o.price for o in ob.buy_orders  if o.volume - o.own_volume > 0]
            asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
            if bids and asks:
                self._last_quoted_mid[sym] = (max(bids) + min(asks)) / 2.0

    def _cancel_quotes(self, sym: str) -> None:
        with self._quotes_lock:
            ids = self._active_quotes.pop(sym, (None, None))
        for oid in ids:
            if oid:
                try:
                    self.cancel_order(oid)
                except Exception:
                    pass

    def _send_ioc(self, sym: str, price: float, side: Side, volume: int) -> None:
        """Send an order and immediately cancel any unfilled remainder (IOC)."""
        resp = self.send_order(OrderRequest(sym, price, side, volume))
        if resp and resp.volume > resp.filled:
            try:
                self.cancel_order(resp.id)
            except Exception:
                pass
        if resp:
            print(f"[ioc]  {sym:12} {side} {resp.filled}/{volume}@{price:.0f}")

    def _get_position(self, sym: str) -> int:
        with self._pos_lock:
            return self._positions.get(sym, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "REPLACE_WITH" in USERNAME or "REPLACE_WITH" in PASSWORD:
        print(
            "ERROR: Set your credentials first.\n"
            "  Option A: environment variables  CMI_USER  CMI_PASS\n"
            "  Option B: edit EXCHANGE_URL / USERNAME / PASSWORD in src/bot.py"
        )
        sys.exit(1)

    bot = IMCityBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run()
