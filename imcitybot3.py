"""
IMCity London Challenge — Signal & Volatility Trading Bot
==========================================================
NO market making. Four focused strategies + passive traps:

  1. Alpha Signals — Use live data (tides, weather) to estimate fair values.
     When the market price diverges meaningfully from our estimate, take a
     directional position aggressively via IOC.

  2. ETF & Option Arbitrage — Structural mispricings between LON_ETF and
     its components, and LON_FLY vs its analytical value.

  3. Volatility Fade — Track rolling price ranges per product. When a
     price spikes far from its recent range (panic buying/selling by other
     teams' bots), fade the move. Uses a cooldown to avoid fading real
     trends. This exploits the fact that 40-50 student teams will
     inevitably cause short-lived dislocations.

  4. Trap Orders — Maintain small resting orders at extreme prices where
     fair value can never settle. If a malfunctioning bot market-orders
     into these, it's free profit. Checked/refreshed every 60 seconds.
"""

import json
import math
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from bot_template import (
    BaseBot, OrderBook, Order, OrderRequest, OrderResponse,
    Trade, Side, Product,
)

# ─────────────────────── Configuration ───────────────────────

LONDON_LAT, LONDON_LON = 51.5074, -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

# Product symbols
TIDE_SPOT   = "TIDE_SPOT"
TIDE_SWING  = "TIDE_SWING"
WX_SPOT     = "WX_SPOT"
WX_SUM      = "WX_SUM"
LHR_COUNT   = "LHR_COUNT"
LHR_INDEX   = "LHR_INDEX"
LON_ETF     = "LON_ETF"
LON_FLY     = "LON_FLY"

ALL_PRODUCTS = [TIDE_SPOT, TIDE_SWING, WX_SPOT, WX_SUM, LHR_COUNT, LHR_INDEX, LON_ETF, LON_FLY]
ETF_COMPONENTS = [TIDE_SPOT, WX_SPOT, LHR_COUNT]

# Option structure: +2 P6200, +1 C6200, -2 C6600, +3 C7000
OPT_STRIKES = [(2, "P", 6200), (1, "C", 6200), (-2, "C", 6600), (3, "C", 7000)]

# ── Risk ──
MAX_POSITION = 85           # Hard ceiling per product (exchange limit is +/-100)
INVENTORY_WARN = 65
INVENTORY_CRITICAL = 75
SKEW_NORMAL = 2.5    # Price skew per contract (normal regime)
SKEW_WARN = 6.0      # Price skew per contract (warn regime)
SKEW_CRITICAL = 15.0 # Price skew per contract (critical regime)
POSITION_PER_SIGNAL = 5     # Base size for a single signal trade
POSITION_PER_ARB = 2        # Size for arb legs

# ── Signal thresholds (how far market must deviate from FV to trade) ──
SIGNAL_ENTRY_FRAC = 0.015   # 1.5% edge to enter
SIGNAL_EXIT_FRAC  = 0.003   # 0.3% — close when edge shrinks

# ── Volatility fade ──
VOL_WINDOW = 40             # Number of mid-price observations for rolling stats
VOL_ZSCORE_ENTRY = 2.5      # Z-score to trigger a fade
VOL_ZSCORE_EXIT  = 0.8      # Z-score to take profit
VOL_COOLDOWN_S = 30         # Seconds between fade trades on same product
VOL_TRADE_SIZE = 3

# ── ETF Arb ──
ETF_ARB_THRESHOLD = 10      # Minimum mispricing in ticks to act
OPT_ARB_THRESHOLD = 20      # Options are wider, need more edge

# ── Trap orders ("roach motel") ──
# Resting orders at absurd prices that only fill if another bot malfunctions.
# Bounds are chosen so that the fair value can NEVER end up there.
#
# Reasoning per product:
#   TIDE_SPOT  = abs(level_mAOD)*1000.  Thames range: -3 to +5 mAOD → 0-5000.
#                Buy@1 is safe (tide is never 0.001m). Sell@6000 is safe (never 6m).
#   TIDE_SWING = sum of strangle payoffs*100.  Always ≥0.  Realistic 100-2500.
#                Buy@1 safe.  Sell@5000 safe (would need every interval to be extreme).
#   WX_SPOT    = round(temp_F)*humidity.  London March: 30-60°F, hum 30-100%.
#                Range ~900-6000.  Buy@50 safe. Sell@10000 safe.
#   WX_SUM     = cumulative (T*H/100) over 97 intervals.  Each ~9-60.
#                Range ~900-5800.  Buy@50 safe. Sell@10000 safe.
#   LHR_COUNT  = total flights 24h.  Sunday Heathrow: 800-1500.
#                Buy@100 safe. Sell@2500 safe.
#   LHR_INDEX  = abs(sum of 30min imbalance metrics).  48 bins, each ±100.
#                Realistic 0-200.  Buy@1 safe. Sell@600 safe.
#   LON_ETF    = TIDE_SPOT + WX_SPOT + LHR_COUNT.  Sum range ~1700-12500.
#                Buy@200 safe. Sell@16000 safe.
#   LON_FLY    = option structure on ETF.  Range 0-~6500 for realistic ETF.
#                Buy@1 safe. Sell@12000 safe (would need ETF near 0 or >9000).
TRAP_ORDERS: dict[str, tuple[float, float]] = {
    # product:  (trap_buy_price, trap_sell_price)
    TIDE_SPOT:  (1,     6000),
    TIDE_SWING: (1,     5000),
    WX_SPOT:    (50,    10000),
    WX_SUM:     (50,    10000),
    LHR_COUNT:  (100,   2500),
    LHR_INDEX:  (1,     600),
    LON_ETF:    (200,   16000),
    LON_FLY:    (1,     12000),
}
TRAP_VOLUME = 2                 # Small: just enough to profit, not enough to hurt
TRAP_REFRESH_INTERVAL = 60      # Seconds between checking trap orders are alive

# ── Timing ──
DATA_REFRESH_INTERVAL = 120     # Seconds between API calls
FV_RECOMPUTE_INTERVAL = 20     # Seconds between FV recomputation
LOOP_INTERVAL = 2.0            # Main loop sleep


# ─────────────────────── Data Fetching ───────────────────────
@dataclass
class BookSnap:
    """
    Stores the FULL order book depth, not just the top level.
    This is essential for WAP calculation and slippage estimation.
    [C1, C5 fix]
    """
    product:    str
    bids:       list[Order] = field(default_factory=list)  # sorted high→low
    asks:       list[Order] = field(default_factory=list)  # sorted low→high

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def bid_volume(self) -> int:
        return self.bids[0].volume if self.bids else 0

    @property
    def ask_volume(self) -> int:
        return self.asks[0].volume if self.asks else 0

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def total_bid_depth(self) -> int:
        return sum(o.volume for o in self.bids)

    @property
    def total_ask_depth(self) -> int:
        return sum(o.volume for o in self.asks)

    def wap_buy(self, volume: int) -> Optional[float]:
        """
        Weighted average price to BUY `volume` contracts by walking the ask side.
        Returns None if insufficient depth exists.
        [C5 fix]
        """
        return _wap(self.asks, volume)

    def wap_sell(self, volume: int) -> Optional[float]:
        """
        Weighted average price to SELL `volume` contracts by walking the bid side.
        Returns None if insufficient depth exists.
        [C5 fix]
        """
        return _wap(self.bids, volume)


def _wap(levels: list[Order], volume: int) -> Optional[float]:
    """Walk order book levels to compute weighted average fill price."""
    if not levels:
        return None
    remaining = volume
    total_cost = 0.0
    for level in levels:
        fill = min(remaining, level.volume)
        total_cost += fill * level.price
        remaining -= fill
        if remaining <= 0:
            break
    if remaining > 0:
        return None  # insufficient depth
    return total_cost / volume


def _snap(ob: OrderBook) -> BookSnap:
    """Build a full-depth BookSnap from an OrderBook event. [C1, C5 fix]"""
    return BookSnap(
        product=ob.product,
        bids=list(ob.buy_orders),   # already sorted high→low by framework
        asks=list(ob.sell_orders),  # already sorted low→high by framework
    )

def fetch_weather(past_steps=192, forecast_steps=96) -> Optional[pd.DataFrame]:
    """15-min weather for London."""
    try:
        variables = ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                     "precipitation,wind_speed_10m,cloud_cover,visibility")
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LONDON_LAT, "longitude": LONDON_LON,
            "minutely_15": variables,
            "past_minutely_15": past_steps,
            "forecast_minutely_15": forecast_steps,
            "timezone": "Europe/London",
        }, timeout=10)
        resp.raise_for_status()
        m = resp.json()["minutely_15"]
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temperature_c": m["temperature_2m"],
            "humidity": m["relative_humidity_2m"],
        })
        df["temperature_f"] = df["temperature_c"] * 9 / 5 + 32
        df["temp_f_rounded"] = df["temperature_f"].round()
        df["t_x_h"] = df["temp_f_rounded"] * df["humidity"]
        return df
    except Exception as e:
        print(f"[DATA] Weather fetch failed: {e}")
        return None


def fetch_thames(limit=300) -> Optional[pd.DataFrame]:
    """Fetch Thames tidal readings at Westminster."""
    try:
        resp = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/"
            f"{THAMES_MEASURE}/readings",
            params={"_sorted": "", "_limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        df = pd.DataFrame(items)[["dateTime", "value"]].rename(
            columns={"dateTime": "time", "value": "level"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"[DATA] Thames fetch failed: {e}")
        return None


# ─────────────────────── Fair Value Engine ───────────────────────

def compute_option_payoff(etf_value: float) -> float:
    """LON_FLY settlement: +2 P6200, +1 C6200, -2 C6600, +3 C7000."""
    s = etf_value
    payoff = 0.0
    for qty, opt_type, strike in OPT_STRIKES:
        if opt_type == "P":
            payoff += qty * max(0, strike - s)
        else:
            payoff += qty * max(0, s - strike)
    return payoff


def strangle_payoff(diff_cm: float, k_put=20.0, k_call=25.0) -> float:
    """Strangle payoff on a single 15-min difference."""
    return max(0, k_put - diff_cm) + max(0, diff_cm - k_call)


@dataclass
class FairValues:
    values: dict        # product -> estimated fair value
    confidence: dict    # product -> 0.0 to 1.0
    timestamp: float


class FairValueEngine:
    """Computes settlement-anchored fair values from live data."""

    def __init__(self):
        self.weather_df: Optional[pd.DataFrame] = None
        self.thames_df: Optional[pd.DataFrame] = None
        self.weather_ts: float = 0
        self.thames_ts: float = 0
        self._session_start: Optional[pd.Timestamp] = None
        self._session_end: Optional[pd.Timestamp] = None

    def _compute_session_window(self):
        """Saturday 12pm -> Sunday 12pm London time."""
        now = pd.Timestamp.now(tz="Europe/London")
        days_back = (now.weekday() - 5) % 7
        sat = (now - pd.Timedelta(days=days_back)).normalize() + pd.Timedelta(hours=12)
        if sat > now:
            sat -= pd.Timedelta(weeks=1)
        self._session_start = sat
        self._session_end = sat + pd.Timedelta(hours=24)

    def refresh_data(self):
        now = time.monotonic()
        if now - self.weather_ts > DATA_REFRESH_INTERVAL:
            df = fetch_weather()
            if df is not None:
                self.weather_df = df
                self.weather_ts = now
                print(f"[DATA] Weather refreshed: {len(df)} rows")

        if now - self.thames_ts > DATA_REFRESH_INTERVAL:
            df = fetch_thames()
            if df is not None:
                self.thames_df = df
                self.thames_ts = now
                print(f"[DATA] Thames refreshed: {len(df)} rows")

        if self._session_start is None:
            self._compute_session_window()

    def _hours_to_settlement(self) -> float:
        if self._session_end is None:
            return 12.0
        now = pd.Timestamp.now(tz="Europe/London")
        return max(0, (self._session_end - now).total_seconds() / 3600)

    def compute(self) -> FairValues:
        values, confidence = {}, {}
        h2s = self._hours_to_settlement()

        v, c = self._tide_spot(h2s);   values[TIDE_SPOT]  = v; confidence[TIDE_SPOT]  = c
        v, c = self._tide_swing();     values[TIDE_SWING] = v; confidence[TIDE_SWING] = c
        v, c = self._wx_spot(h2s);     values[WX_SPOT]    = v; confidence[WX_SPOT]    = c
        v, c = self._wx_sum();         values[WX_SUM]     = v; confidence[WX_SUM]     = c

        # Flight products — no free data, low confidence priors
        values[LHR_COUNT]  = 1150; confidence[LHR_COUNT]  = 0.10
        values[LHR_INDEX]  = 15;   confidence[LHR_INDEX]  = 0.08

        # Derived
        etf = max(0, values[TIDE_SPOT] + values[WX_SPOT] + values[LHR_COUNT])
        etf_c = min(confidence[TIDE_SPOT], confidence[WX_SPOT], confidence[LHR_COUNT])
        values[LON_ETF]  = etf;                       confidence[LON_ETF]  = etf_c
        values[LON_FLY]  = compute_option_payoff(etf); confidence[LON_FLY] = etf_c * 0.7

        return FairValues(values, confidence, time.monotonic())

    # ── Helpers ──

    def _session_filter(self, df: pd.DataFrame, time_col="time") -> pd.DataFrame:
        if self._session_start and self._session_end:
            return df[(df[time_col] >= self._session_start) &
                      (df[time_col] <= self._session_end)]
        return df.tail(96)

    # ── Individual estimators ──

    def _tide_spot(self, h2s: float) -> tuple[float, float]:
        """TIDE_SPOT = abs(level) * 1000 at settlement.
        Uses harmonic extrapolation of tidal cycle (~12.42h period)."""
        if self.thames_df is None or len(self.thames_df) < 5:
            return 1500, 0.05

        df = self.thames_df
        current_mm = abs(df["level"].iloc[-1]) * 1000

        session = self._session_filter(df)
        if len(session) >= 20:
            levels = session["level"].values
            times_h = np.array([
                (t - session["time"].iloc[0]).total_seconds() / 3600
                for t in session["time"]
            ])

            # Harmonic model: level(t) ~ mean + A * sin(2pi/T * t + phi)
            mean_lvl = levels.mean()
            amplitude = (levels.max() - levels.min()) / 2
            T = 12.42  # tidal period hours

            # Phase estimation from latest observation
            latest_t = times_h[-1]
            ratio = np.clip((levels[-1] - mean_lvl) / max(amplitude, 0.01), -1, 1)
            phase = np.arcsin(ratio) - (2 * np.pi / T) * latest_t

            # Also estimate phase from second-to-last to pick the right branch
            # (arcsin has two solutions per cycle)
            if len(levels) >= 2:
                prev_ratio = np.clip((levels[-2] - mean_lvl) / max(amplitude, 0.01), -1, 1)
                dt = times_h[-1] - times_h[-2]
                # If level is rising, we're on the ascending branch
                rising = levels[-1] > levels[-2]
                if rising and ratio > 0.5:
                    # Approaching peak — use arcsin directly
                    pass
                elif not rising and ratio > 0:
                    # Descending from peak — use pi - arcsin
                    phase = (np.pi - np.arcsin(ratio)) - (2 * np.pi / T) * latest_t

            settle_t = latest_t + h2s
            forecast = mean_lvl + amplitude * np.sin(2 * np.pi / T * settle_t + phase)
            forecast_mm = abs(forecast) * 1000

            if h2s < 0.5:
                return round(current_mm), 0.92
            elif h2s < 2:
                return round(0.6 * forecast_mm + 0.4 * current_mm), 0.70
            elif h2s < 6:
                return round(forecast_mm), 0.45
            else:
                return round(forecast_mm), 0.28
        else:
            return round(current_mm), (0.80 if h2s < 0.5 else 0.15)

    def _tide_swing(self) -> tuple[float, float]:
        """TIDE_SWING = sum of strangle payoffs on 15min diffs * 100."""
        if self.thames_df is None or len(self.thames_df) < 10:
            return 500, 0.05

        session = self._session_filter(self.thames_df)
        if len(session) < 2:
            return 500, 0.05

        levels = session["level"].values
        diffs_cm = np.abs(np.diff(levels)) * 100
        payoffs = np.array([strangle_payoff(d) for d in diffs_cm])
        observed_sum = payoffs.sum() * 100

        n_obs = len(diffs_cm)
        n_total = 96

        if n_obs >= n_total * 0.9:
            return round(observed_sum), 0.85
        elif n_obs > 5:
            avg = payoffs.mean()
            projected = observed_sum + avg * (n_total - n_obs) * 100
            return round(projected), min(0.65, 0.15 + 0.55 * n_obs / n_total)
        else:
            return 500, 0.10

    def _wx_spot(self, h2s: float) -> tuple[float, float]:
        """WX_SPOT = round(temp_F) * humidity at settlement."""
        if self.weather_df is None or len(self.weather_df) < 5:
            return 4000, 0.05

        df = self.weather_df

        if self._session_end:
            diffs = (df["time"] - self._session_end).abs()
            idx = diffs.idxmin()
            row = df.loc[idx]
            hours_off = diffs[idx].total_seconds() / 3600
            val = round(row["temp_f_rounded"] * row["humidity"])

            if hours_off < 0.5:
                conf = 0.90
            elif hours_off < 3:
                conf = 0.70
            elif hours_off < 8:
                conf = 0.45
            else:
                conf = 0.20
            return max(0, val), conf
        else:
            latest = df.iloc[-1]
            return max(0, round(latest["t_x_h"])), 0.20

    def _wx_sum(self) -> tuple[float, float]:
        """WX_SUM = sum(temp_F_rounded * humidity / 100) over session intervals."""
        if self.weather_df is None or len(self.weather_df) < 5:
            return 4000, 0.05

        session = self._session_filter(self.weather_df)
        if len(session) < 2:
            return 4000, 0.05

        contribs = session["t_x_h"] / 100
        observed = contribs.sum()
        n_obs = len(session)
        n_total = 97

        if n_obs >= n_total * 0.9:
            return round(observed), 0.80
        elif n_obs > 5:
            avg = contribs.mean()
            projected = observed + avg * (n_total - n_obs)
            return round(projected), min(0.60, 0.10 + 0.55 * n_obs / n_total)
        else:
            return 4000, 0.08


# ─────────────────────── Volatility Tracker ───────────────────────

class VolTracker:
    """Rolling mid-price stats to detect short-term dislocations."""

    def __init__(self, window: int = VOL_WINDOW):
        self.window = window
        self.mids: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_fade_time: dict[str, float] = defaultdict(float)

    def update(self, product: str, mid: float):
        self.mids[product].append(mid)

    def get_zscore(self, product: str) -> Optional[float]:
        q = self.mids[product]
        if len(q) < 12:
            return None
        arr = np.array(q)
        history = arr[:-1]
        mu, sigma = history.mean(), history.std()
        if sigma < 0.5:
            return None
        return (arr[-1] - mu) / sigma

    def get_range(self, product: str) -> Optional[tuple[float, float]]:
        """Return (low, high) of rolling window."""
        q = self.mids[product]
        if len(q) < 12:
            return None
        arr = np.array(q)
        return float(arr.min()), float(arr.max())

    def can_fade(self, product: str) -> bool:
        return (time.monotonic() - self.last_fade_time[product]) > VOL_COOLDOWN_S

    def record_fade(self, product: str):
        self.last_fade_time[product] = time.monotonic()

def _inventory_regime(pos: int) -> str:
    """Return 'normal', 'warn', or 'critical' based on absolute position. [C2 fix]"""
    abs_pos = abs(pos)
    if abs_pos >= INVENTORY_CRITICAL:
        return "critical"
    if abs_pos >= INVENTORY_WARN:
        return "warn"
    return "normal"

# ─────────────────────── Main Bot ───────────────────────

class SignalBot(BaseBot):
    """
    Signal-driven bot. No resting quotes — only IOC-style aggression
    when we have a clear edge.
    """

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)

        self.fv_engine = FairValueEngine()
        self.fv: Optional[FairValues] = None
        self.vol_tracker = VolTracker()

        self._orderbooks: dict[str, OrderBook] = {}
        self._ob_lock = threading.Lock()
        self._positions: dict[str, int] = defaultdict(int)
        self._market_mids: dict[str, float] = {}
        self._market_spreads: dict[str, float] = {}

        self._products: dict[str, Product] = {}
        self._tick: dict[str, float] = {}

        self._last_data_refresh = 0.0
        self._last_fv_compute = 0.0
        self._trade_count = 0

        self._signal_last_time: dict[str, float] = defaultdict(float)
        self._arb_last_time: dict[str, float] = defaultdict(float)

        # Trap order tracking: maps "SYMBOL_BUY" / "SYMBOL_SELL" -> order_id
        self._trap_order_ids: dict[str, str] = {}
        self._last_trap_refresh = 0.0

        self._books: dict[str, BookSnap] = {}
       
        # Tracks resting unwind orders: maps product -> order_id
        self._unwind_orders: dict[str, str] = {} 
        self._last_unwind_time = 0.0

    # ─────────── SSE Callbacks ───────────

    def on_orderbook(self, ob: OrderBook):
        snap = _snap(ob)
        with self._ob_lock:
            self._books[ob.product] = snap
            self._orderbooks[ob.product] = ob
            bids = [o.price for o in ob.buy_orders if o.volume - o.own_volume > 0]
            asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
            if bids and asks:
                best_bid, best_ask = max(bids), min(asks)
                mid = (best_bid + best_ask) / 2
                self._market_mids[ob.product] = mid
                self._market_spreads[ob.product] = best_ask - best_bid
                self.vol_tracker.update(ob.product, mid)

    def on_trades(self, trade: Trade):
        self._trade_count += 1
        if trade.buyer == self.username:
            self._positions[trade.product] += trade.volume
            tag = "BOUGHT"
        else:
            self._positions[trade.product] -= trade.volume
            tag = "SOLD"
        pos = self._positions[trade.product]

        # Check if this was a trap fill (extreme price)
        trap_bounds = TRAP_ORDERS.get(trade.product)
        is_trap = False
        if trap_bounds:
            buy_trap, sell_trap = trap_bounds
            if (tag == "BOUGHT" and trade.price <= buy_trap * 1.5) or \
               (tag == "SOLD" and trade.price >= sell_trap * 0.75):
                is_trap = True

        if is_trap:
            print(f"  [TRAP] ** {tag} {trade.volume}x {trade.product} @ {trade.price} **"
                  f"  pos={pos:+d}  FREE MONEY!")
        else:
            print(f"  [FILL] {tag} {trade.volume}x {trade.product} @ {trade.price}"
                  f"  pos={pos:+d}  (#{self._trade_count})")

    # ─────────── Execution helpers ───────────

    def _ioc(self, symbol: str, side: Side, volume: int, price: float) -> bool:
        """Send + immediate cancel of remainder. Returns True if any fill."""
        resp = self.send_order(OrderRequest(symbol, price, side, volume))
        if resp is None:
            return False
        filled = resp.filled
        if resp.volume > 0:
            self.cancel_order(resp.id)
        return filled > 0

    def _aggress(self, symbol: str, side: Side, volume: int) -> bool:
        """Hit the best resting price on the opposite side."""
        with self._ob_lock:
            ob = self._orderbooks.get(symbol)
        if ob is None:
            return False

        if side == Side.BUY:
            levels = [o for o in ob.sell_orders if o.volume - o.own_volume > 0]
            if not levels:
                return False
            price = levels[0].price
        else:
            levels = [o for o in ob.buy_orders if o.volume - o.own_volume > 0]
            if not levels:
                return False
            price = levels[0].price

        return self._ioc(symbol, side, volume, price)

    def _aggress_multiple_levels(self, symbol: str, side: Side, total_vol: int) -> int:
        """Sweep through multiple price levels to fill larger size. Returns total filled."""
        with self._ob_lock:
            ob = self._orderbooks.get(symbol)
        if ob is None:
            return 0

        if side == Side.BUY:
            levels = [o for o in ob.sell_orders if o.volume - o.own_volume > 0]
        else:
            levels = [o for o in ob.buy_orders if o.volume - o.own_volume > 0]

        if not levels:
            return 0

        remaining = total_vol
        filled_total = 0
        for level in levels[:3]:  # Sweep up to 3 levels
            if remaining <= 0:
                break
            vol = min(remaining, level.volume - level.own_volume)
            if vol <= 0:
                continue
            if self._ioc(symbol, side, vol, level.price):
                filled_total += vol
                remaining -= vol

        return filled_total

    # ─────────── Fair value blending ───────────

    def _blended_fv(self, symbol: str) -> Optional[float]:
        data_fv = self.fv.values.get(symbol) if self.fv else None
        mid = self._market_mids.get(symbol)
        conf = (self.fv.confidence.get(symbol, 0) if self.fv else 0)

        if data_fv is not None and mid is not None:
            alpha = min(conf, 0.85)
            return alpha * data_fv + (1 - alpha) * mid
        return data_fv or mid

    def _can_add(self, symbol: str, side: Side, size: int) -> bool:
        pos = self._positions.get(symbol, 0)
        new = pos + size if side == Side.BUY else pos - size
        return abs(new) <= MAX_POSITION

    def _skew_for_position(self, symbol: str) -> float:
        """
        Computes price skew based on inventory regime.
        Returns a value to subtract from both bid and ask prices.
        Positive position -> negative skew (push prices down to shed longs).
        Negative position -> positive skew (push prices up to shed shorts).
        """
        # Get our current position for this specific product
        pos = self._positions.get(symbol, 0)
        
        # Check which regime we are in using the global function
        regime = _inventory_regime(pos)
        
        if regime == "critical":
            rate = SKEW_CRITICAL
        elif regime == "warn":
            rate = SKEW_WARN
        else:
            rate = SKEW_NORMAL
            
        return pos * rate
    # ═══════════════════ STRATEGY 1: ALPHA SIGNALS ═══════════════════

    def _strategy_alpha_signals(self):
        """
        Directional trades when data-driven FV diverges from market.
        Only acts on products with meaningful confidence.
        Uses adaptive sizing: bigger when confident, smaller near limits.
        """
        if self.fv is None:
            return

        now = time.monotonic()
        for symbol in ALL_PRODUCTS:
            conf = self.fv.confidence.get(symbol, 0)
            if conf < 0.25:
                continue

            fv = self._blended_fv(symbol)
            mid = self._market_mids.get(symbol)
            spread = self._market_spreads.get(symbol, 999)
            if fv is None or mid is None or mid == 0:
                continue

            # Cooldown
            if now - self._signal_last_time.get(symbol, 0) < 8:
                continue

            edge_frac = (fv - mid) / abs(mid)
            pos = self._positions.get(symbol, 0)

            # Dynamic entry threshold: tighter when confident, wider for noisy products
            entry_thresh = SIGNAL_ENTRY_FRAC / max(conf, 0.3)

            # ── Entry ──
            if edge_frac > entry_thresh:
                size = self._scale_size(conf, pos, Side.BUY)
                if size > 0 and self._can_add(symbol, Side.BUY, size):
                    print(f"  [SIGNAL] BUY {symbol}  edge={edge_frac:+.3f}"
                          f"  conf={conf:.2f}  fv={fv:.0f}  mkt={mid:.0f}  sz={size}")
                    filled = self._aggress_multiple_levels(symbol, Side.BUY, size)
                    if filled:
                        self._signal_last_time[symbol] = now

            elif edge_frac < -entry_thresh:
                size = self._scale_size(conf, pos, Side.SELL)
                if size > 0 and self._can_add(symbol, Side.SELL, size):
                    print(f"  [SIGNAL] SELL {symbol}  edge={edge_frac:+.3f}"
                          f"  conf={conf:.2f}  fv={fv:.0f}  mkt={mid:.0f}  sz={size}")
                    filled = self._aggress_multiple_levels(symbol, Side.SELL, size)
                    if filled:
                        self._signal_last_time[symbol] = now

            # ── Exit: edge collapsed, reduce position ──
            """
            elif abs(edge_frac) < SIGNAL_EXIT_FRAC and abs(pos) > 0:
                if now - self._signal_last_time.get(symbol, 0) < 15:
                    continue  # Don't exit too fast
                if pos > 0:
                    reduce = min(pos, POSITION_PER_SIGNAL)
                    print(f"  [EXIT] Reducing {symbol} long by {reduce}  edge={edge_frac:+.3f}")
                    self._aggress(symbol, Side.SELL, reduce)
                    self._signal_last_time[symbol] = now
                elif pos < 0:
                    reduce = min(-pos, POSITION_PER_SIGNAL)
                    print(f"  [EXIT] Reducing {symbol} short by {reduce}  edge={edge_frac:+.3f}")
                    self._aggress(symbol, Side.BUY, reduce)
                    self._signal_last_time[symbol] = now
                """
    def _strategy_passive_unwind(self):
        """Places resting orders to exit stale positions profitably."""
        now = time.monotonic()
        
        # Cooldown: Only update our resting quotes every 5 seconds so we don't spam the API
        if now - self._last_unwind_time < 5.0:
            return
        self._last_unwind_time = now

        for symbol in ALL_PRODUCTS:
            pos = self._positions.get(symbol, 0)
            
            # If we are flat (0 position), cancel any leftover unwind orders and skip
            if pos == 0:
                if symbol in self._unwind_orders:
                    self.cancel_order(self._unwind_orders[symbol])
                    del self._unwind_orders[symbol]
                continue
                
            mid = self._market_mids.get(symbol)
            spread = self._market_spreads.get(symbol, 2.0)
            if not mid: 
                continue

            # Get the skew (pushes price down if long, up if short)
            skew = self._skew_for_position(symbol)
            tick = self._tick.get(symbol, 1.0)
            
            # Unwind in chunks of 3 so we don't show our full hand to the market
            unwind_vol = min(abs(pos), 3) 
            
            order = None
            if pos > 0:
                # We are long -> place a resting SELL order above mid, pulled down by our skew
                ask_price = math.ceil((mid + (spread / 2.0) - skew) / tick) * tick
                if ask_price > 0:
                    order = OrderRequest(symbol, ask_price, Side.SELL, unwind_vol)
                    
            elif pos < 0:
                # We are short -> place a resting BUY order below mid, pushed up by our skew
                bid_price = math.floor((mid - (spread / 2.0) - skew) / tick) * tick
                if bid_price > 0:
                    order = OrderRequest(symbol, bid_price, Side.BUY, unwind_vol)
                    
            if order:
                # Cancel the old unwind order for this product to avoid layering
                if symbol in self._unwind_orders:
                    self.cancel_order(self._unwind_orders[symbol])
                    
                # Send the new order and save its ID
                resp = self.send_order(order)
                if resp and resp.id:
                    self._unwind_orders[symbol] = resp.id
                    print(f"  [UNWIND] Resting {order.side.value} {order.volume}x {symbol} @ {order.price:.0f} (skew={-skew:+.1f})")

    def _scale_size(self, confidence: float, current_pos: int, side: Side) -> int:
        base = POSITION_PER_SIGNAL

        if confidence > 0.7:
            base += 4
        elif confidence > 0.5:
            base += 2
        elif confidence > 0.35:
            base += 1

        # Reduce if already extended in this direction
        if side == Side.BUY and current_pos > 40:
            base = max(1, base - 3)
        elif side == Side.SELL and current_pos < -40:
            base = max(1, base - 3)

        # Increase if this trade would reduce our position (mean-reverting)
        if side == Side.BUY and current_pos < -10:
            base += 2
        elif side == Side.SELL and current_pos > 10:
            base += 2

        new_pos = current_pos + base if side == Side.BUY else current_pos - base
        if abs(new_pos) > MAX_POSITION:
            base = max(0, MAX_POSITION - abs(current_pos))

        return base

    # ═══════════════════ STRATEGY 2: ETF ARBITRAGE ═══════════════════

    def _strategy_etf_arb(self):
        """Trade LON_ETF vs sum of components when they diverge."""
        now = time.monotonic()
        if now - self._arb_last_time.get("etf", 0) < 5:
            return

        comp_mids = []
        for c in ETF_COMPONENTS:
            m = self._market_mids.get(c)
            if m is None:
                return
            comp_mids.append(m)

        etf_mid = self._market_mids.get(LON_ETF)
        if etf_mid is None:
            return

        implied = sum(comp_mids)
        gap = etf_mid - implied

        if abs(gap) < ETF_ARB_THRESHOLD:
            return

        if gap > ETF_ARB_THRESHOLD:
            # ETF expensive -> sell ETF, buy components
            if self._can_add(LON_ETF, Side.SELL, POSITION_PER_ARB):
                print(f"  [ARB] ETF expensive by {gap:.0f}  sell ETF, buy comps")
                self._aggress(LON_ETF, Side.SELL, POSITION_PER_ARB)
                for c in ETF_COMPONENTS:
                    if self._can_add(c, Side.BUY, POSITION_PER_ARB):
                        self._aggress(c, Side.BUY, POSITION_PER_ARB)
                self._arb_last_time["etf"] = now

        elif gap < -ETF_ARB_THRESHOLD:
            # ETF cheap -> buy ETF, sell components
            if self._can_add(LON_ETF, Side.BUY, POSITION_PER_ARB):
                print(f"  [ARB] ETF cheap by {abs(gap):.0f}  buy ETF, sell comps")
                self._aggress(LON_ETF, Side.BUY, POSITION_PER_ARB)
                for c in ETF_COMPONENTS:
                    if self._can_add(c, Side.SELL, POSITION_PER_ARB):
                        self._aggress(c, Side.SELL, POSITION_PER_ARB)
                self._arb_last_time["etf"] = now

    # ═══════════════════ STRATEGY 2b: OPTION ARB ═══════════════════

    def _strategy_option_arb(self):
        """Price LON_FLY analytically from ETF and trade mispricings."""
        now = time.monotonic()
        if now - self._arb_last_time.get("opt", 0) < 8:
            return

        etf_fv = self._blended_fv(LON_ETF)
        fly_mid = self._market_mids.get(LON_FLY)
        if etf_fv is None or fly_mid is None:
            return

        # Compute theoretical at a range of ETF values to get a confidence band
        theo = compute_option_payoff(etf_fv)

        # Also compute at +/- 1 std of our ETF uncertainty
        etf_conf = self.fv.confidence.get(LON_ETF, 0) if self.fv else 0
        etf_mid = self._market_mids.get(LON_ETF, etf_fv)
        etf_uncertainty = abs(etf_fv - etf_mid) * 2 if etf_conf < 0.5 else abs(etf_fv - etf_mid)

        theo_high = compute_option_payoff(etf_fv + etf_uncertainty)
        theo_low = compute_option_payoff(etf_fv - etf_uncertainty)
        theo_range = abs(theo_high - theo_low)

        # Only trade if mispricing exceeds our uncertainty
        effective_threshold = max(OPT_ARB_THRESHOLD, theo_range * 0.5)
        gap = fly_mid - theo

        if abs(gap) < effective_threshold:
            return

        fly_pos = self._positions.get(LON_FLY, 0)

        if gap > effective_threshold and self._can_add(LON_FLY, Side.SELL, POSITION_PER_ARB):
            print(f"  [OPT] FLY overpriced by {gap:.0f}  (theo={theo:.0f} mkt={fly_mid:.0f})")
            self._aggress(LON_FLY, Side.SELL, POSITION_PER_ARB)
            self._arb_last_time["opt"] = now

        elif gap < -effective_threshold and self._can_add(LON_FLY, Side.BUY, POSITION_PER_ARB):
            print(f"  [OPT] FLY underpriced by {abs(gap):.0f}  (theo={theo:.0f} mkt={fly_mid:.0f})")
            self._aggress(LON_FLY, Side.BUY, POSITION_PER_ARB)
            self._arb_last_time["opt"] = now

    # ═══════════════════ STRATEGY 3: VOLATILITY FADE ═══════════════════

    def _strategy_volatility_fade(self):
        """
        Exploit chaotic price action from 40-50 student bots.

        When a product's mid spikes to extreme z-scores vs its recent
        rolling window, fade the move (sell spikes, buy crashes). The
        key insight: most student bots either follow momentum blindly
        or panic on stale data, creating mean-reverting dislocations.

        Two layers:
        1. Z-score fade: classic mean reversion on extreme moves
        2. Range breakout fade: if price overshoots the rolling range
           by a lot, fade back toward range midpoint
        """
        for symbol in ALL_PRODUCTS:
            z = self.vol_tracker.get_zscore(symbol)
            rng = self.vol_tracker.get_range(symbol)
            pos = self._positions.get(symbol, 0)
            mid = self._market_mids.get(symbol)

            if z is None or rng is None or mid is None:
                continue

            range_mid = (rng[0] + rng[1]) / 2
            range_width = rng[1] - rng[0]

            # ── Entry: extreme z-score ──
            if abs(z) > VOL_ZSCORE_ENTRY and self.vol_tracker.can_fade(symbol):

                # Extra check: don't fade if our FV actually agrees with the move
                # (i.e., if the spike is toward our fair value, it's not a dislocation)
                fv = self._blended_fv(symbol)
                if fv is not None:
                    if z > 0 and mid < fv:
                        continue  # Spike is toward FV, not away — don't fade
                    if z < 0 and mid > fv:
                        continue

                if z > VOL_ZSCORE_ENTRY and self._can_add(symbol, Side.SELL, VOL_TRADE_SIZE):
                    print(f"  [VFADE] SELL {symbol}  z={z:+.2f}  mid={mid:.0f}  "
                          f"range=[{rng[0]:.0f},{rng[1]:.0f}]")
                    if self._aggress(symbol, Side.SELL, VOL_TRADE_SIZE):
                        self.vol_tracker.record_fade(symbol)

                elif z < -VOL_ZSCORE_ENTRY and self._can_add(symbol, Side.BUY, VOL_TRADE_SIZE):
                    print(f"  [VFADE] BUY {symbol}  z={z:+.2f}  mid={mid:.0f}  "
                          f"range=[{rng[0]:.0f},{rng[1]:.0f}]")
                    if self._aggress(symbol, Side.BUY, VOL_TRADE_SIZE):
                        self.vol_tracker.record_fade(symbol)

            # ── Exit: z-score reverted ──
            '''
            elif abs(z) < VOL_ZSCORE_EXIT and abs(pos) > 0:
                if pos > 0 and z > -0.3:
                    reduce = min(pos, VOL_TRADE_SIZE)
                    self._aggress(symbol, Side.SELL, reduce)
                elif pos < 0 and z < 0.3:
                    reduce = min(-pos, VOL_TRADE_SIZE)
                    self._aggress(symbol, Side.BUY, reduce)
            '''

    # ═══════════════════ STRATEGY 5: TRAP ORDERS ═══════════════════

    def _strategy_trap_orders(self):
        """
        Maintain resting orders at extreme prices that can never be fair value.
        These only fill if another team's bot malfunctions and market-sells into
        our buy at 1, or market-buys into our sell at 10000, etc.

        Pure free money — the position we end up with is always profitable
        because we bought/sold at a price far from any possible settlement.

        Refreshed infrequently to avoid spamming the exchange.
        """
        now = time.monotonic()
        if now - self._last_trap_refresh < TRAP_REFRESH_INTERVAL:
            return
        self._last_trap_refresh = now

        # Get all our current resting orders to see which traps are alive
        all_orders = self.get_orders()
        live_order_ids = {o["id"] for o in all_orders}

        # Prune dead trap IDs
        dead_keys = [k for k, oid in self._trap_order_ids.items()
                     if oid not in live_order_ids]
        for k in dead_keys:
            del self._trap_order_ids[k]

        orders_to_place = []

        for symbol, (buy_price, sell_price) in TRAP_ORDERS.items():
            if symbol not in self._products:
                continue  # Product doesn't exist on this exchange

            tick = self._tick.get(symbol, 1.0)
            pos = self._positions.get(symbol, 0)

            # ── Trap BUY (absurdly low price) ──
            buy_key = f"{symbol}_BUY"
            if buy_key not in self._trap_order_ids:
                # Only place if we have room to go long
                if pos + TRAP_VOLUME <= MAX_POSITION:
                    snap_price = math.floor(buy_price / tick) * tick
                    if snap_price > 0:
                        orders_to_place.append((buy_key, OrderRequest(
                            symbol, snap_price, Side.BUY, TRAP_VOLUME)))

            # ── Trap SELL (absurdly high price) ──
            sell_key = f"{symbol}_SELL"
            if sell_key not in self._trap_order_ids:
                # Only place if we have room to go short
                if pos - TRAP_VOLUME >= -MAX_POSITION:
                    snap_price = math.ceil(sell_price / tick) * tick
                    orders_to_place.append((sell_key, OrderRequest(
                        symbol, snap_price, Side.SELL, TRAP_VOLUME)))

        # Send all trap orders
        for key, order in orders_to_place:
            resp = self.send_order(order)
            if resp and resp.id:
                self._trap_order_ids[key] = resp.id
                # Check if it somehow filled immediately (lucky us!)
                if resp.filled > 0:
                    print(f"  [TRAP] ** FILLED ** {order.side.value} {resp.filled}x"
                          f" {order.product} @ {order.price}  FREE MONEY!")

        if orders_to_place:
            n_placed = len(orders_to_place)
            n_alive = len(self._trap_order_ids)
            print(f"  [TRAP] Placed {n_placed} new traps, {n_alive} total alive")

    # ═══════════════════ MAIN LOOP ═══════════════════

    def run(self, interval: float = LOOP_INTERVAL):
        self._products = {p.symbol: p for p in self.get_products()}
        self._tick = {s: p.tickSize for s, p in self._products.items()}
        self._positions = defaultdict(int, self.get_positions())

        print(f"[INIT] Products: {list(self._products.keys())}")
        print(f"[INIT] Positions: {dict(self._positions)}")

        self.start()
        time.sleep(1)

        self.fv_engine.refresh_data()
        self.fv = self.fv_engine.compute()
        self._print_fv()

        iteration = 0
        while True:
            try:
                iteration += 1
                now = time.monotonic()

                if now - self._last_data_refresh > DATA_REFRESH_INTERVAL:
                    self.fv_engine.refresh_data()
                    self._last_data_refresh = now

                if now - self._last_fv_compute > FV_RECOMPUTE_INTERVAL:
                    self.fv = self.fv_engine.compute()
                    self._last_fv_compute = now

                if iteration % 15 == 0:
                    self._positions = defaultdict(int, self.get_positions())

                # ── Run all strategies ──
                self._strategy_alpha_signals()
                self._strategy_passive_unwind()
                self._strategy_etf_arb()
                self._strategy_option_arb()
                self._strategy_volatility_fade()
                self._strategy_trap_orders()

                if iteration % 40 == 0:
                    self._print_fv()
                    self._print_status()

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n[STOP] Shutting down...")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback; traceback.print_exc()
                time.sleep(3)

        self.cancel_all_orders()
        self.stop()
        print("[STOP] Done.")

    def _print_fv(self):
        if self.fv is None:
            return
        print("\n[FV] Fair Values & Market:")
        for s in ALL_PRODUCTS:
            fv = self.fv.values.get(s)
            c = self.fv.confidence.get(s, 0)
            mid = self._market_mids.get(s)
            spr = self._market_spreads.get(s)
            bld = self._blended_fv(s)
            z = self.vol_tracker.get_zscore(s)
            pos = self._positions.get(s, 0)

            fv_s = f"{fv:>8.0f}" if fv is not None else "       ?"
            mid_s = f"{mid:>8.1f}" if mid is not None else "       ?"
            bld_s = f"{bld:>8.1f}" if bld is not None else "       ?"
            spr_s = f"spr={spr:>4.0f}" if spr is not None else "spr=   ?"
            z_s = f"z={z:+5.1f}" if z is not None else "z=    ?"
            print(f"  {s:<12} data={fv_s}  mkt={mid_s}  blend={bld_s}"
                  f"  conf={c:.2f}  {spr_s}  {z_s}  pos={pos:+d}")
        print()

    def _print_status(self):
        pos = {k: v for k, v in self._positions.items() if v != 0}
        pnl = self.get_pnl()
        print(f"[STATUS] fills={self._trade_count}  pos={pos}")
        print(f"         pnl={pnl}\n")


# ─────────────────────── Entry Point ───────────────────────

def main():
    # ══ CONFIGURE THESE ══
    EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
    # EXCHANGE_URL = "REPLACE_WITH_CHALLENGE_URL"
    USERNAME = "Market Fakers"
    PASSWORD = "marketfakers123" 

    bot = SignalBot(EXCHANGE_URL, USERNAME, PASSWORD)
    print("=" * 62)
    print("  IMCity Signal & Volatility Bot")
    print("  Strategies: Alpha | ETF Arb | Option Arb | Vol Fade | Traps")
    print(f"  User: {USERNAME}")
    print("=" * 62)
    bot.run()


if __name__ == "__main__":
    main()
