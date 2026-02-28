"""
IMCity ETF Arb Bot — Production Edition
=========================================
Fixes applied vs v2 (critiques addressed):

  [C1] Legging risk       → Full order-book depth stored; WAP used for all arb
                             profit calculations; arb only fires if WAP-adjusted
                             profit still exceeds threshold.

  [C2] Position limits    → Hard abort at ±POSITION_LIMIT (90).
                             Aggressive skew kicks in at ±INVENTORY_WARN (80).
                             Emergency skew at ±INVENTORY_CRITICAL (90).
                             Quoted volume scales down automatically near limits.

  [C3] Settlement risk    → FV blend changed to 90% fundamental / 10% market.
                             FV sanity gate on every arb: if either side deviates
                             more than FV_SANITY_PCT from our fundamental estimate,
                             the arb is rejected (market has bad data, not free money).

  [C4] API ban risk       → Own ETF order IDs tracked in _our_etf_orders set.
                             Requote cancels only those IDs (no extra GET call).
                             _check_cross_arb deduplicates: only runs once per
                             *new* price level change, not on every SSE tick.

  [C5] Winner's curse     → wap_cost() walks full book depth to compute realistic
                             fill price for a given volume. Arb profit is computed
                             as WAP(sell legs) - WAP(buy legs), not best_bid/ask.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from cmi_bot import BaseBot, Order, OrderBook, OrderRequest, OrderResponse, Side, Trade

# ── Product symbols ────────────────────────────────────────────────────────────
TIDE_SPOT = "TIDE_SPOT"
WX_SPOT   = "WX_SPOT"
LHR_COUNT = "LHR_COUNT"
LON_ETF   = "LON_ETF"

COMPONENTS   = [TIDE_SPOT, WX_SPOT, LHR_COUNT]
ALL_PRODUCTS = COMPONENTS + [LON_ETF]

# ── Tunable parameters ─────────────────────────────────────────────────────────
POSITION_LIMIT       = 100    # exchange hard limit
POSITION_SOFT_LIMIT  = 90     # our soft limit — never intentionally exceed
INVENTORY_WARN       = 80     # aggressive skew kicks in here
INVENTORY_CRITICAL   = 90     # near hard limit — emergency skew, no new buys/sells

ORDER_VOLUME         = 3      # baseline contracts per quote leg
MIN_QUOTE_SPREAD     = 20.0   # minimum total spread we'll ever quote (10 each side)
SPREAD_FRACTION      = 0.35   # our spread = 35% of sum of component spreads

SKEW_NORMAL          = 2.5    # price skew per contract (normal regime)
SKEW_WARN            = 6.0    # price skew per contract (warn regime, pos > 80)
SKEW_CRITICAL        = 15.0   # price skew per contract (critical regime, pos > 90)

ARB_THRESHOLD        = 10.0   # min WAP-adjusted profit per contract to fire arb
FV_SANITY_PCT        = 0.25   # reject arb if either price deviates > 25% from our FV
                               # (protects against the whole market having bad data)
FV_BLEND_FUNDAMENTAL = 0.90   # weight given to real London data in FV calculation
FV_BLEND_MARKET      = 0.10   # weight given to market mid (price discovery signal)

REQUOTE_INTERVAL     = 20.0   # seconds between ETF requotes
DATA_REFRESH_SECS    = 60*14  # refresh London data every 14 min
HEDGE_AGGRESSION     = 5.0    # extra ticks when posting hedge in empty book
ARB_COOLDOWN         = 2.0    # min seconds between opportunistic arb attempts

LHR_BASELINE         = 1400.0

THAMES_URL = (
    "https://environment.data.gov.uk/flood-monitoring/id/measures"
    "/0006-level-tidal_level-i-15_min-mAOD/readings?_limit=1&_sorted"
)
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=51.5074&longitude=-0.1278"
    "&current=temperature_2m,relative_humidity_2m"
    "&temperature_unit=fahrenheit"
)


# ── Full-depth book snapshot ───────────────────────────────────────────────────
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


# ── Inventory state helper ─────────────────────────────────────────────────────
def _inventory_regime(pos: int) -> str:
    """Return 'normal', 'warn', or 'critical' based on absolute position. [C2 fix]"""
    abs_pos = abs(pos)
    if abs_pos >= INVENTORY_CRITICAL:
        return "critical"
    if abs_pos >= INVENTORY_WARN:
        return "warn"
    return "normal"


# ── Main Bot ───────────────────────────────────────────────────────────────────
class ThinMarketETFBot(BaseBot):

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)

        # Full-depth books [C1, C5 fix]
        self._books: dict[str, BookSnap] = {}
        self._books_lock = threading.Lock()

        # Last price seen per product — used to deduplicate arb checks [C4 fix]
        self._last_price: dict[str, Optional[float]] = {p: None for p in ALL_PRODUCTS}

        # Fundamental fair-value estimates from live London data [C3 fix]
        # These are 90% weighted in all FV calculations
        self._fv: dict[str, float] = {
            TIDE_SPOT: 1400.0,
            WX_SPOT:   3500.0,
            LHR_COUNT: LHR_BASELINE,
        }
        self._fv_lock = threading.Lock()

        # Position tracking
        self._pos: dict[str, int] = {p: 0 for p in ALL_PRODUCTS}
        self._pos_lock = threading.Lock()

        # Track our own ETF order IDs to avoid GET call on cancel [C4 fix]
        self._our_etf_orders: set[str] = set()
        self._order_id_lock  = threading.Lock()

        # Unhedged ETF delta (ETF fills awaiting component hedges)
        self._unhedged_etf = 0
        self._hedge_lock   = threading.Lock()

        # Stats
        self._arb_count     = 0
        self._hedge_count   = 0
        self._total_pnl_est = 0.0
        self._last_arb_time = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(f"\n{'='*56}")
        print(f"  IMCity ETF Bot v3  |  user: {self.username}")
        print(f"  C1:WAP  C2:Inventory  C3:FV-gate  C4:SSE  C5:Slippage")
        print(f"{'='*56}\n")

        self._refresh_london_data()
        self._sync_positions()
        self.start()

        threading.Thread(target=self._data_refresh_loop, daemon=True).start()
        threading.Thread(target=self._requote_loop,      daemon=True).start()
        threading.Thread(target=self._status_loop,       daemon=True).start()

        print("[Bot] Running. Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Bot] Shutting down...")
        finally:
            self.cancel_all_orders()
            self.stop()

    def _sync_positions(self) -> None:
        live = self.get_positions()
        with self._pos_lock:
            for p, qty in live.items():
                if p in self._pos:
                    self._pos[p] = qty
        print(f"[Init] Positions: {dict(self._pos)}")

    # ── SSE Callbacks ──────────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook) -> None:
        snap     = _snap(orderbook)
        product  = orderbook.product
        new_mid  = snap.mid

        with self._books_lock:
            self._books[product] = snap

        # [C4 fix] Only run arb check if the mid price actually changed.
        # Prevents hammering arb logic on every heartbeat / size-only update.
        old_mid = self._last_price.get(product)
        price_changed = (new_mid != old_mid)
        self._last_price[product] = new_mid

        if price_changed:
            self._check_cross_arb()

    def on_trades(self, trade: Trade) -> None:
        is_ours = (trade.buyer == self.username or trade.seller == self.username)
        if not is_ours:
            return

        is_buy = trade.buyer == self.username
        delta  = trade.volume if is_buy else -trade.volume

        with self._pos_lock:
            self._pos[trade.product] = self._pos.get(trade.product, 0) + delta

        regime = _inventory_regime(self._pos[trade.product])
        print(f"[Fill] {'BUY' if is_buy else 'SELL'} {trade.volume} "
              f"{trade.product} @ {trade.price}  "
              f"pos={self._pos[trade.product]}  [{regime}]")

        # ETF fill → hedge immediately
        if trade.product == LON_ETF:
            with self._hedge_lock:
                self._unhedged_etf += delta
                to_hedge = self._unhedged_etf
            if to_hedge != 0:
                self._fire_component_hedges(to_hedge)
                with self._hedge_lock:
                    self._unhedged_etf = 0
                self._hedge_count += 1

    # ── London Data ────────────────────────────────────────────────────────────

    def _refresh_london_data(self) -> None:
        print("[Data] Refreshing fundamental fair values from London APIs...")

        try:
            r = requests.get(THAMES_URL, timeout=10)
            level = float(r.json()["items"][0]["value"])
            fv = abs(level) * 1000
            with self._fv_lock:
                self._fv[TIDE_SPOT] = fv
            print(f"[Data] Thames {level:.3f} mAOD → TIDE_SPOT FV={fv:.1f}")
        except Exception as e:
            print(f"[Data] Thames failed: {e}")

        try:
            r = requests.get(WEATHER_URL, timeout=10)
            cur      = r.json()["current"]
            temp_f   = cur["temperature_2m"]
            humidity = cur["relative_humidity_2m"]
            fv = round(temp_f) * humidity
            with self._fv_lock:
                self._fv[WX_SPOT] = fv
            print(f"[Data] Weather {temp_f:.1f}°F {humidity}% → WX_SPOT FV={fv:.1f}")
        except Exception as e:
            print(f"[Data] Weather failed: {e}")

        print(f"[Data] LHR baseline FV={self._fv[LHR_COUNT]:.1f}  (static)")

    def _data_refresh_loop(self) -> None:
        while True:
            time.sleep(DATA_REFRESH_SECS)
            self._refresh_london_data()

    # ── Fair Value Engine ──────────────────────────────────────────────────────

    def _component_fv(self, product: str) -> float:
        """
        Fair value for one component.
        Blend: 90% fundamental (real London data) + 10% market mid.
        [C3 fix] High fundamental weight means bad market data can't fool us.
        """
        with self._fv_lock:
            fundamental = self._fv[product]
        with self._books_lock:
            snap = self._books.get(product)

        market_mid = snap.mid if snap else None

        if market_mid is None:
            return fundamental

        return FV_BLEND_FUNDAMENTAL * fundamental + FV_BLEND_MARKET * market_mid

    def _etf_fv(self) -> float:
        """Synthetic ETF fair value = sum of component FVs."""
        return sum(self._component_fv(p) for p in COMPONENTS)

    def _fv_sanity_check(self, product: str, market_price: float) -> bool:
        """
        [C3 fix] Return True only if market_price is within FV_SANITY_PCT of our
        fundamental estimate. Rejects arbs where the whole market has bad data.
        """
        with self._fv_lock:
            fundamental = self._fv.get(product)

        if fundamental is None or fundamental == 0:
            return True  # no data to reject on

        deviation = abs(market_price - fundamental) / fundamental
        if deviation > FV_SANITY_PCT:
            print(f"[FV-Gate] {product} market={market_price:.1f} "
                  f"fundamental={fundamental:.1f} "
                  f"deviation={deviation:.1%} > {FV_SANITY_PCT:.0%} → REJECTED")
            return False
        return True

    # ── Inventory Management ───────────────────────────────────────────────────

    def _skew_for_position(self, product: str) -> float:
        """
        [C2 fix] Compute price skew based on inventory regime.
        Returns a value to subtract from both bid and ask prices.
        Positive position → negative skew (push prices down to shed longs).
        Negative position → positive skew (push prices up to shed shorts).
        """
        with self._pos_lock:
            pos = self._pos.get(product, 0)

        regime = _inventory_regime(pos)
        if regime == "critical":
            rate = SKEW_CRITICAL
        elif regime == "warn":
            rate = SKEW_WARN
        else:
            rate = SKEW_NORMAL

        return pos * rate

    def _vol_within_limit(self, product: str, side: Side, volume: int) -> int:
        """
        [C2 fix] Clip volume to stay within POSITION_SOFT_LIMIT.
        Also scales down volume in warn/critical regime.
        """
        with self._pos_lock:
            pos = self._pos.get(product, 0)

        regime = _inventory_regime(pos)

        if side == Side.BUY:
            room = POSITION_SOFT_LIMIT - pos
        else:
            room = POSITION_SOFT_LIMIT + pos

        # [C2 fix] In warn/critical regime, be more conservative on new trades
        if regime == "critical":
            # Only allow trades that REDUCE position
            if (side == Side.BUY and pos > 0) or (side == Side.SELL and pos < 0):
                return 0  # hard abort: would increase already-critical position
            max_vol = 1  # very small
        elif regime == "warn":
            max_vol = max(1, volume // 2)  # half volume in warn regime
        else:
            max_vol = volume

        return max(0, min(max_vol, room))

    # ── ETF Quote Posting ──────────────────────────────────────────────────────

    def _requote_etf(self) -> None:
        """
        Post two-sided ETF quotes anchored to synthetic fair value.
        Uses inventory skew to mean-revert position.
        Cancels only OUR tracked order IDs — no extra GET call. [C4 fix]
        """
        fv   = self._etf_fv()
        skew = self._skew_for_position(LON_ETF)

        with self._books_lock:
            etf_snap = self._books.get(LON_ETF)
            comp_spreads = [
                self._books[p].spread
                for p in COMPONENTS
                if p in self._books and self._books[p].spread is not None
            ]

        if comp_spreads:
            half_spread = max(MIN_QUOTE_SPREAD / 2,
                              sum(comp_spreads) * SPREAD_FRACTION / 2)
        else:
            half_spread = 40.0  # wide fallback when no component data

        bid_price = fv - half_spread - skew
        ask_price = fv + half_spread - skew

        # [C4 fix] Cancel only our tracked ETF orders — no GET request needed
        with self._order_id_lock:
            ids_to_cancel = list(self._our_etf_orders)
            self._our_etf_orders.clear()

        if ids_to_cancel:
            threads = [threading.Thread(target=self.cancel_order, args=(oid,))
                       for oid in ids_to_cancel]
            for t in threads: t.start()
            for t in threads: t.join()

        orders_to_send = []
        bid_vol = self._vol_within_limit(LON_ETF, Side.BUY, ORDER_VOLUME)
        if bid_vol > 0:
            orders_to_send.append(
                OrderRequest(LON_ETF, bid_price, Side.BUY, bid_vol)
            )
        ask_vol = self._vol_within_limit(LON_ETF, Side.SELL, ORDER_VOLUME)
        if ask_vol > 0:
            orders_to_send.append(
                OrderRequest(LON_ETF, ask_price, Side.SELL, ask_vol)
            )

        if orders_to_send:
            responses = self.send_orders(orders_to_send)
            # [C4 fix] Track returned order IDs
            with self._order_id_lock:
                for r in responses:
                    if r and hasattr(r, "id"):
                        self._our_etf_orders.add(r.id)

        with self._pos_lock:
            pos = self._pos[LON_ETF]
        regime = _inventory_regime(pos)

        print(f"[Quote] ETF  fv={fv:.0f}  "
              f"bid={bid_price:.0f}  ask={ask_price:.0f}  "
              f"spread={half_spread*2:.0f}  skew={-skew:+.0f}  "
              f"pos={pos}  [{regime}]")

    def _requote_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self._requote_etf()
            except Exception as e:
                print(f"[Quote] Error: {e}")
            time.sleep(REQUOTE_INTERVAL)

    # ── Component Hedge Execution ──────────────────────────────────────────────

    def _fire_component_hedges(self, etf_delta: int) -> None:
        """
        Hedge ETF position by trading components.
        [C1 fix] Uses WAP to find best available price; posts aggressively.
        [C2 fix] Respects inventory limits on each component.
        """
        hedge_side = Side.SELL if etf_delta > 0 else Side.BUY
        volume     = abs(etf_delta)

        print(f"\n[Hedge] ETF Δ={etf_delta:+d} → {hedge_side} {volume} × each component")

        orders = []
        for product in COMPONENTS:
            with self._books_lock:
                snap = self._books.get(product)
            with self._fv_lock:
                fundamental = self._fv[product]

            vol = self._vol_within_limit(product, hedge_side, volume)
            if vol == 0:
                print(f"  [Hedge] {product} skipped — position limit reached")
                continue

            # Determine price: walk the book first, fall back to fundamental
            if hedge_side == Side.SELL:
                price = (snap.wap_sell(vol) if snap else None) or \
                        (snap.best_bid if snap else None) or \
                        (snap.best_ask - HEDGE_AGGRESSION if snap and snap.best_ask else None) or \
                        fundamental - HEDGE_AGGRESSION
            else:
                price = (snap.wap_buy(vol) if snap else None) or \
                        (snap.best_ask if snap else None) or \
                        (snap.best_bid + HEDGE_AGGRESSION if snap and snap.best_bid else None) or \
                        fundamental + HEDGE_AGGRESSION

            orders.append(OrderRequest(product, price, hedge_side, vol))
            print(f"  → {hedge_side} {vol} {product} @ {price:.1f}")

        if orders:
            self.send_orders(orders)

    # ── Opportunistic Cross-Arb ────────────────────────────────────────────────

    def _check_cross_arb(self) -> None:
        """
        Cross an arb only when ALL five conditions pass:
          1. All four books have quotes
          2. ARB_COOLDOWN has elapsed
          3. [C5] WAP-adjusted profit > ARB_THRESHOLD (not just best bid/ask)
          4. [C3] Both sides are within FV_SANITY_PCT of our fundamental estimates
          5. [C2] Sufficient position headroom exists on all legs
        """
        now = time.monotonic()
        if now - self._last_arb_time < ARB_COOLDOWN:
            return

        with self._books_lock:
            if not all(p in self._books for p in ALL_PRODUCTS):
                return
            etf  = self._books[LON_ETF]
            tide = self._books[TIDE_SPOT]
            wx   = self._books[WX_SPOT]
            lhr  = self._books[LHR_COUNT]

        vol = ORDER_VOLUME  # will be clipped below

        # ── Case A: buy ETF, sell components ──────────────────────────────────
        if etf.asks and tide.bids and wx.bids and lhr.bids:

            # [C5] WAP across full book depth
            wap_buy_etf    = etf.wap_buy(vol)
            wap_sell_tide  = tide.wap_sell(vol)
            wap_sell_wx    = wx.wap_sell(vol)
            wap_sell_lhr   = lhr.wap_sell(vol)

            if all(x is not None for x in
                   [wap_buy_etf, wap_sell_tide, wap_sell_wx, wap_sell_lhr]):

                wap_profit_a = (wap_sell_tide + wap_sell_wx + wap_sell_lhr) - wap_buy_etf

                if wap_profit_a > ARB_THRESHOLD:
                    # [C3] FV sanity gate — reject if market data is wrong
                    fv_ok = all([
                        self._fv_sanity_check(LON_ETF,   wap_buy_etf),
                        self._fv_sanity_check(TIDE_SPOT, wap_sell_tide),
                        self._fv_sanity_check(WX_SPOT,   wap_sell_wx),
                        self._fv_sanity_check(LHR_COUNT, wap_sell_lhr),
                    ])
                    if not fv_ok:
                        return

                    # [C2] Position headroom check
                    safe_vol = self._wap_arb_volume(
                        buys  = [(LON_ETF,   etf.ask_volume)],
                        sells = [(TIDE_SPOT, tide.bid_volume),
                                 (WX_SPOT,   wx.bid_volume),
                                 (LHR_COUNT, lhr.bid_volume)],
                    )
                    if safe_vol > 0:
                        print(f"\n[CrossArb-A] WAP profit={wap_profit_a:.1f} "
                              f"vol={safe_vol}")
                        self._execute_cross_arb(
                            buys  = [(LON_ETF,   etf.best_ask)],
                            sells = [(TIDE_SPOT, tide.best_bid),
                                     (WX_SPOT,   wx.best_bid),
                                     (LHR_COUNT, lhr.best_bid)],
                            volume = safe_vol,
                            profit = wap_profit_a * safe_vol,
                        )
                        self._last_arb_time = now
                        return

        # ── Case B: sell ETF, buy components ──────────────────────────────────
        if etf.bids and tide.asks and wx.asks and lhr.asks:

            wap_sell_etf  = etf.wap_sell(vol)
            wap_buy_tide  = tide.wap_buy(vol)
            wap_buy_wx    = wx.wap_buy(vol)
            wap_buy_lhr   = lhr.wap_buy(vol)

            if all(x is not None for x in
                   [wap_sell_etf, wap_buy_tide, wap_buy_wx, wap_buy_lhr]):

                wap_profit_b = wap_sell_etf - (wap_buy_tide + wap_buy_wx + wap_buy_lhr)

                if wap_profit_b > ARB_THRESHOLD:
                    # [C3] FV sanity gate
                    fv_ok = all([
                        self._fv_sanity_check(LON_ETF,   wap_sell_etf),
                        self._fv_sanity_check(TIDE_SPOT, wap_buy_tide),
                        self._fv_sanity_check(WX_SPOT,   wap_buy_wx),
                        self._fv_sanity_check(LHR_COUNT, wap_buy_lhr),
                    ])
                    if not fv_ok:
                        return

                    safe_vol = self._wap_arb_volume(
                        buys  = [(TIDE_SPOT, tide.ask_volume),
                                 (WX_SPOT,   wx.ask_volume),
                                 (LHR_COUNT, lhr.ask_volume)],
                        sells = [(LON_ETF,   etf.bid_volume)],
                    )
                    if safe_vol > 0:
                        print(f"\n[CrossArb-B] WAP profit={wap_profit_b:.1f} "
                              f"vol={safe_vol}")
                        self._execute_cross_arb(
                            buys  = [(TIDE_SPOT, tide.best_ask),
                                     (WX_SPOT,   wx.best_ask),
                                     (LHR_COUNT, lhr.best_ask)],
                            sells = [(LON_ETF,   etf.best_bid)],
                            volume = safe_vol,
                            profit = wap_profit_b * safe_vol,
                        )
                        self._last_arb_time = now

    def _wap_arb_volume(
        self,
        buys:  list[tuple[str, int]],
        sells: list[tuple[str, int]],
    ) -> int:
        """
        [C1, C2 fix] Compute maximum safe arb volume respecting:
          - Available book depth on each leg (no slippage beyond what we modelled)
          - Position limits on each product
        """
        vol = ORDER_VOLUME
        for product, available in buys:
            vol = min(vol, available)
            vol = min(vol, self._vol_within_limit(product, Side.BUY,  vol))
        for product, available in sells:
            vol = min(vol, available)
            vol = min(vol, self._vol_within_limit(product, Side.SELL, vol))
        return max(0, vol)

    def _execute_cross_arb(
        self,
        buys:   list[tuple[str, float]],
        sells:  list[tuple[str, float]],
        volume: int,
        profit: float,
    ) -> None:
        orders = (
            [OrderRequest(p, price, Side.BUY,  volume) for p, price in buys]  +
            [OrderRequest(p, price, Side.SELL, volume) for p, price in sells]
        )
        responses = self.send_orders(orders)
        self._arb_count     += 1
        self._total_pnl_est += profit
        filled = sum(1 for r in responses if r is not None)
        print(f"  → {len(orders)} legs  {filled} confirmed  "
              f"locked≈{profit:.1f}  arb#{self._arb_count}")

    # ── Status ─────────────────────────────────────────────────────────────────

    def _status_loop(self) -> None:
        while True:
            time.sleep(60)
            self._print_status()

    def _print_status(self) -> None:
        with self._pos_lock:    pos   = dict(self._pos)
        with self._fv_lock:     fv    = dict(self._fv)
        with self._books_lock:  books = dict(self._books)

        etf_fv = self._etf_fv()

        print(f"\n{'─'*66}")
        print(f"  STATUS  arbs={self._arb_count}  hedges={self._hedge_count}"
              f"  est_pnl≈{self._total_pnl_est:.1f}")
        print(f"{'─'*66}")
        print(f"  {'Prod':<12} {'Pos':>5} {'Reg':>8} {'FV':>8}"
              f"  {'Bid':>8} {'Ask':>8} {'Depth_B':>8} {'Depth_A':>8}")
        for p in ALL_PRODUCTS:
            b    = books.get(p)
            f    = fv.get(p, etf_fv) if p != LON_ETF else etf_fv
            pos_ = pos.get(p, 0)
            reg  = _inventory_regime(pos_)
            print(
                f"  {p:<12} {pos_:>5} {reg:>8} {f:>8.0f}"
                f"  {b.best_bid or 0:>8.1f} {b.best_ask or 0:>8.1f}"
                f"  {b.total_bid_depth:>8} {b.total_ask_depth:>8}"
                if b else
                f"  {p:<12} {pos_:>5} {reg:>8} {f:>8.0f}"
                f"  {'—':>8} {'—':>8} {'—':>8} {'—':>8}"
            )
        print(f"{'─'*66}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CMI_URL  = os.getenv("CMI_URL",  "http://localhost:8080")
    USERNAME = os.getenv("CMI_USER", "your_team_username")
    PASSWORD = os.getenv("CMI_PASS", "your_team_password")

    bot = ThinMarketETFBot(CMI_URL, USERNAME, PASSWORD)
    bot.run()