"""
IMCity Simple Bot — Weighted Fair Value Strategy
================================================
Target: TIDE_SPOT, WX_SPOT
Strategy:
  1. Calculate Market EMA (Exponential Moving Average).
  2. Calculate Theoretical Value from API data.
  3. Blended FV = (0.8 * Market_EMA) + (0.2 * Theoretical).
  4. Trade only when spread allows a profit vs Blended FV.
"""

import time
import requests
import pandas as pd
import numpy as np
from collections import deque
from bot_template import BaseBot, OrderRequest, Side

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

SYMBOLS = ["TIDE_SPOT", "WX_SPOT"]

# Weights: How much we trust the crowd vs the data
WEIGHT_MARKET = 0.80   # Trust the market price 80%
WEIGHT_THEORY = 0.20   # Trust the API data 20%

# Safety
MAX_POS = 50           # Max contracts long or short
TRADE_SIZE = 5         # Contracts per trade
BUFFER = 15            # Minimum profit tick distance to trigger a trade
WARMUP_STEPS = 10      # How many loops to watch before trading

# Data Settings
LONDON_LAT, LONDON_LON = 51.5074, -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def get_theoretical_prices():
    """Fetches real world data to calculate theoretical price."""
    prices = {}
    
    # 1. Weather (WX_SPOT = Temp(F) * Humidity)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LONDON_LAT, "longitude": LONDON_LON,
            "current": "temperature_2m,relative_humidity_2m",
            "timezone": "Europe/London"
        }
        r = requests.get(url, params=params, timeout=2).json()
        temp_c = r["current"]["temperature_2m"]
        humid = r["current"]["relative_humidity_2m"]
        temp_f = (temp_c * 9/5) + 32
        prices["WX_SPOT"] = round(temp_f) * humid
    except Exception:
        prices["WX_SPOT"] = None

    # 2. Tides (TIDE_SPOT = abs(mAOD) * 1000)
    try:
        url = f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings"
        r = requests.get(url, params={"_limit": 1, "_sorted": ""}, timeout=2).json()
        val = r["items"][0]["value"]
        prices["TIDE_SPOT"] = abs(val) * 1000
    except Exception:
        prices["TIDE_SPOT"] = None

    return prices

# ─── THE BOT ──────────────────────────────────────────────────────────────────

class SimpleBot(BaseBot):
    def __init__(self, url, user, password):
        super().__init__(url, user, password)
        
        # State
        self.market_prices = {s: None for s in SYMBOLS}
        self.theoretical_prices = {s: None for s in SYMBOLS}
        self.emas = {s: None for s in SYMBOLS} # Exponential Moving Average
        
        # Helper to track loops
        self.loop_count = 0

    def sync_positions(self):
        """Fetch current positions from the exchange."""
        try:
            exchange_positions = self.get_positions()
            print(f"[SYNC] Positions from exchange: {exchange_positions}")
            return {s: exchange_positions.get(s, 0) for s in SYMBOLS}
        except Exception as e:
            print(f"[ERROR] Failed to fetch positions: {e}")
            return {s: 0 for s in SYMBOLS}

    def on_orderbook(self, ob):
        """Update market price based on live orderbook mid-price."""
        if ob.product not in SYMBOLS:
            return
            
        # Calculate mid price
        best_bid = ob.buy_orders[0].price if ob.buy_orders else None
        best_ask = ob.sell_orders[0].price if ob.sell_orders else None
        
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
            self.market_prices[ob.product] = mid
            
            # Update Exponential Moving Average (Smooths out noise)
            if self.emas[ob.product] is None:
                self.emas[ob.product] = mid
            else:
                # 20% new price, 80% history (Slow reaction)
                self.emas[ob.product] = (mid * 0.2) + (self.emas[ob.product] * 0.8)

    def on_trades(self, trade):
        # Log trades involving us (positions are synced from exchange)
        if trade.buyer == self.username:
            print(f"[TRADE] Bought {trade.volume} {trade.product} @ {trade.price}")
        elif trade.seller == self.username:
            print(f"[TRADE] Sold {trade.volume} {trade.product} @ {trade.price}")

    def calculate_blended_price(self, symbol):
        """The Secret Sauce: Combine Market Momentum with Data Reality."""
        market_val = self.emas[symbol]
        theory_val = self.theoretical_prices[symbol]

        if market_val is None: return None
        if theory_val is None: return market_val # Fallback to pure market if API fails

        # Formula: Blend = (Market * 0.8) + (Theory * 0.2)
        return (market_val * WEIGHT_MARKET) + (theory_val * WEIGHT_THEORY)

    def run_strategy(self):
        """Main decision loop."""
        self.loop_count += 1
        
        # 1. Update Real Data (every 5 loops / 20 seconds)
        if self.loop_count % 5 == 1:
            print("[DATA] Fetching live weather & tide data...")
            self.theoretical_prices = get_theoretical_prices()

        # 2. Warmup Period
        if self.loop_count < WARMUP_STEPS:
            print(f"[WAIT] Warming up... ({self.loop_count}/{WARMUP_STEPS}) collecting market averages.")
            return

        # 3. Sync positions from exchange before trading
        positions = self.sync_positions()

        # 4. Trade Logic
        for symbol in SYMBOLS:
            fv = self.calculate_blended_price(symbol)
            if fv is None: continue

            # Get current best prices to trade against
            ob = self.get_orderbook(symbol)
            if not ob.buy_orders or not ob.sell_orders: continue
            
            best_bid = ob.buy_orders[0].price
            best_ask = ob.sell_orders[0].price
            
            curr_pos = positions.get(symbol, 0)

            # ── BUY LOGIC ──
            # If the seller is offering cheaper than our Fair Value minus Buffer
            if best_ask < (fv - BUFFER):
                if curr_pos + TRADE_SIZE <= MAX_POS:
                    print(f"[BUY]  {symbol} @ {best_ask} | FV: {fv:.1f} (Mkt: {self.emas[symbol]:.1f} / Theo: {self.theoretical_prices[symbol]})")
                    self.send_order(OrderRequest(symbol, best_ask, Side.BUY, TRADE_SIZE))

            # ── SELL LOGIC ──
            # If the buyer is paying more than our Fair Value plus Buffer
            elif best_bid > (fv + BUFFER):
                if curr_pos - TRADE_SIZE >= -MAX_POS:
                    print(f"[SELL] {symbol} @ {best_bid} | FV: {fv:.1f} (Mkt: {self.emas[symbol]:.1f} / Theo: {self.theoretical_prices[symbol]})")
                    self.send_order(OrderRequest(symbol, best_bid, Side.SELL, TRADE_SIZE))
            
            else:
                # No trade
                print(f"[HOLD] {symbol} Mkt:{self.emas[symbol]:.0f} vs Theo:{self.theoretical_prices[symbol]} -> FV:{fv:.0f}")

    def run_forever(self):
        self.start()
        print("Bot Started. Watching market...")
        try:
            while True:
                self.run_strategy()
                time.sleep(4) # Slow loop
        except KeyboardInterrupt:
            self.cancel_all_orders()
            self.stop()

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # UPDATE THESE URLS AND CREDENTIALS
    EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
    USERNAME = "Market Fakers"
    PASSWORD = "marketfakers123"

    bot = SimpleBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()