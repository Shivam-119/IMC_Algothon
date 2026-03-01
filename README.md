# IMCity Algothon 2026

## The Challenge

IMCity is a 24-hour live trading competition run on a simulated exchange. Eight products settle at 12pm London time based on real-world data collected during the session window:

- **TIDE_SPOT** and **TIDE_SWING**: derived from tidal level readings at the Westminster gauge on the Thames
- **WX_SPOT** and **WX_SUM**: derived from London weather (temperature and humidity via Open-Meteo)
- **LHR_COUNT** and **LHR_INDEX**: derived from Heathrow arrivals and departures (via AeroDataBox)
- **LON_ETF**: a basket combining the three spot products
- **LON_FLY**: an options structure on the ETF

The goal is to build a bot that trades profitably against other teams on this exchange over the course of the 24 hours.

## This Approach

We split the problem into two parts: building accurate fair-value estimates for each product, and then using those estimates to trade.

### Fair Value Models

For the tidal products, we fit a harmonic regression model to historical Westminster gauge data from the EA Flood Monitoring API. Tidal levels follow well-understood periodic patterns driven by gravitational forcing, so they were modelled as a sum of sinusoidal components at the dominant tidal frequencies: M2 (12.42h), K1 (23.93h), M4 (6.21h) and MK3 (8.18h). We started with just M2 and K1 but added M4 and MK3 after noticing systematic residuals at a ~6 hour period, which is characteristic of the Thames estuary's shallow-water geometry distorting the tidal shape. This brought the model R-squared from 0.95 to 0.97.

For prediction intervals, rather than relying on the in-sample OLS residual variance (which tends to underestimate real forecast uncertainty), we calibrated the interval width empirically using walk-forward holdout windows at different prediction horizons. This means short-range predictions get a tighter interval and longer-range ones get a wider one, which better reflects how tidal forecast error grows with lead time.

For TIDE_SWING (which pays off based on 15-minute changes in tidal level), we combined known historical payoffs for intervals that had already occurred with Monte Carlo simulation over the remaining intervals, using the same tidal model to generate scenario paths.

For weather, we fetch 15-minute forecast and observation data from Open-Meteo (free, no API key required) and compute the expected settlement values directly from the temperature and humidity series.

For flights, we use the AeroDataBox API to fetch arrivals and departures across the session window and compute LHR_COUNT and LHR_INDEX from those. If no API key is available the bot falls back to the exchange starting price with a conservative confidence interval.

For the derived products (LON_ETF and LON_FLY), we propagate uncertainty from the three component products through Monte Carlo simulation to get a distribution over the ETF, then price the options structure analytically against that distribution.

### Trading Bot

The bot (`src/bot.py`) connects to the exchange using the provided `BaseBot` framework and runs two strategies simultaneously:

1. **Market making**: it posts a bid and ask around its fair value estimate, with a spread width proportional to the uncertainty in that estimate. Products where we have high confidence (tidal) get tighter spreads, and products where we are less sure (flights, the fly structure) get wider ones. Quotes are only refreshed when the market mid moves by at least one tick, to preserve queue priority.

2. **Directional taking**: if the best price in the market is more than 15 ticks away from our fair value, the bot sends an immediate-or-cancel style order to capture that edge before requoting.

Position risk is managed with a hard cap of 12 contracts per product on each side.

Fair values are refreshed from live data every 60 seconds in a background thread, and the bot listens to the exchange SSE stream to react to orderbook changes in real time, to account for a relatively volatile market.

## Why This Method

The tidal products were the most tractable because tidal physics is well understood and the data is publicly available at high resolution. A harmonic regression model is a standard approach in physical oceanography and fits well here because the signal is genuinely periodic and the noise is relatively small compared to the amplitude. It also gives us a natural way to quantify uncertainty through the residual structure.

For the other products we used simpler data-driven approaches because the relationships are more direct (temperature times humidity, flight counts) and there is less room to build a sophisticated model in a 24-hour competition.

The trading logic is deliberately straightforward. A market-making strategy with data-driven fair values is a good baseline for this kind of competition because it generates profit from spread capture while also giving directional exposure when the market is clearly mispriced relative to our estimates. We did not pursue more complex strategies like cross-product arbitrage or order-book momentum, though those would be natural next steps.

## Running the Bot

```bash
cp .env.example .env
# Edit .env: fill in CMI_URL, CMI_USER, CMI_PASS (and optionally RAPIDAPI_KEY)
bash run_bot.sh
```

## Files

```
src/
  tide_predictor.py       Tidal harmonic model, TIDE_SPOT and TIDE_SWING predictions
  bot.py                  Full trading bot (all 8 products)
  test_tide_predictor.py  Unit and integration tests for the tidal model
algothon-templates/
  bot_template.py         Exchange framework template (BaseBot, OrderBook, etc.)
  example.ipynb           Official challenge walkthrough notebook
.env.example              Credentials template
run_bot.sh                Launch script
```
