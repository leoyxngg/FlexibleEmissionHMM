# Wang, Lin & Mikhelson (2020) - Regime-Switching Factor Investing with Hidden Markov Models

**Cite:** `@wang2020regime` ([reference.bib](reference.bib)) - _J. Risk and Financial Management_ 13(12):311, MDPI, doi:10.3390/jrfm13120311. Local PDF: `jrfm-13-00311-v2.pdf`.

## Core idea

The HMM is not used to scale exposure. It detects the market regime and **rotates the portfolio between six factor models**; each regime holds the factor model that earned the best Sharpe in that regime during training.

Factor models: leveraged long/short Fama-French (modified), modified FF (1-month price change proxy), Carhart 4-factor, long-only Value (2x), AQR quality-minus-junk style, and SPY itself as the benchmark model.

## HMM configuration

| setting | value |
| --- | --- |
| library | `hmmlearn.GaussianHMM` |
| asset | SPY (S&P 500 ETF), daily OHLC |
| features (2) | daily return = simple close-to-close % change; "volatility" = mean-squared deviation of close from its 10-day moving average (price units) |
| states | K = 3: bull (1), sideways "kangaroo" (0), bear (2) |
| K selection | raw log-likelihood improvement K=2 -> K=3 (Fig. 3); no AIC/BIC |
| covariance | full (keeps return-volatility correlation) |
| EM | max_iter = 75 |
| fitted transitions | sticky, diagonals ~0.90-0.94; average regime duration ~12.5 days |
| fitted regimes (Table 3) | bull: 44.6% of days, +0.046%/day, vol 0.94; sideways: 42.4%, +0.040%/day, vol 3.47; bear: 13%, -0.066%/day, vol 13.6 |

State labels assigned post-hoc by fitted return/vol ranking (same convention as `high_volatility_state` in our `market_strategy.signals`).

## Periods

- **In-sample (Jan 2007 - Sep 2017, ~10.5y):** backtest all six factor models, train HMM, compute per-regime Sharpe per model, freeze the regime -> best-model map.
- **Out-of-sample (Sep 2017 - Apr 2020, ~2.6y):** trade the rotation; includes Dec 2018 and Mar 2020 crashes.
- **Live mode:** sliding-window retrain **every day** before open on the most recent **2707 days** (~10.5y).

## Regime detection mechanism (idiosyncratic - do not copy)

They ignore the HMM posterior. Per regime they KS-fit the historical (return, vol) observations to a standard distribution family (normal/lognormal/pareto/gamma/beta/exponential), then for each new day compare the fitted **PDF values**; a regime is declared when vol-PDF > 0.3 **and** return-PDF > 0.5.

Problems: densities are not probabilities (can exceed 1); the 0.3/0.5 thresholds are ad hoc; the KS choice discards the joint (return, vol) model the HMM just estimated. The canonical mechanism is the one-step-ahead forward filter `P(S_t | data < t)` with a probability threshold - implemented in `src/model_evaluation/market_strategy/signals.py`.

## Backtest protocol

- Platform: QuantConnect; daily evaluation, switch model when the detector fires.
- Metric stack: Sharpe (rf = 10y Treasury), information ratio vs SPY, Treynor, Treynor-Mazuy gamma (timing: excess return regressed on (mkt, mkt^2)), total return, max drawdown.
- Sanity check: regression of HMM-model return on French/Frazzini factors (MKT, SMB, HML, MOM, QMJ) to show residual alpha.

## Headline results (OOS Sep 2017 - Apr 2020)

| model               | Sharpe                | IR       | max DD    |
| ------------------- | --------------------- | -------- | --------- |
| **HMM rotation**    | **2.017**             | **1.64** | **0.128** |
| Value               | 0.463                 | 0.989    | 0.536     |
| Modified FF         | 0.208                 | 0.249    | 0.131     |
| SPY                 | -0.174                | -        | 0.341     |
| FF3 / Carhart / AQR | -1.42 / -0.67 / -1.42 | neg      | 0.28-0.30 |

Alpha ~2%/yr over common factors (t = 3.31). Treynor-Mazuy gamma NOT statistically significant (their own admission) - timing skill is not established by that test.

## Important caveats

1. **One event drives the result:** the OOS edge largely comes from rotating out of 2x Value during Mar 2020. 2.6 years, one crash - the win is not distributed.
2. **K chosen by raw likelihood elbow** - our AIC/BIC (hmmlearn `model.aic()/bic()`) is the stricter selection.
3. **"Volatility" feature in price units** (MSE of close vs 10d MA): non-stationary, scales with the index level. A return-based rolling std is cleaner.
4. **No transaction costs reported** for factor-model turnover (rebalance/leverage switching is expensive).
5. **No multiple-restart reporting** for Baum-Welch (EM local optima unaddressed).
