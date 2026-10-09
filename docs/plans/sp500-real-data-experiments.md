# S&P 500 real-data HMM experiment plan

Status: in progress. Created October 1, 2026; Gaussian development baseline implemented October 2, 2026.

## Objective

Evaluate whether a regime-switching HMM provides useful predictive distributions,
interpretable market regimes, and improved trading performance on real S&P 500
data. Separate these three objectives: good density forecasts do not necessarily
produce a profitable strategy, and plausible regime plots do not establish
predictive performance.

Use `notebooks/gaussian_synthetic_experiments.ipynb` as the experimental template.
Unlike the synthetic experiment, real data have no known emission density or
true hidden-state labels. Prelabeling the test set is not required. Any externally
defined regimes are reference groups, not ground truth.

## 1. Adapt Leo's experiment

Retain multiple initializations, selection by training likelihood, convergence
diagnostics, deterministic seeds, parameter exports, and run manifests.

| Synthetic experiment | Real-data replacement |
| --- | --- |
| Known generating distributions | Unknown distribution of market returns |
| Independent simulated replicates | Chronological evaluation periods |
| True-state alignment and accuracy | Regime characteristics, stability, and reference-group agreement |
| KL against the true density | Out-of-sample predictive log loss and calibration |
| Error against the true transition matrix | Estimated persistence and stability across refits |
| Smoothed test-state inference | Forward filtering for predictions and trading |

Random starts measure optimization sensitivity; they are not independent market
experiments. Do not use the synthetic evaluator's true-label requirements as the
interface for real-data evaluation.

## 2. Data and features

The current processed file, `data/processed/sp500_index_price.csv`, contains
15,982 observations from July 2, 1962 through December 31, 2025. It contains
price-index levels and price returns. The corresponding raw file's total-return
fields are entirely missing as of this plan's creation.

- Sort dates, check duplicates and missing observations, and verify returns.
- Preserve extreme observations unless they are confirmed data errors.
- Model daily log returns. Use simple returns for portfolio accounting.
- Obtain dividend-inclusive index or tradable-proxy returns and an aligned cash
  return series for the trading evaluation. Until then, label results explicitly
  as a price-index backtest. Record the instrument and its actual coverage.
- Fit all learned transformations on the current training window only.
- Use historical observations before each split to warm up rolling features;
  past information is allowed. Do not use centered windows or future values.

Run two separately identified configurations:

| Configuration | Purpose |
| --- | --- |
| Primary: daily log return only | Compare emission distributions on a common scalar target |
| Secondary: return, 60-day volatility, 20-day mean return | Assess engineered features for regime interpretation and trading |

Use the functions in `src/flexible_emission_hmm/feature.py`; retain configurable
window lengths. Rolling features overlap and can create apparent persistence
through smoothing. Their dependence also makes a conventional joint-emission
model an approximation. Do not compare one-dimensional and three-dimensional
joint likelihoods directly. Cross-feature comparisons must use the same return
forecast target or the same trading outcomes, with the forecasting construction
specified explicitly.

## 3. Chronological development and final evaluation

Proposed schedule, subject to data availability and prior use of the holdout:

| Stage | Period | Purpose |
| --- | --- | --- |
| Development | 1990–2015 | Select configuration and strategy using walk-forward validation |
| Final evaluation | 2016–2025 | Evaluate the frozen procedure |
| Historical sensitivity | Earlier available history | Assess dependence on the modern sample |

For each evaluation year:

1. Fit on the preceding ten years.
2. Freeze fitted parameters for the coming year.
3. Forecast and update state probabilities sequentially throughout that year.
4. Advance one year and refit using only information available at that cutoff.

Example: train on 2006–2015 and evaluate 2016, then train on 2007–2016 and
evaluate 2017. Earlier evaluation observations may enter later training windows
once they are historical. Hyperparameters and trading rules remain fixed during
the final evaluation.

Use five- and twenty-year windows as development sensitivity checks. Do not
select the training window using final-test Sharpe. Do not randomly shuffle
observations. If forward-horizon targets are introduced, exclude training targets
whose realization extends beyond the training cutoff. Trailing feature windows
alone do not require discarding valid past context at the boundary.

If the proposed final period has already been used extensively for tuning,
describe it as retrospective evaluation rather than an untouched holdout.

## 4. Model fitting and selection

Initial configuration:

- Three-state Gaussian HMM.
- Ten deterministic random initializations per training window.
- Maximum 300 EM updates.
- Per-observation log-likelihood tolerance of 1e-6, following Leo's notebook.
- Choose the initialization with the highest finite training likelihood.
- Record all attempts, including failures, warnings, convergence, iterations,
  likelihood history, and fitted parameters.

Use development-period predictive performance to select model configurations.
Investigate state counts as a secondary sensitivity analysis, keeping the initial
comparison small and recording all attempted configurations.

After establishing the Gaussian baseline, implement a fitted Student-t emission
model, then consider Gaussian-mixture emissions. Leo's non-Gaussian generators
do not by themselves supply the corresponding fitted emission models.

## 5. Forward-only inference

The current `HMM.predict_proba()` uses forward–backward smoothing and
`HMM.predict()` uses Viterbi decoding. Calling them on the entire test sequence
can use later observations to interpret earlier dates. Keep those methods for
retrospective analysis; add a filtering interface for forecasting and trading.

For each date t, maintain:

    alpha_t(k) = P(z_t = k | x_1, ..., x_t)
    q_(t+1) = alpha_t @ A

For the return-only model:

    p(r_(t+1) | r_1, ..., r_t) = sum_k q_(t+1)(k) * f_k(r_(t+1))

Save the predictive density before observing the next return, then update the
filter. Initialize the evaluation period using the training-end filtered state
distribution under the fitted model. Do not reset the filter every day. At each
refit, recompute the relevant historical filter under the new model.

Required checks:

- Altering future observations cannot change earlier forecasts or positions.
- Sequential filtering agrees with the final posterior of each observed prefix
  under fixed parameters.
- State and predictive probabilities remain normalized and finite.
- Train/evaluation boundaries preserve the intended state distribution.

## 6. Predictive evaluation without true labels

| Metric | Purpose |
| --- | --- |
| Mean one-step negative log likelihood | Primary score for probability assigned to future returns |
| 50%, 90%, and 95% interval coverage and width | Calibration and sharpness |
| Probability integral transform diagnostics | Distributional bias and remaining temporal dependence |
| 1% and 5% downside-quantile exceedances | Tail-risk calibration and clustering of violations |

Include a single-state Gaussian and a simple time-varying-volatility density
forecast as benchmarks. Compare models on identical dates, targets, and return
units. Convert standardized density scores to original units, including the
transformation Jacobian.

Report results by evaluation period and on the combined out-of-sample sequence.
Use paired temporal blocks to quantify uncertainty in model score differences;
do not treat daily observations or adjacent evaluation periods as independent
replicates. There is no directly observable real-data analogue of true-emission
KL, state accuracy, or true-transition RMSE.

## 7. Regime interpretation and conditional performance

### Model-defined regimes

Assign descriptive state names from training-period characteristics, such as
low, medium, and high return volatility. Do not assume three states necessarily
mean bull, bear, and sideways. Reordering states for reporting does not establish
that their economic meanings remain identical across refits.

Report occupancy, duration, return distribution, downside risk, transition
probabilities, posterior uncertainty, and subsequent strategy performance.

For predictive claims, condition on the regime information available before the
evaluated return. Grouping a return by a state inferred using that same return
is descriptive, not predictive. Smoothed regime plots must be labeled as
retrospective and must never drive the trading simulation.

### Common external reference groups

Define a shared set of market conditions, for example high/low trailing
volatility crossed with positive/negative trailing momentum. Determine volatility
thresholds from training data and freeze them for the evaluation year. Use only
information available at the forecast or trading decision time.

These groups allow comparisons across models on identical dates. They are not
true hidden-state labels; agreement with a reference derived from an input
feature is not independent validation. Historical crises can be included as
predeclared retrospective case studies.

Always report observation counts and uncertainty for conditional results.

## 8. Trading policy and Sharpe ratio

A trading strategy, not an HMM alone, has a trading Sharpe ratio. Freeze the
forecast-to-position rule before final evaluation.

First strategy: long-only volatility targeting with a 10% annual target and no
leverage:

    target_equity_weight = min(1, 0.10 / forecast_annual_volatility)

The remaining allocation earns the cash return. Define handling of zero or
invalid forecast volatility explicitly. Compute predictive variance from the
full mixture, including variation between state means, rather than merely
averaging state variances.

Benchmarks:

- Buy-and-hold.
- A constant equity/cash allocation fixed during development.
- The same volatility-targeting rule using rolling or EWMA volatility.

Use identical dates, instruments, cash returns, execution assumptions, and cost
accounting. With the current close-only data, use a conservative execution lag:
observe Monday's close, execute Tuesday's close, and earn the subsequent
close-to-close return. Match the forecast horizon to this delay, using the
appropriate additional state-transition propagation. An alternative next-open
execution design requires suitable open-price data and explicit interval
accounting.

Charge transaction costs on actual traded notional, accounting for portfolio
weight drift. Report gross returns and predefined cost sensitivities of 1, 5,
and 10 basis points per unit of one-way turnover; these are experimental
assumptions, not claims about achievable trading costs.

Compute daily net simple excess returns and conventional annualized Sharpe:

    excess_t = net_strategy_return_t - cash_return_t
    annualized_sharpe = sqrt(252) * mean(excess) / sample_std(excess)

Report undefined Sharpe when the denominator is zero. Also report CAGR,
maximum drawdown, annualized volatility, turnover, and average equity exposure.
Examine serial dependence and provide block-bootstrap uncertainty; square-root
annualization is an approximation when returns are serially correlated.

Calculate headline Sharpe from the stitched chronological out-of-sample series,
not the average of yearly Sharpes. Preserve portfolio continuity and charge
turnover at refits. For regime-specific reporting, show conditional return/risk
statistics and sample sizes. Do not treat disconnected regime days as a
continuously invested annual strategy.

## 9. Notebook and artifacts

Create `notebooks/sp500_real_experiments.ipynb` with these sections:

1. Data audit and experiment specification.
2. Features and chronological splits.
3. Model fitting and initialization diagnostics.
4. Sequential forecasts and filtered probabilities.
5. Predictive evaluation.
6. Regime interpretation and conditional performance.
7. Trading backtest and benchmarks.
8. Uncertainty and sensitivity checks.
9. Artifact export and conclusions.

Suggested reusable components:

- Forward filtering and predictive scoring in the HMM/inference modules.
- A real-data evaluation module separate from the synthetic true-label evaluator.
- Walk-forward orchestration and portfolio accounting helpers.

Export a manifest, data/source hashes, settings, seeds, split dates, fitted
transformations, parameters, attempt diagnostics, and metrics. Save one row per
forecast date containing the training cutoff, model identifier, predictive
parameters, realized return, filtered and predicted regime probabilities,
reference group, position, turnover, costs, cash return, and net portfolio return.

Generate all tables and figures from the same identified saved run. Avoid mixing
outputs from different configurations or executions, as occurred in the saved
synthetic notebook summaries.

## 10. Implementation milestones

- [ ] Audit price data and obtain total-return and cash series.
- [ ] Freeze the development/final split and document prior holdout use.
- [x] Implement and verify forward filtering and predictive scoring.
- [ ] Run the return-only, three-state Gaussian baseline with ten-year training
  windows, ten initializations, and annual refits.
- [ ] Add predictive benchmarks, calibration plots, and regime reporting.
- [x] Implement the lagged volatility-targeting backtest and benchmark strategies.
- [x] Verify execution timing, cash accrual, turnover, cost deductions, and
  stitched out-of-sample metrics using controlled examples.
- [ ] Add block-bootstrap uncertainty and development sensitivity analyses.
- [ ] Implement and compare Student-t emissions; consider mixtures afterward.
- [ ] Evaluate the three-feature configuration as a separate experiment.
- [ ] Freeze the chosen procedure, execute final evaluation, and export one
  internally consistent report and artifact set.

First milestone deliverable: a defensible Gaussian baseline with sequential
out-of-sample density evaluation and a cost-aware, lagged trading comparison.
Success is reliable evaluation; improved Sharpe is an empirical outcome, not an
assumed requirement.

The notebook now contains an executable development-period Gaussian baseline,
with a one-year smoke run completed using one start and two EM updates. The full
ten-start, 1990–2015 run has not yet been executed. Trading currently assumes
zero cash returns and uses price-index returns because aligned cash and
dividend-inclusive series have not been obtained. The 2016–2025 period remains
reserved; prior use of that period still needs to be documented before calling
it an untouched holdout.

## References

- [Leo's synthetic experiment notebook](../../notebooks/gaussian_synthetic_experiments.ipynb)
- [Time-series cross-validation — Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html)
- [Preprocessing and data leakage — scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html)
- [Gneiting and Raftery: Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
- [William F. Sharpe: The Sharpe Ratio](https://www-leland.stanford.edu/~wfsharpe/art/sr/sr.htm)
