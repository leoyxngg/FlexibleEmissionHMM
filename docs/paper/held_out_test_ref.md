# Held-out predictive-density evaluation references

## Method used in this project

- Expanding-window out-of-sample evaluation with refitting every 63 trading observations.
- Fit preprocessing and HMM parameters using observations strictly before each test block.
- Keep parameters fixed within each block, but update state probabilities sequentially by filtering.
- Score each observation before incorporating it into the filter:

  ```text
  log_score_t = log p(x_t | x_1, ..., x_{t-1}; fitted parameters)
  ```

- Compare mean held-out log scores on identical dates and within the same PCA setting. Higher is better.
- A 63-day test block contains successive one-day-ahead forecasts, not a single 63-day-ahead forecast.
- Implementation: [run_heldout_regime.py](../../notebooks/sp500/run_heldout_regime.py).

## 1. Financial application: predictive distributions of asset returns

**Geweke, J., & Amisano, G. (2010).** Comparing and evaluating Bayesian predictive distributions of asset returns. *International Journal of Forecasting*, **26**(2), 216–230.

- [Accessible working-paper version, October 2009](https://fraser.stlouisfed.org/title/center-quantitative-economic-research-working-papers-7165/comparing-evaluating-bayesian-predictive-distributions-asset-returns-656235/fulltext)
- Evaluates daily S&P 500 returns using out-of-sample predictive likelihoods.
- Includes Markov normal mixture models and a hierarchical extension with normal mixtures within Markov states.
- Section 3, especially equations (10)–(12), explains cumulative predictive log scores and comparisons with non-Bayesian plug-in predictive densities.
- **Use this citation for:** the financial precedent for evaluating Markov-mixture models through out-of-sample predictive densities.
- **Difference from this project:** their principal approach integrates Bayesian parameter uncertainty. Our runner uses fitted parameter estimates frozen within each block. Their paper does not establish our particular 63-day refitting schedule.

## 2. Comparing competing density forecasts

**Amisano, G., & Giacomini, R. (2007).** Comparing density forecasts via weighted likelihood ratio tests. *Journal of Business & Economic Statistics*, **25**(2), 177–190.

- [DOI: 10.1198/073500106000000332](https://doi.org/10.1198/073500106000000332)
- Provides a framework for comparing predictive densities through weighted log-score differences.
- For an unweighted comparison of two component counts:

  ```text
  d_t = log p_M=4(x_t | past) - log p_M=3(x_t | past)
  ```

- Positive average differences favor M = 4; negative differences favor M = 3.
- **Use this citation for:** the methodology of comparing density forecasts and, when implemented, the corresponding formal statistical tests.
- **Implementation limitation:** the current runner saves predictive scores but does not implement the Amisano–Giacomini test. Reporting mean score differences is not the same as performing that test.

## 3. Statistical justification for the logarithmic score

**Gneiting, T., & Raftery, A. E. (2007).** Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, **102**(477), 359–378.

- [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Establishes the theory of proper scoring rules, including the logarithmic score.
- Under the relevant regularity conditions, the logarithmic score is strictly proper: expected score is maximized by forecasting the true distribution.
- **Use this citation for:** why held-out log predictive density is an appropriate criterion for evaluating distributional forecasts.

## Interpretation and scope

- Main financial precedent: **Geweke and Amisano (2010)**.
- Scoring-rule justification: **Gneiting and Raftery (2007)**.
- Formal density-forecast comparisons: **Amisano and Giacomini (2007)**.
- The runner scores the full transformed feature vector, including engineered indicators. Its results describe predictive fit of the feature-based HMM, not directly return-only forecasting accuracy or trading profitability.
- Within a fold, compare models using the same fitted feature transformation. Do not directly compare PCA and no-PCA log-density levels: they generally describe different-dimensional outcomes.
- Improved training scores accompanied by worse held-out scores support an overfitting interpretation. A held-out plateau supports parsimony, but does not by itself prove that extra components fit noise.
