"""Causal two-regime emission and PCA experiment on S&P index prices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from flexible_emission_hmm import (
    BaumWelchTrainer,
    GaussianEmission,
    GaussianMixtureEmission,
    HMM,
)
from flexible_emission_hmm.emissions import EmissionModel
from flexible_emission_hmm.feature import regime_price_features
from flexible_emission_hmm.validation import as_sequence


class MultivariateStudentTEmission(EmissionModel):
    """Full-covariance multivariate Student-t emissions with state-specific df."""

    def __init__(self, n_states=2, random_state=None, min_variance=1e-5):
        super().__init__(n_states)
        self.random_state = random_state
        self.min_variance = min_variance
        self.means = None
        self.covariances = None
        self.dof = None

    def _regularize(self, covariance):
        values, vectors = np.linalg.eigh(covariance)
        return (vectors * np.maximum(values, self.min_variance)) @ vectors.T

    def initialize(self, X):
        """Initialize locations from observations and scales from pooled data."""
        X = as_sequence(X)
        self.n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_states, replace=len(X) < self.n_states)
        self.means = X[indices].copy()
        cov = np.cov(X, rowvar=False, bias=True)
        cov = np.atleast_2d(cov)
        self.covariances = np.repeat(self._regularize(cov)[None],
                                     self.n_states, axis=0)
        self.dof = np.full(self.n_states, 10.0)

    def _distance_and_logdet(self, X, state):
        factor = np.linalg.cholesky(self.covariances[state])
        centered = np.linalg.solve(factor, (X - self.means[state]).T)
        return (centered**2).sum(axis=0), 2 * np.log(np.diag(factor)).sum()

    def log_prob(self, X):
        """Compute per-state log densities for each multivariate observation."""
        X = as_sequence(X, self.n_features)
        D = self.n_features
        result = np.empty((len(X), self.n_states))
        for k in range(self.n_states):
            nu = self.dof[k]
            delta, logdet = self._distance_and_logdet(X, k)
            result[:, k] = (gammaln((nu + D) / 2) - gammaln(nu / 2)
                            - 0.5 * (D * np.log(nu * np.pi) + logdet)
                            - 0.5 * (nu + D) * np.log1p(delta / nu))
        return result

    def m_step(self, X, responsibilities):
        """Update locations, scatter, and df with latent Gamma precision weights."""
        X = as_sequence(X, self.n_features)
        weights = np.asarray(responsibilities)
        D = self.n_features
        for k in range(self.n_states):
            w = weights[:, k]
            mass = w.sum()
            if mass <= np.finfo(float).eps:
                continue
            nu = self.dof[k]
            delta, _ = self._distance_and_logdet(X, k)
            u = (nu + D) / (nu + delta)
            log_u = digamma((nu + D) / 2) - np.log((nu + delta) / 2)
            self.means[k] = (w * u) @ X / (w * u).sum()
            centered = X - self.means[k]
            self.covariances[k] = self._regularize(
                (centered.T * (w * u)) @ centered / mass)
            average = w @ (log_u - u) / mass
            def score(candidate):
                return (np.log(candidate / 2) + 1
                        - digamma(candidate / 2) + average)
            self.dof[k] = (2.1 if score(2.1) <= 0 else
                           200.0 if score(200.0) >= 0 else
                           brentq(score, 2.1, 200.0))


@dataclass
class DailyFit:
    """One date's causal signal and fit diagnostics."""

    date: pd.Timestamp
    model: str
    pca: bool
    bull_probability: float
    bull_raw: int
    log_likelihood: float
    n_components: int
    converged: bool
    iterations: int
    selected_seed: int


def build_observations(prices, windows=(5, 10, 60), half_lives=(5, 10, 60)):
    """Build price-only paper-inspired features without discarding warm-up dates."""
    return regime_price_features(prices, windows=windows, half_lives=half_lives)


def label_bull(model, X, returns):
    """Name the state with larger training-period return-weighted posterior bull."""
    _, filtered, _ = model.filter(X)
    mass = filtered.sum(axis=0)
    state_returns = (filtered.T @ np.asarray(returns)) / mass
    return int(np.argmax(state_returns))


def fit_one(X, returns, model_name, seed, max_iter=75):
    """Fit one locally implemented HMM and name states using training returns."""
    if model_name == "Gaussian":
        emission = GaussianEmission(2, random_state=seed)
    elif model_name == "GMM":
        emission = GaussianMixtureEmission(2, n_components=2, random_state=seed)
    elif model_name == "Student-t":
        emission = MultivariateStudentTEmission(2, random_state=seed)
    else:
        raise ValueError(f"Unknown emission: {model_name}")
    model = HMM(emission, trainer=BaumWelchTrainer(
        max_iter=max_iter, tol=len(X) * 1e-5)).fit(X)
    return model, label_bull(model, X, returns)


def fit_day(features, simple_returns, date, years=10, use_pca=False,
            seeds=(0, 1), max_iter=75):
    """Fit on a trailing calendar window and issue a decision for the next day.

    Scaler, PCA, state labels, and HMM parameters use observations through
    ``date`` only. The final filtered probability uses no later observation.
    """
    cutoff = pd.Timestamp(date)
    start = cutoff - pd.DateOffset(years=years)
    window = features.loc[(features.index > start) & (features.index <= cutoff)]
    window = window.dropna()
    returns = simple_returns.reindex(window.index)
    if len(window) < years * 200 or returns.isna().any():
        raise ValueError("Insufficient complete training observations")
    X = StandardScaler().fit_transform(window)
    if use_pca:
        X = PCA(n_components=0.95, svd_solver="full").fit_transform(X)
    records = []
    for name in ("Gaussian", "GMM", "Student-t"):
        fits = []
        for seed in seeds:
            try:
                model, bull = fit_one(X, returns.to_numpy(), name,
                                      seed=seed, max_iter=max_iter)
                likelihood = model.score(X)
                if np.isfinite(likelihood):
                    fits.append((likelihood, model, bull, seed))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if not fits:
            raise RuntimeError(f"All {name} fits failed on {date}")
        likelihood, model, bull, selected_seed = max(fits, key=lambda item: item[0])
        # Current state probabilities condition on the observation at date.
        # The next-day predicted prior is used as the trading forecast.
        _, filtered, _ = model.filter(X)
        predicted_next = filtered[-1] @ model.states.transitions
        records.append(DailyFit(cutoff, name, use_pca,
                                float(predicted_next[bull]),
                                int(predicted_next[bull] >= 0.5),
                                float(likelihood), X.shape[1],
                                model.trainer.converged, model.trainer.n_iter,
                                selected_seed))
    return records


def rolling_experiment(prices, start, end, years=10, seeds=(0, 1),
                       max_iter=75, pca_options=(False, True),
                       windows=(5, 10, 60), half_lives=(5, 10, 60)):
    """Generate each next-day decision with a fresh trailing-window fit."""
    features = build_observations(prices, windows=windows, half_lives=half_lives)
    returns = prices.pct_change(fill_method=None)
    dates = features.loc[start:end].index
    records = []
    for date in dates:
        for use_pca in pca_options:
            records.extend(fit_day(features, returns, date, years=years,
                                   use_pca=use_pca, seeds=seeds, max_iter=max_iter))
    return pd.DataFrame([vars(record) for record in records])


def evaluate_signals(fits, prices, median_window=12):
    """Compare 0/1 positions with buy-and-hold on common next-day dates.

    Even-sized rolling medians can equal 0.5; ties preserve the previous
    position (initially cash). Cash earns zero, and costs are excluded.
    """
    prices = prices.astype(float)
    next_returns = prices.pct_change(fill_method=None).shift(-1)
    if not isinstance(median_window, int) or median_window < 1:
        raise ValueError("median_window must be a positive integer")
    results, weights = {}, {}
    for (name, pca), group in fits.groupby(["model", "pca"]):
        group = group.sort_values("date").set_index("date")
        raw = group["bull_raw"].astype(float)
        median = raw.rolling(median_window, min_periods=1).median()
        exposure = median.where(median != 0.5).ffill().fillna(0).astype(int)
        key = f"{name} / {'PCA' if pca else 'no PCA'}"
        payoff = next_returns.reindex(exposure.index)
        if payoff.isna().any():
            raise ValueError("A decision has no next-day return")
        results[key] = np.log1p(exposure * payoff)
        weights[key] = exposure
    if not results:
        raise ValueError("No model signals were supplied")
    common_dates = next(iter(results.values())).index
    if any(not common_dates.equals(series.index) for series in results.values()):
        raise ValueError("Model decisions must share the same dates")
    results["Buy and hold"] = np.log1p(next_returns.reindex(common_dates))
    weights["Buy and hold"] = pd.Series(1, index=common_dates)
    return results, weights
