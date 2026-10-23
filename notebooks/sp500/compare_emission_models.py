"""Student-t emission and restart fitting for the S&P 500 emission comparison."""

import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

from flexible_emission_hmm import (
    HMM, BaumWelchTrainer, GaussianEmission, GaussianMixtureEmission,
)
from flexible_emission_hmm.emissions import EmissionModel
from flexible_emission_hmm.validation import as_sequence

NU_MIN, NU_MAX = 2.1, 200.0
"""Degrees-of-freedom bounds; the lower bound keeps state variances finite."""


## Student-t emission


class StudentTEmission(EmissionModel):
    """Univariate Student-t emission with a location, scale, and df per state."""

    def __init__(self, n_states, min_variance=1e-10, random_state=None):
        super().__init__(n_states)
        if not np.isfinite(min_variance) or min_variance <= 0:
            raise ValueError("min_variance must be finite and positive")
        self.min_variance = min_variance
        self.random_state = random_state
        self.means = None
        self.scales_squared = None
        self.dof = None

    def initialize(self, X):
        """Start at random observations with the pooled scale and 5 df."""
        X = as_sequence(X)
        if X.shape[1] != 1:
            raise ValueError("StudentTEmission supports one feature")
        self.n_features = 1
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_states, replace=len(X) < self.n_states)
        self.means = X[indices, 0].copy()
        self.scales_squared = np.full(self.n_states, max(X.var() * 0.6, self.min_variance))
        self.dof = np.full(self.n_states, 5.0)

    def check_data(self, X):
        """Return ``X`` as a ``(T, 1)`` array after initialization."""
        if self.means is None:
            raise ValueError("Initialize the emission model before using it")
        return as_sequence(X, 1)

    def log_prob(self, X):
        """Return log p(x_t | state k), shaped ``(T, K)``."""
        x = self.check_data(X)[:, 0:1]
        nu, s2 = self.dof, self.scales_squared
        z2 = (x - self.means) ** 2 / s2
        return (
            gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * s2)
            - (nu + 1) / 2 * np.log1p(z2 / nu)
        )

    def m_step(self, X, responsibilities):
        """Run one EM update using the latent Gamma scale weights."""
        x = self.check_data(X)[:, 0]
        weights = np.asarray(responsibilities, dtype=float)
        if weights.shape != (len(x), self.n_states) or not np.isfinite(weights).all():
            raise ValueError("responsibilities must be finite weights of shape (T, K)")
        for k in range(self.n_states):
            w = weights[:, k]
            mass = w.sum()
            if mass <= np.finfo(float).eps:
                continue
            nu = self.dof[k]
            z2 = (x - self.means[k]) ** 2 / self.scales_squared[k]
            u = (nu + 1) / (nu + z2)
            log_u = digamma((nu + 1) / 2) - np.log((nu + z2) / 2)
            self.means[k] = (w * u) @ x / (w * u).sum()
            scale = (w * u) @ (x - self.means[k]) ** 2 / mass
            self.scales_squared[k] = max(scale, self.min_variance)
            average = (w @ (log_u - u)) / mass
            self.dof[k] = _solve_dof(average)

    def variances(self):
        """Return each state's variance, ``scale² · ν / (ν − 2)``."""
        return self.scales_squared * self.dof / (self.dof - 2)


def _solve_dof(average):
    """Solve the degrees-of-freedom score equation within the allowed bounds."""
    def score(nu):
        return np.log(nu / 2) + 1 - digamma(nu / 2) + average

    if score(NU_MIN) <= 0:
        return NU_MIN
    if score(NU_MAX) >= 0:
        return NU_MAX
    return brentq(score, NU_MIN, NU_MAX)


## Restart fitting


def build_emission(name, n_states, seed, n_mix=2):
    """Create one named candidate emission; ``name`` is a key of the caller's table."""
    if name == "Gaussian":
        return GaussianEmission(n_states, random_state=seed)
    if name.startswith("GMM"):
        return GaussianMixtureEmission(n_states, n_components=n_mix, random_state=seed)
    if name == "Student-t":
        return StudentTEmission(n_states, random_state=seed)
    raise ValueError(f"Unknown emission: {name}")


def parameter_count(name, n_states, n_mix=2):
    """Count free parameters: start, transitions, and emission parameters."""
    shared = (n_states - 1) + n_states * (n_states - 1)
    if name == "Gaussian":
        return shared + 2 * n_states
    if name.startswith("GMM"):
        return shared + n_states * (3 * n_mix - 1)
    if name == "Student-t":
        return shared + 3 * n_states
    raise ValueError(f"Unknown emission: {name}")


def fit_restarts(name, X, n_states=3, n_mix=2, n_starts=10, max_iter=500,
                 tol_per_observation=1e-6):
    """Fit every restart and return all fits, with the best by training likelihood.

    Args:
        name: Candidate emission name.
        X: Training observations shaped ``(T, 1)``.
        n_states: Number of hidden states.
        n_mix: Mixture components per state for GMM candidates.
        n_starts: Deterministic seeds ``0 .. n_starts - 1``.
        max_iter: Maximum EM updates per restart.
        tol_per_observation: Stop when the likelihood gain is below this times ``T``.

    Returns:
        A list of attempt records and the index of the highest-likelihood fit.

    Raises:
        RuntimeError: Every restart failed.
    """
    attempts, best = [], None
    for seed in range(n_starts):
        record = {"seed": seed, "model": None, "log_likelihood": np.nan,
                  "converged": False, "iterations": 0, "seconds": np.nan,
                  "likelihood_decreases": 0, "error": None}
        began = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = HMM(
                    build_emission(name, n_states, seed, n_mix),
                    trainer=BaumWelchTrainer(
                        max_iter=max_iter, tol=len(X) * tol_per_observation),
                )
                model.fit(X)
            history = np.asarray(model.history)
            if not np.isfinite(history).all():
                raise FloatingPointError("Nonfinite likelihood history")
            record.update(
                model=model, log_likelihood=float(history[-1]),
                converged=bool(model.trainer.converged),
                iterations=model.trainer.n_iter,
                likelihood_decreases=int(np.sum(np.diff(history) < -1e-6)),
            )
            if best is None or record["log_likelihood"] > attempts[best]["log_likelihood"]:
                best = len(attempts)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - began
        attempts.append(record)
    if best is None:
        raise RuntimeError(f"All {n_starts} restarts failed for {name}")
    return attempts, best


def student_t_from_gaussian(gaussian, X, max_iter=1000, tol_per_observation=1e-6):
    """Continue EM as a Student-t HMM from a fitted Gaussian solution.

    The Student-t family contains the Gaussian as ``dof -> inf``, so its best
    likelihood cannot be lower. Random starts can miss that region, so this
    start copies the Gaussian transitions and state moments, with large df.

    Args:
        gaussian: Fitted Gaussian ``HMM`` with one feature.
        X: Training observations shaped ``(T, 1)``.
        max_iter: Maximum EM updates.
        tol_per_observation: Stop when the likelihood gain is below this times ``T``.

    Returns:
        The fitted Student-t ``HMM`` and its likelihood history.
    """
    n_states = gaussian.n_states
    emission = StudentTEmission(n_states)
    emission.initialize(X)
    emission.means = gaussian.emission.means[:, 0].copy()
    emission.scales_squared = gaussian.emission.covariances[:, 0, 0].copy()
    emission.dof = np.full(n_states, NU_MAX)
    model = HMM(emission)
    model.states.start = gaussian.states.start.copy()
    model.states.transitions = gaussian.states.transitions.copy()
    model.fitted = True
    history = [model.posterior(X).log_likelihood]
    for _ in range(max_iter):
        result = model.posterior(X)
        start = result.responsibilities[0]
        model.states.start = start / start.sum()
        counts = result.transition_counts
        model.states.transitions = counts / counts.sum(axis=1, keepdims=True)
        emission.m_step(X, result.responsibilities)
        history.append(model.posterior(X).log_likelihood)
        if abs(history[-1] - history[-2]) <= len(X) * tol_per_observation:
            break
    model.history = history
    model.trainer.n_iter = len(history) - 1
    model.trainer.converged = abs(history[-1] - history[-2]) <= len(X) * tol_per_observation
    return model, history


def fit_candidates(X, candidates, n_states=3, n_starts=8, max_iter=500):
    """Fit each candidate emission on the same observations.

    Student-t candidates also try a start copied from the best Gaussian fit,
    because random starts can miss the Gaussian region the family contains.

    Args:
        X: Training observations shaped ``(T, 1)``.
        candidates: Map from display name to mixture components per state
            (ignored except for names starting with ``GMM``).
        n_states: Number of hidden states.
        n_starts: Random restarts per candidate.
        max_iter: Maximum EM updates per restart.

    Returns:
        A map from name to the best model and its fit summary, plus a table of
        every attempt. The best attempt maximizes training log likelihood.
    """
    fits, rows, gaussian = {}, [], None
    for name, n_mix in candidates.items():
        attempts, best = fit_restarts(
            name, X, n_states, n_mix, n_starts=n_starts, max_iter=max_iter)
        for attempt in attempts:
            attempt["start"] = f"seed {attempt['seed']}"
        if name == "Gaussian":
            gaussian = attempts[best]["model"]
        if name == "Student-t" and gaussian is not None:
            began = time.perf_counter()
            model, history = student_t_from_gaussian(gaussian, X)
            attempts.append({
                "start": "Gaussian fit", "seed": -1, "model": model,
                "log_likelihood": float(history[-1]),
                "converged": bool(model.trainer.converged),
                "iterations": model.trainer.n_iter,
                "seconds": time.perf_counter() - began,
                "likelihood_decreases": int(np.sum(np.diff(history) < -1e-6)),
                "error": None,
            })
            if attempts[-1]["log_likelihood"] > attempts[best]["log_likelihood"]:
                best = len(attempts) - 1
        chosen = attempts[best]
        n_parameters = parameter_count(name, n_states, n_mix)
        fits[name] = {
            "model": chosen["model"], "start": chosen["start"],
            "log_likelihood": chosen["log_likelihood"],
            "n_parameters": n_parameters,
            "aic": -2 * chosen["log_likelihood"] + 2 * n_parameters,
            "bic": -2 * chosen["log_likelihood"] + n_parameters * np.log(len(X)),
        }
        for attempt in attempts:
            rows.append({
                "model": name, "start": attempt["start"],
                "log_likelihood": attempt["log_likelihood"],
                "converged": attempt["converged"],
                "iterations": attempt["iterations"],
                "likelihood_decreases": attempt["likelihood_decreases"],
                "seconds": attempt["seconds"],
                "selected": attempt["start"] == chosen["start"],
            })
    return fits, pd.DataFrame(rows)


## Fitted-state summaries


def state_summary(model):
    """Return per-state mean, standard deviation, and weight/df details.

    Mixtures are collapsed to their overall state mean and variance; Student-t
    states report the finite-variance standard deviation. Columns hold plain
    daily log-return units.
    """
    emission = model.emission
    if isinstance(emission, GaussianEmission):
        mean = emission.means[:, 0]
        variance = emission.covariances[:, 0, 0]
        extra = {}
    elif isinstance(emission, GaussianMixtureEmission):
        weights = emission.mixture_weights
        means = emission.means[..., 0]
        variances = emission.covariances[..., 0, 0]
        mean = (weights * means).sum(axis=1)
        variance = (weights * (variances + means**2)).sum(axis=1) - mean**2
        extra = {"largest_component_weight": weights.max(axis=1)}
    elif isinstance(emission, StudentTEmission):
        mean = emission.means
        variance = emission.variances()
        extra = {"dof": emission.dof}
    else:
        raise TypeError(f"Unsupported emission: {type(emission).__name__}")
    return {"mean": np.asarray(mean), "sd": np.sqrt(variance), **extra}


def volatility_order(model):
    """Return state indices from lowest to highest fitted standard deviation."""
    return np.argsort(state_summary(model)["sd"])


def stationary_distribution(transitions):
    """Return the stationary distribution of a transition matrix."""
    values, vectors = np.linalg.eig(transitions.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return vector / vector.sum()


def marginal_density(model, grid):
    """Return the stationary-weighted density of the fitted emission on ``grid``."""
    log_p = model.emission.log_prob(np.asarray(grid)[:, None])
    return np.exp(log_p) @ stationary_distribution(model.states.transitions)
