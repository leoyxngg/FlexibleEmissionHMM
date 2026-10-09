"""Causal regime signals for market-strategy benchmarks.

Every function here answers one question: "given data through yesterday, what
regime probability should today carry?" Probabilities returned are one-step
ahead: row t is P(S_t | observations before t), which is legal to trade on at
the open of day t. Smoothed posteriors (forward-backward, ``predict_proba``)
peek at future data and must never enter a strategy.
"""

import numpy as np


def forward_predictive_probabilities(
    transitions: np.ndarray,
    initial_prior: np.ndarray,
    log_emissions: np.ndarray,
) -> np.ndarray:
    """Return P(S_t | emissions < t) for every t, shaped (T, K).

    A pure-NumPy causal filter: each row is produced by the forward pass only,
    so it is known before the observation at that index. Mirrors
    ``flexible_emission_hmm.inference.ExactInference.filter`` for any model that
    exposes a transition matrix, an initial prior, and per-day log emission
    densities (e.g. hmmlearn fitted with private log-likelihood access).
    """
    transitions = np.asarray(transitions, dtype=float)
    initial_prior = np.asarray(initial_prior, dtype=float)
    log_emissions = np.asarray(log_emissions, dtype=float)
    if transitions.ndim != 2 or transitions.shape[0] != transitions.shape[1]:
        raise ValueError("transitions must be a square (K, K) matrix")
    n_states = transitions.shape[0]
    if initial_prior.shape != (n_states,) or not np.isclose(initial_prior.sum(), 1.0):
        raise ValueError("initial_prior must be a normalized vector of length K")
    if log_emissions.ndim != 2 or log_emissions.shape[1] != n_states:
        raise ValueError("log_emissions must have shape (T, K)")
    if not np.isfinite(log_emissions).all():
        raise ValueError("log_emissions must be finite")

    prior = initial_prior.copy()
    predicted = np.empty_like(log_emissions)
    for t, log_density in enumerate(log_emissions):
        predicted[t] = prior
        log_joint = np.log(prior, where=prior > 0, out=np.full_like(prior, -np.inf)) + log_density
        log_total = np.logaddexp.reduce(log_joint)
        filtered = np.exp(log_joint - log_total)
        prior = filtered @ transitions
    return predicted


def _emission_covariances(model: object) -> np.ndarray:
    """Covariance tensors from a fitted local HMM or an hmmlearn GaussianHMM."""
    emission = getattr(model, "emission", None)
    if emission is not None and getattr(emission, "covariances", None) is not None:
        return np.asarray(emission.covariances, dtype=float)
    covariances = getattr(model, "covars_", None)
    if covariances is None:
        raise ValueError("model exposes neither emission.covariances nor covars_; is it fitted?")
    return np.asarray(covariances, dtype=float)


def high_volatility_state(model: object, return_feature: int = 0) -> int:
    """Index of the fitted state with the largest standard deviation of the return feature.

    Assumes column ``return_feature`` of the feature matrix is the daily return.
    """
    covariances = _emission_covariances(model)
    return int(np.argmax(covariances[:, return_feature, return_feature]))


def hmm_causal_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    """One-step-ahead regime probabilities from a fitted flexible_emission_hmm.HMM.

    Delegates to ``model.filter``, whose first output is by construction
    available before each corresponding observation.
    """
    features = np.asarray(features, dtype=float)
    predicted, _filtered, _scores = model.filter(features)
    return predicted


def hmmlearn_causal_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    """One-step-ahead regime probabilities from a fitted hmmlearn GaussianHMM.

    Uses hmmlearn's internal log-emission lattice (implementation detail) so the
    same causal filter applies to both packages.
    """
    features = np.asarray(features, dtype=float)
    log_emissions = model._compute_log_likelihood(features)
    return forward_predictive_probabilities(model.transmat_, model.startprob_, log_emissions)
