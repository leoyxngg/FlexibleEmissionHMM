from abc import ABC, abstractmethod
import numpy as np
from .validation import as_sequence


class EmissionModel(ABC):
    def __init__(self, n_states: int):
        if not isinstance(n_states, (int, np.integer)) or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        self.n_states = int(n_states)
        self.n_features = None

    @abstractmethod
    def initialize(self, X: np.ndarray) -> None:
        """Initialize parameters from X, shaped (T, D)"""
        pass

    @abstractmethod
    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Return shape (T, K)
        e.g. [t, k] is log p(X[t] | state=k).
        """
        pass

    @abstractmethod
    def m_step(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
    ) -> None:
        """
        Update emission parameters
        responsibilities has shape (T, K)
        responsibilities[t, k] = P(state_t=k | all observations)
        """
        pass


class GaussianEmission(EmissionModel):
    def __init__(self, n_states, min_variance=1e-6, random_state=None):
        """
            - n_states: number of states
            - min_variance: bounds covariance eigenvalues, keeping densities defined for
            constant or collinear observations
            """
        super().__init__(n_states)
        if not np.isfinite(min_variance) or min_variance <= 0:
            raise ValueError("min_variance must be finite and positive")
        self.min_variance = min_variance
        self.random_state = random_state
        self.means = None
        self.covariances = None

    def regularize(self, covariance):
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, self.min_variance)
        return (vectors * values) @ vectors.T

    def initialize(self, X):
        X = as_sequence(X)
        self.n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_states, replace=len(X) < self.n_states)
        self.means = X[indices].copy()

        centered = X - X.mean(axis=0)
        covariance = self.regularize(centered.T @ centered / len(X))
        self.covariances = np.repeat(covariance[None], self.n_states, axis=0)

    def check_data(self, X):
        if self.means is None or self.covariances is None:
            raise ValueError("Initialize the emission model before using it")
        return as_sequence(X, self.n_features)

    def log_prob(self, X):
        X = self.check_data(X)
        result = np.empty((len(X), self.n_states))
        constant = self.n_features * np.log(2 * np.pi)
        for k in range(self.n_states):
            # Cholesky avoids explicitly inverting the covariance matrix.
            factor = np.linalg.cholesky(self.covariances[k])
            standardized = np.linalg.solve(factor, (X - self.means[k]).T)
            log_determinant = 2 * np.log(np.diag(factor)).sum()
            result[:, k] = -0.5 * (
                constant + log_determinant + (standardized**2).sum(axis=0)
            )
        return result

    def m_step(self, X, responsibilities):
        X = self.check_data(X)
        weights = np.asarray(responsibilities, dtype=float)
        if (
            weights.shape != (len(X), self.n_states)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights.sum(axis=1), 1)
        ):
            raise ValueError("responsibilities must be normalized weights of shape (T, K)")
        for k in range(self.n_states):
            mass = weights[:, k].sum()
            if mass <= np.finfo(float).eps:
                continue  # Keep parameters for states with no posterior support. (i.e. if E-step give 0 prob)
            self.means[k] = weights[:, k] @ X / mass
            centered = X - self.means[k]
            covariance = (centered.T * weights[:, k]) @ centered / mass
            self.covariances[k] = self.regularize(covariance)
