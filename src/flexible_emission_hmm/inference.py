from abc import ABC, abstractmethod
import numpy as np
from .parameters import InferenceResult, StateParameters


class InferenceEngine(ABC):
    @abstractmethod
    def infer(self, states: StateParameters, log_emissions: np.ndarray) -> InferenceResult:
        """Compute the likelihood and posterior sufficient statistics."""

    @abstractmethod
    def decode(self, states: StateParameters, log_emissions: np.ndarray) -> np.ndarray:
        """Return a most likely hidden-state path."""


class ExactInference(InferenceEngine):
    """Forward/backward alg and Viterbi decoding."""

    def prepare(self, states, log_emissions):
        start = np.asarray(states.start, dtype=float)
        transitions = np.asarray(states.transitions, dtype=float)
        emissions = np.asarray(log_emissions, dtype=float)
        K = start.size
        if start.shape != (K,) or K == 0 or transitions.shape != (K, K):
            raise ValueError("State probabilities must have shapes (K,) and (K, K)")
        for probabilities in (start, transitions):
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=-1), 1)
            ):
                raise ValueError("State probabilities must be finite, nonnegative, and normalized")
        if (
            emissions.ndim != 2
            or emissions.shape[0] == 0
            or emissions.shape[1] != K
            or np.isnan(emissions).any()
            or np.isposinf(emissions).any()
        ):
            raise ValueError("log_emissions must have shape (T, K) with finite values or -inf")
        with np.errstate(divide="ignore"):
            return np.log(start), np.log(transitions), emissions

    def infer(self, states, log_emissions):
        log_start, log_transitions, emissions = self.prepare(states, log_emissions)
        T, K = emissions.shape
        alpha = np.empty((T, K))
        alpha[0] = log_start + emissions[0]
        for t in range(1, T):
            alpha[t] = emissions[t] + np.logaddexp.reduce(
                alpha[t - 1, :, None] + log_transitions, axis=0
            )
        log_likelihood = float(np.logaddexp.reduce(alpha[-1]))
        if not np.isfinite(log_likelihood):
            raise ValueError("The sequence has zero probability or a non-finite likelihood")

        beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = np.logaddexp.reduce(
                log_transitions + emissions[t + 1] + beta[t + 1], axis=1
            )

        log_gamma = alpha + beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        responsibilities = np.exp(log_gamma)

        counts = np.zeros((K, K))
        for t in range(T - 1):
            log_xi = (
                alpha[t, :, None] + log_transitions + emissions[t + 1] + beta[t + 1]
            )
            counts += np.exp(log_xi - np.logaddexp.reduce(log_xi.ravel()))
        return InferenceResult(log_likelihood, responsibilities, counts)

    def decode(self, states, log_emissions):
        log_start, log_transitions, emissions = self.prepare(states, log_emissions)
        T, K = emissions.shape
        backpointers = np.zeros((T, K), dtype=int)
        scores = log_start + emissions[0]
        for t in range(1, T):
            candidates = scores[:, None] + log_transitions
            backpointers[t] = candidates.argmax(axis=0)
            scores = candidates.max(axis=0) + emissions[t]
        if not np.isfinite(scores.max()):
            raise ValueError("No possible hidden-state path for this sequence")

        path = np.empty(T, dtype=int)
        path[-1] = scores.argmax()
        for t in range(T - 2, -1, -1):
            path[t] = backpointers[t + 1, path[t + 1]]
        return path
