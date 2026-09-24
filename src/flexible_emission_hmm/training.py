from abc import ABC, abstractmethod
import numpy as np
from .emissions import EmissionModel
from .inference import InferenceEngine
from .parameters import StateParameters


class Trainer(ABC):
    @abstractmethod
    def fit(
        self,
        states: StateParameters,
        emission: EmissionModel,
        inference: InferenceEngine,
        sequences: list[np.ndarray],
    ) -> list[float]:
        """Update parameters in place and return total log-likelihood history."""


class BaumWelchTrainer(Trainer):
    """
    Each fit starts from uniform state probabilities and fresh emissions
    History includes the initial likelihood and one entry after each M-step
    """

    def __init__(self, max_iter=100, tol=1e-4):
        """
        - max_iter: maximum number of parameter updates
        - tol: tolerance, if total log likelihood change by at most this amount we stop
        """
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        self.max_iter = max_iter
        self.tol = tol
        self.converged = False
        self.n_iter = 0

    def fit(self, states, emission, inference, sequences):
        X = np.concatenate(sequences, axis=0)
        K = emission.n_states
        states.start = np.full(K, 1 / K)
        states.transitions = np.full((K, K), 1 / K)
        emission.initialize(X)
        self.converged = False
        self.n_iter = 0

        # E-step 
        def expectation():
            return [inference.infer(states, emission.log_prob(seq)) for seq in sequences]
        results = expectation()


        history = [sum(result.log_likelihood for result in results)]
        for iteration in range(self.max_iter):
            starts = np.sum([result.responsibilities[0] for result in results], axis=0)
            states.start = starts / starts.sum()

            counts = np.sum([result.transition_counts for result in results], axis=0)
            totals = counts.sum(axis=1)
            supported = totals > 0
            # Preserve rows with no observed outgoing transitions (e.g. T=1).
            states.transitions[supported] = counts[supported] / totals[supported, None]

            weights = np.concatenate([result.responsibilities for result in results])
            emission.m_step(X, weights)
            results = expectation()
            history.append(sum(result.log_likelihood for result in results))
            self.n_iter = iteration + 1
            if abs(history[-1] - history[-2]) <= self.tol:
                self.converged = True
                break
        return history
