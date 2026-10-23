import numpy as np
from .validation import as_sequence
from .emissions import EmissionModel
from .inference import ExactInference, InferenceEngine
from .parameters import StateParameters
from .training import BaumWelchTrainer, Trainer


class HMM:
    def __init__(
        self,
        emission: EmissionModel,
        inference: InferenceEngine | None = None,
        trainer: Trainer | None = None,
    ):
        self.emission = emission
        self.inference = inference if inference is not None else ExactInference()
        self.trainer = trainer if trainer is not None else BaumWelchTrainer()
        self.n_states = emission.n_states
        K = self.n_states
        self.states = StateParameters(np.full(K, 1 / K), np.full((K, K), 1 / K))
        self.history = []
        self.fitted = False
        self.fit_version = 0

    def fit(self, X):
        """Fit one NumPy array (T, D)
        - T: number of observations
        - D: number of features per observation
        """
        sequences = [X] if isinstance(X, np.ndarray) else list(X)
        if not sequences:
            raise ValueError("Provide at least one sequence")
        sequences = [as_sequence(sequence) for sequence in sequences]
        D = sequences[0].shape[1]
        if any(sequence.shape[1] != D for sequence in sequences):
            raise ValueError("All sequences must have the same number of features")
        self.fitted = False
        self.fit_version += 1
        self.history = []
        self.history = self.trainer.fit(self.states, self.emission, self.inference, sequences)
        self.fitted = True
        return self

    def continue_fit(self, X, max_iter=3, tol=None):
        """Continue a fitted Baum-Welch model without reinitializing parameters.

        Every update processes the entire supplied sequence; ``X`` must
        contain the original observations plus any appended observations.
        """
        if not self.fitted:
            raise ValueError("Call fit before continue_fit")
        if not isinstance(self.trainer, BaumWelchTrainer):
            raise TypeError("Continuation requires BaumWelchTrainer")
        sequences = [X] if isinstance(X, np.ndarray) else list(X)
        if not sequences:
            raise ValueError("Provide at least one sequence")
        sequences = [as_sequence(seq, self.emission.n_features) for seq in sequences]
        previous_max, previous_tol = self.trainer.max_iter, self.trainer.tol
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if tol is not None and (not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be finite and nonnegative")
        try:
            self.trainer.max_iter = max_iter
            if tol is not None:
                self.trainer.tol = tol
            self.history = self.trainer.continue_fit(
                self.states, self.emission, self.inference, sequences
            )
            self.fit_version += 1
        finally:
            self.trainer.max_iter, self.trainer.tol = previous_max, previous_tol
        return self

    def log_emissions(self, X):
        if not self.fitted:
            raise ValueError("Call fit before inference")
        return self.emission.log_prob(as_sequence(X, self.emission.n_features))

    def posterior(self, X):
        """Return likelihood, posterior state probabilities, and transition counts."""
        return self.inference.infer(self.states, self.log_emissions(X))

    def filter(self, X, initial_prior=None):
        """Return causal state priors, filtered probabilities, and log scores."""
        return self.inference.filter(self.states, self.log_emissions(X), initial_prior)

    def predict_proba(self, X):
        """Return posterior state probabilities shaped (T, K)."""
        return self.posterior(X).responsibilities

    def predict(self, X):
        """Return the Viterbi state sequence, shaped (T,)."""
        return self.inference.decode(self.states, self.log_emissions(X))

    def score(self, X):
        """Return the total log likelihood of one sequence."""
        return self.posterior(X).log_likelihood
