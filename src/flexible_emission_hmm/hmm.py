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
        self.history = []
        self.history = self.trainer.fit(self.states, self.emission, self.inference, sequences)
        self.fitted = True
        return self

    def log_emissions(self, X):
        if not self.fitted:
            raise ValueError("Call fit before inference")
        return self.emission.log_prob(as_sequence(X, self.emission.n_features))

    def posterior(self, X):
        """Return likelihood, posterior state probabilities, and transition counts."""
        return self.inference.infer(self.states, self.log_emissions(X))

    def predict_proba(self, X):
        """Return posterior state probabilities shaped (T, K)."""
        return self.posterior(X).responsibilities

    def predict(self, X):
        """Return the Viterbi state sequence, shaped (T,)."""
        return self.inference.decode(self.states, self.log_emissions(X))

    def score(self, X):
        """Return the total log likelihood of one sequence."""
        return self.posterior(X).log_likelihood
