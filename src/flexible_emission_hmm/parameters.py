from dataclasses import dataclass
import numpy as np

@dataclass
class StateParameters:
    start: np.ndarray  # (K,)
    transitions: np.ndarray  # (K, K)

@dataclass
class InferenceResult:
    log_likelihood: float
    responsibilities: np.ndarray  # (T, K): posterior state probabilities
    transition_counts: np.ndarray  # (K, K): summed posterior transition counts
