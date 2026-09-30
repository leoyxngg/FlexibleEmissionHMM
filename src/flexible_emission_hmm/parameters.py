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


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    state_mapping: np.ndarray | None = None  # learned label -> true label
    aligned_states: list[np.ndarray] | None = None  # one (T,) array per sequence
    aligned_probabilities: list[np.ndarray] | None = None  # one (T, K) per sequence
    aligned_transitions: np.ndarray | None = None  # (K, K), in true-state order
