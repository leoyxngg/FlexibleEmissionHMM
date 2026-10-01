from numbers import Integral

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from flexible_emission_hmm.parameters import EvaluationResult
from flexible_emission_hmm.validation import as_sequence

STATE_LOG_LOSS_FLOOR = np.finfo(float).eps


def _labels(values, n_states):
    labels = np.asarray(values)
    if (
        labels.ndim != 1
        or labels.size == 0
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= n_states)
    ):
        raise ValueError(f"State labels must be a nonempty integer vector in [0, {n_states})")
    return labels


def _label_sequences(y, sequences, n_states):
    if y is None:
        raise ValueError("Synthetic evaluation requires true state labels")
    labels = [y] if isinstance(y, np.ndarray) and y.ndim == 1 else list(y)
    # Also accept a flat Python label list for a single sequence.
    if labels and np.isscalar(labels[0]):
        labels = [labels]
    if len(labels) != len(sequences):
        raise ValueError("Provide one label vector per observation sequence")
    labels = [_labels(values, n_states) for values in labels]
    if any(len(values) != len(X) for values, X in zip(labels, sequences)):
        raise ValueError("Each label vector must match its observation sequence length")
    return labels


def _probabilities(values, n_states, *, square=False):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != n_states
        or (square and values.shape[0] != n_states)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values.sum(axis=1), 1)
    ):
        shape = "(K, K)" if square else "(T, K)"
        raise ValueError(f"Probabilities must have shape {shape}, be finite, nonnegative, and row-normalized")
    return values


def state_log_loss_metrics(probabilities, true, *, floor=STATE_LOG_LOSS_FLOOR):
    """Capped log loss and tail diagnostics for already aligned posteriors.

    The default restores the original float64 epsilon floor (about 36.04
    nats per observation). Zero probabilities include numerical underflow.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Probabilities must have shape (T, K) with K > 0")
    probabilities = _probabilities(probabilities, probabilities.shape[1])
    true = _labels(true, probabilities.shape[1])
    if len(true) != len(probabilities):
        raise ValueError("True labels must match the number of posterior rows")
    if not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("The probability floor must be finite and between 0 and 1")
    true_probabilities = probabilities[np.arange(len(true)), true]
    return {
        "state_log_loss": float(-np.log(np.clip(true_probabilities, floor, 1)).mean()),
        "state_log_loss_capped_fraction": float(np.mean(true_probabilities < floor)),
        "state_zero_probability_fraction": float(np.mean(true_probabilities == 0)),
    }


class StateAlignment:
    """One-to-one assignment between K learned states and K true states.

    True training labels must cover 0, ..., K-1. Unvisited learned states
    and tied assignment scores can leave the mapping ambiguous; inspect
    ``training_counts`` when interpreting results.
    """

    def __init__(self, n_states):
        if isinstance(n_states, bool) or not isinstance(n_states, Integral) or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        self.n_states = n_states
        self._mapping = None
        self.training_counts = None

    def fit(self, predicted, true):
        """Fit from paired hard training assignments; never from test labels."""
        self._mapping = None
        self.training_counts = None
        predicted = _labels(predicted, self.n_states)
        true = _labels(true, self.n_states)
        if predicted.shape != true.shape:
            raise ValueError("Predicted and true labels must have the same length")
        if np.unique(true).size != self.n_states:
            raise ValueError("Training labels must include all K true states for one-to-one alignment")
        counts = np.zeros((self.n_states, self.n_states), dtype=int)
        np.add.at(counts, (predicted, true), 1)
        learned, truth = linear_sum_assignment(counts, maximize=True)
        mapping = np.empty(self.n_states, dtype=int)
        mapping[learned] = truth
        self._mapping = mapping
        self.training_counts = counts
        return self

    @property
    def mapping(self):
        """Copy of the learned-label -> true-label mapping."""
        if self._mapping is None:
            raise ValueError("Call fit on training assignments before applying alignment")
        return self._mapping.copy()

    @property
    def matrix(self):
        """M[learned, true] = 1, so aligned transitions are M.T @ A @ M."""
        return np.eye(self.n_states)[self.mapping]

    def align_states(self, states):
        return self.mapping[_labels(states, self.n_states)]

    def align_probabilities(self, probabilities):
        order = np.argsort(self.mapping)
        return _probabilities(probabilities, self.n_states)[:, order]

    def align_transitions(self, transitions):
        order = np.argsort(self.mapping)
        return _probabilities(transitions, self.n_states, square=True)[np.ix_(order, order)]


class SyntheticEvaluator:
    """Evaluate a fitted HMM using a fixed training-set state permutation.

    Like HMM.fit, accepts one NumPy sequence or an iterable of sequences.
    For multiple sequences, supply one label vector per sequence. Results
    always contain lists, including when evaluating one sequence.
    """

    def __init__(self, model):
        self.model = model
        self.alignment = None
        self._aligned_model = None
        self._fit_version = None

    def _sequences(self, X):
        if not self.model.fitted:
            raise ValueError("Call model.fit before evaluation")
        sequences = [X] if isinstance(X, np.ndarray) else list(X)
        if not sequences:
            raise ValueError("Provide at least one sequence")
        return [as_sequence(sequence, self.model.emission.n_features) for sequence in sequences]

    def fit_alignment(self, X, y):
        """Learn the state mapping from training data and return this evaluator."""
        self.alignment = None
        model = self.model
        X = self._sequences(X)
        labels = _label_sequences(y, X, model.n_states)
        predicted = [model.predict(sequence) for sequence in X]
        alignment = StateAlignment(model.n_states).fit(
            np.concatenate(predicted), np.concatenate(labels)
        )
        self.alignment = alignment
        self._aligned_model = model
        self._fit_version = model.fit_version
        return self

    def evaluate(self, X, y, *, true_transitions=None) -> EvaluationResult:
        """Score test data with the fixed mapping, without updating the HMM."""
        model = self.model
        if self.alignment is None:
            raise ValueError("Call fit_alignment on training data before evaluation")
        if model is not self._aligned_model or model.fit_version != self._fit_version or not model.fitted:
            raise ValueError("The model changed or was refitted; call fit_alignment again")
        X = self._sequences(X)
        labels = _label_sequences(y, X, model.n_states)
        if true_transitions is not None:
            true_transitions = _probabilities(true_transitions, model.n_states, square=True)

        inferred = [model.posterior(sequence) for sequence in X]
        predicted = [model.predict(sequence) for sequence in X]
        aligned_states = [self.alignment.align_states(states) for states in predicted]
        probabilities = [
            self.alignment.align_probabilities(result.responsibilities)
            for result in inferred
        ]
        transitions = self.alignment.align_transitions(model.states.transitions)
        truth = np.concatenate(labels)
        hard = np.concatenate(predicted)
        soft = np.concatenate(probabilities)
        one_hot = np.eye(model.n_states)[truth]
        metrics = {
            "ari": float(adjusted_rand_score(truth, hard)),
            "nmi": float(normalized_mutual_info_score(truth, hard, average_method="arithmetic")),
            "aligned_accuracy": float(np.mean(np.concatenate(aligned_states) == truth)),
            **state_log_loss_metrics(soft, truth),
            "state_brier_score": float(np.mean(np.sum((soft - one_hot) ** 2, axis=1))),
            "log_likelihood_per_observation": float(
                sum(result.log_likelihood for result in inferred) / len(truth)
            ),
        }
        if true_transitions is not None:
            metrics["transition_rmse"] = float(np.sqrt(np.mean((transitions - true_transitions) ** 2)))
        return EvaluationResult(
            metrics=metrics,
            state_mapping=self.alignment.mapping,
            aligned_states=aligned_states,
            aligned_probabilities=probabilities,
            aligned_transitions=transitions,
        )
