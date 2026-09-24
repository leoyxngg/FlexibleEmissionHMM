### Overview

The HMM implementation uses NumPy and separates emissions, inference, and
training into replaceable classes.

`ExactInference` and `BaumWelchTrainer` are the defaults, so
`HMM(GaussianEmission(2, random_state=42))` is sufficient for typical use if you want to use Gaussian emissions.

Pass one NumPy array of shape `(T, D)` to `fit`, where T is the number of observations and D is the dimension of each observation, or a list of NumPy arrays for
independent sequences: `model.fit([X1, X2])`. A 1D array is interpreted as `(T, 1).
Sequences may differ in length but must share the same features. A Python list
passed to `fit` always means a collection of sequences; convert a list of
observations to a NumPy array first. Prediction and scoring take one sequence.
Inputs must be nonempty and finite.

We use log-probabilities to prevent underflow when multiplying multiple small probabilities. Covariance eigenvalues are bounded below by `min_variance` (default `1e-6`) to handle constant or collinear features.

### Structure

| Module | Responsibility | Extension point |
| --- | --- | --- |
| `hmm.py` | Public API and component coordination | Usually reuse unchanged |
| `emissions.py` | Observation distributions | Subclass `EmissionModel` |
| `inference.py` | Posterior statistics and decoding | Subclass `InferenceEngine` |
| `training.py` | Parameter updates and convergence | Subclass `Trainer` |
| `parameters.py` | Shared state parameters and inference results | Common data structures |