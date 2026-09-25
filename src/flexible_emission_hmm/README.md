### Overview

The HMM implementation uses NumPy and separates emissions, inference, and
training into replaceable classes.

We use log-probabilities to prevent underflow when multiplying multiple small probabilities. Covariance eigenvalues are bounded below by `min_variance` (default `1e-6`) to handle constant or collinear features.

### Structure

| Module | Responsibility | Extension point |
| --- | --- | --- |
| `hmm.py` | Public API and component coordination | Usually reuse unchanged |
| `emissions.py` | Observation distributions | Subclass `EmissionModel` |
| `inference.py` | Posterior statistics and decoding | Subclass `InferenceEngine` |
| `training.py` | Parameter updates and convergence | Subclass `Trainer` |
| `parameters.py` | Shared state parameters and inference results | Common data structures |