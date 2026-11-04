# Flow emissions on hmmlearn

`FlowHMM` extends `hmmlearn.base.BaseHMM` with the existing
`flexible_emission_hmm.FlowEmission`. It supports scalar and multivariate
observations, normal and Student-t bases, and CPU/CUDA/MPS emission computation.
The existing HMM package and saved runs remain available. The four-model
synthetic comparison notebook uses this backend for its two flow models.

Install the project with the `flows` extra (`uv sync --extra flows` or
`pip install -e '.[flows]'`).

```python
import torch

from hmmlearn_flow_hmm import FlowHMM

# X_train and X_test have shape (T, D), including (T, 1) for scalar data.
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = FlowHMM(
    n_components=3,  # HMM states, not mixture components.
    base_distribution="student_t",  # Or "normal".
    base_df=5.0,
    bound=5.0,
    count_bins=8,
    num_layers=1,
    m_step_steps=5,
    batch_size=2048,  # None uses full-data emission updates.
    learning_rate=1e-3,
    device=device,
    dtype="float32" if device == "mps" else "float64",
    random_state=42,
    n_iter=1000,
    tol=len(X_train) * 1e-6,
    verbose=True,
)
model.fit(X_train)

log_likelihood = model.score(X_test)
probabilities = model.predict_proba(X_test)  # (T, K), smoothed posteriors.
states = model.predict(X_test)              # (T,), Viterbi by default.
priors, filtered, log_predictive = model.filter(X_test)
samples, sampled_states = model.sample(100, random_state=42)
```

`verbose=True` enables hmmlearn's per-iteration reports of training log likelihood
and likelihood improvement to standard error. It does not run multiple restarts:
each `fit()` is one initialization. For ten starts, create and fit ten models with
different seeds and retain the highest `score(X_train)`, without test selection.

`filter()` processes a single sequence with the compiled forward pass. It returns
the state priors before each observation, causal filtered state probabilities,
and predictive log densities. To continue after a training prefix, pass
`initial_prior=filtered_train[-1] @ model.transmat_`. This differs from
`predict_proba()`, which uses future observations when smoothing.

## Multiple sequences

Use hmmlearn's concatenated observations and `lengths` convention:

```python
import numpy as np

lengths = [len(sequence) for sequence in training_sequences]
X = np.concatenate(training_sequences)
model.fit(X, lengths=lengths)
training_log_likelihood = model.score(X, lengths=lengths)
```

All sequences must have the same feature count. State transitions are counted only
within sequences. The emission M-step combines observations with their own fixed
posterior weights, without inventing transitions between sequence boundaries.

## Computation and parameter updates

hmmlearn supplies compiled CPU forward–backward, Viterbi, likelihoods, and standard
initial/transition probability updates. The existing Pyro flows supply density
evaluation, sampling, and responsibility-weighted Adam updates on `device`.
This is generalized EM, not differentiation through the full HMM likelihood.
Scalar data uses direct splines; vector data uses autoregressive splines that
capture dependencies. Each `hidden_dims` width must be at least the feature count.
The Student-t base has independent unit-variance coordinates
and fixed degrees of freedom greater than two. Bounds and base DOF are not trained.

The default `implementation="log"` uses stable log-space inference. hmmlearn's
`implementation="scaling"` is also supported and may be faster, but ordinary
emission densities can underflow for extreme observations or large feature counts;
use log inference in that case. Both variants use CPU float64 HMM calculations,
including when emissions run in float32 on a GPU. GPU scores are transferred to CPU.

The emission M-step supports `batch_size=None` (full data) or a positive integer.
For smaller batches, each Adam step samples observations uniformly with replacement,
keeping the full E-step responsibilities and their normalization fixed. The final
full-data weighted objective must improve on the starting value or the emission
parameters are restored. Full-batch mode checks every iterate instead. A batch
size at least as large as the dataset uses full-batch mode. `m_step_steps` counts
Adam updates per state rather than epochs. Inference still uses complete sequences,
and all observations remain in memory. Each outer iteration starts new Adam
optimizers. Initial and transition
priors retain hmmlearn's semantics; unsupported transition rows are preserved when
an update would otherwise produce a zero row (for example, all singleton sequences).

## Initialization, freezing, and comparison

`init_params="ste"` initializes initial-state probabilities (`s`), transitions
(`t`), and emissions (`e`). The default chain initialization is hmmlearn's random
Dirichlet initialization, which differs from the original HMM's uniform chain.
Flow initialization reuses the existing data-scaled initialization.

`params="ste"` updates all three groups. To preserve a fitted emission and update
only the Markov chain, use `init_params=""` and `params="st"` on the fitted model.
Remove the relevant initialization code as well when supplying custom parameters.

For a controlled comparison with the original HMM's uniform chain:

```python
model = FlowHMM(n_components=3, init_params="e", random_state=42)
model.startprob_ = np.full(3, 1 / 3)
model.transmat_ = np.full((3, 3), 1 / 3)
model.fit(X_train)
```

Match flow settings, seeds, priors, and update counts too. An integer seed produces
repeatable initializations. sklearn-style `RandomState` objects are also supported.

hmmlearn's `monitor_.history` records likelihoods before M-steps; `score()` evaluates
the final model after its last update. `monitor_.iter` is the number of updates.
`monitor_.converged` can mean that the iteration cap was reached, rather than that
the likelihood met `tol`. These conventions differ from the original trainer.

## Attributes and persistence

- `emission_`: the initialized/fitted reusable `FlowEmission` object.
- `startprob_` and `transmat_`: hmmlearn's chain parameters.
- `monitor_`: hmmlearn's convergence history and iteration count.

Standard pickle round-trips retain fitted models; the adapter excludes imported
module handles from the emission's serialized state. Restore on a compatible
PyTorch device/environment. sklearn `clone()` creates an unfitted estimator.
The [synthetic comparison notebook](../../notebooks/nf_spine_synthetic_experiments.ipynb)
uses [nf_hmmlearn_comparison.py](../../notebooks/nf_hmmlearn_comparison.py) for
tuning, fitting, evaluation, and versioned checkpoints. The notebook configures
batch size through `EMISSION_BATCH_SIZE` and the update grid through
`TUNING.m_step_steps`; selected settings carry into final fits. Checkpoints retain
the batch generator's state. Mini-batch experiments use their own run directory
to preserve earlier full-batch results. The original comparison helper remains
available for earlier runs.

The extension follows the custom-emission hooks documented in the
[hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html#implementing-hmms-with-custom-emission-probabilities)
and targets the project's pinned `hmmlearn==0.3.3`.
