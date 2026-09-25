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

### Small Tutorial: train and inspect inferred regimes
#### 1. Train the model

```python
model = HMM(
    emission=GaussianEmission(n_states=2, random_state=42),
    trainer=BaumWelchTrainer(max_iter=200, tol=1e-3),
)
model.fit(X_train)

print("Converged:", model.trainer.converged)
print("Training updates:", model.trainer.n_iter)
```

`converged` means the absolute change in total log likelihood reached
`tol`; fitting can also stop at `max_iter`. Every `fit()` starts fresh.

#### 2. Check the inferred sequence and probabilities

```python
inferred_states = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Keep the same rows as X_test so predictions and true labels stay aligned.
results = df.iloc[split:][["time", "state", "data"]].copy()
results["inferred_state"] = inferred_states
for k in range(model.n_states):
    results[f"prob_state_{k}"] = probabilities[:, k]

print("First 50 inferred regimes:", inferred_states[:50])
print(results.head(20).to_string(index=False))
```
Note that because state numbers are arbitrary (i.e. HMM learns distributions, it does not know the names or numbers we assigned to the true regimes): learned state 0 may correspond to generating regime 2. Use the emission parameters and the cross-tabulation to understand the correspondence before comparing labels or calculating accuracy.