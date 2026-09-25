# CSCD94 Repo

## Topic

This project is aimed to study HMM model with different family of emission distribution other than Gaussian distribution.

### Data Source
The raw data can be downloaded from CRSP https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/index-version-2/daily-index-and-portfolios-on-sp-500/ by selecting every index and the desire range

### Progress

[X] Preprocessing raw data for the model

[X] Simulate labeled GBM paths with scheduled drift and volatility changes [[Example]](https://www.gregorygundersen.com/blog/2024/04/13/simulating-gbm/)

### Dependency Manager

We are using [[uv]](https://docs.astral.sh/uv) for the project. Please follow the installation guide in the doc.

### Generate GBM data with scheduled regimes

Run this from the repository root using the project's Python environment:

```python
from pathlib import Path
import pandas as pd
from synthetic_data_generation import GeometricBrownianMotionGenerator

generator = GeometricBrownianMotionGenerator(
    # Each row is [annual drift, annual volatility]. The row index is its label.
    emission_params=[[0.04, 0.10], [-0.02, 0.40], [0.08, 0.20]],
    # (regime label, number of observations), in chronological order.
    regime_schedule=[(0, 2500), (1, 2500), (2, 2500), (0, 2500)],
    initial_price=100.0,
    dt=1 / 252,  # one trading day, expressed in years
    random_state=42,
)

rows = generator.generate_data()
output = Path("data/synthetic/gbm_scheduled_samples.csv")
output.parent.mkdir(parents=True, exist_ok=True)
generator.write_csv(output, rows)

df = pd.read_csv(output)
X = df[["data"]].to_numpy()  # log returns to feed into HMM.fit()
true_states = df["state"].to_numpy()  # labels for evaluation, not training inputs
```

The CSV columns are `time`, `state`, `data`, `price`, and `simple_return`.
`time` is elapsed years at the end of each interval, starting at `dt`.
`state` identifies the regime that generated that interval's return. The
initial price at time zero is a parameter, not an extra unlabeled CSV row.

Within each interval, the generator uses the exact GBM update:
`log_return = (mu - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z`, with independent
standard normal draws `Z`. Prices compound these log returns across regime
changes; `simple_return = exp(log_return) - 1`.
See the [GBM simulation derivation](https://www.math.uwaterloo.ca/~dlmcleis/s906/chapt1-6.pdf).
The schedule fixes the change points: this example changes regimes before
observations 2501, 5001, and 7501 (counting from 1). The price carries forward
at every change. Regime 0 returns for the last segment with its original
parameters. There is no transition matrix, initial-state probability, or
Markov sampling. Total observation count is the sum of the segment lengths.
Changing `dt` changes their durations in years, not their observation counts.

Change volatility, drift, or both to define different regimes. With one segment,
the generator produces ordinary constant-parameter GBM; with multiple segments,
it produces GBM with piecewise-constant parameters. Only the Gaussian return
noise is random: labels and change points are fixed by the schedule for every
seed. A fresh generator with the same integer seed and settings reproduces the
same path, without changing NumPy's global random state.

Use `data` (log returns) for the Gaussian
HMM; the price level depends on the previous price and is not a Gaussian emission
determined only by the current regime.
