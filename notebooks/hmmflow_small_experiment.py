"""Small HMMFlow experiment on non-Gaussian synthetic data.

Run: uv run --extra rocm python notebooks/hmmflow_small_experiment.py [--device cpu]
"""

import argparse
import time

import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

from flexible_emission_hmm.hmmflow import HMMFlow
from synthetic_data_generation import StateDependentEmissionGenerator

N_STATES = 3
TRANSITIONS = np.array([[.92, .04, .04], [.04, .92, .04], [.04, .04, .92]])
EMISSIONS = [
    ("multimodal", (-2.0, 1.5, [.5, .5], [-1.5, 1.5], [.4, .4])),
    ("skew_normal", (0.5, 1.0, 8.0)),
    ("student_t", (3.0, 1.0, 3.0)),
]


def simulate(n, seed):
    """Return observations (n, 1) and true states from the repo generator."""
    gen = StateDependentEmissionGenerator(
        EMISSIONS, np.arange(N_STATES), np.full(N_STATES, 1 / N_STATES),
        TRANSITIONS, n, random_state=seed)
    path = gen.sample_state_path()
    return gen.sample_emissions(path), path


def permuted_accuracy(pred, true, k=N_STATES):
    """Return accuracy under the best one-to-one label permutation."""
    counts = np.zeros((k, k), dtype=int)
    for p, t in zip(pred, true):
        counts[p, t] += 1
    rows, cols = linear_sum_assignment(-counts)
    return counts[rows, cols].sum() / len(true)


def run(device, seeds, n_train=300, n_test=300, steps=400):
    rows = []
    for seed in seeds:
        x_tr, y_tr = simulate(n_train, 100 + seed)
        x_te, y_te = simulate(n_test, 200 + seed)
        mu, sd = x_tr.mean(), x_tr.std()
        z_tr, z_te = (x_tr - mu) / sd, (x_te - mu) / sd
        jac = n_test * np.log(sd)  # standardization -> original-units density

        g = GaussianHMM(N_STATES, n_iter=200, random_state=seed).fit(z_tr)
        rows.append(("gaussian", seed, g.score(z_te) - jac,
                     permuted_accuracy(g.predict(z_te), y_te), 0.0))

        t0 = time.time()
        torch.manual_seed(seed)
        m = HMMFlow(N_STATES, seed=seed).to(device)
        m.init_from_data(z_tr)
        m.fit(torch.tensor(z_tr, dtype=torch.float32, device=device), steps)
        elapsed = time.time() - t0
        rows.append(("hmmflow", seed, m.score(z_te) - jac,
                     permuted_accuracy(m.viterbi(z_te).numpy(), y_te), elapsed))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    devices = [args.device] if args.device else ["cpu"] + (
        ["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        rows = run(device, range(args.seeds))
        print(f"\n== device {device} (held-out T=300, loglik in original units) ==")
        for name in ("gaussian", "hmmflow"):
            r = np.array([row[2:] for row in rows if row[0] == name])
            print(f"{name:9s} test_ll {r[:,0].mean():8.1f} +- {r[:,0].std():5.1f}"
                  f" | acc {r[:,1].mean():.3f} +- {r[:,1].std():.3f}"
                  f" | fit {r[:,2].mean():5.1f}s")
        if device != devices[0]:
            continue
        for row in rows:
            print("  ", row[0], row[1], f"ll {row[2]:.1f} acc {row[3]:.3f}")


if __name__ == "__main__":
    main()
