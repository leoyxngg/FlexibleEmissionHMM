"""Benchmark mixed-family HMMs against Gaussian, Student-t and flow HMMs.

Run: uv run python notebooks/emission_mix_benchmark.py [--seeds 5]
"""

import argparse

import numpy as np
import torch

from flexible_emission_hmm.hmmflow import FlowEmission
from flexible_emission_hmm.torch_hmm import (
    EmissionHMM, GaussianStateEmission, StudentTStateEmission,
)
from hmmflow_small_experiment import simulate


def make_models(seed):
    """Return name -> (model factory, regularization weight)."""
    g, t = GaussianStateEmission, StudentTStateEmission
    flow = lambda: FlowEmission(identity_init=True)  # noqa: E731
    return {
        "gaussian x2": ([g, g], 0.0),
        "student_t x2": ([t, t], 0.0),
        "gaussian + student_t": ([g, t], 0.0),
        "flow x2 (no reg)": ([flow, flow], 0.0),
        "flow x2 (reg=1)": ([flow, flow], 1.0),
        "flow x2 (reg=10)": ([flow, flow], 10.0),
    }


def run(seeds, n_train=500, n_test=500, steps=400):
    """Return {name: [held-out log-likelihood per seed]} on standardized data."""
    results = {}
    for seed in seeds:
        x_tr, _ = simulate(n_train, 100 + seed)
        x_te, _ = simulate(n_test, 200 + seed)
        mu, sd = x_tr.mean(), x_tr.std()
        z_tr, z_te = (x_tr - mu) / sd, (x_te - mu) / sd
        for name, (factories, reg) in make_models(seed).items():
            torch.manual_seed(seed)
            model = EmissionHMM([f() for f in factories], seed=seed)
            model.init_from_data(z_tr)
            model.fit(z_tr, steps, regularization=reg)
            results.setdefault(name, []).append(model.score(z_te) / n_test)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    print("Held-out mean log-likelihood per observation (standardized units)")
    for name, vals in run(range(args.seeds)).items():
        print(f"{name:24s} {np.mean(vals):8.4f} +- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
