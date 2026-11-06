import numpy as np
import torch

from flexible_emission_hmm.hmmflow import FlowEmission, HMMFlow
from flexible_emission_hmm.torch_hmm import (
    EmissionHMM, GaussianStateEmission, StudentTStateEmission,
)


def test_mixed_families_normalize_and_decode():
    rng = np.random.default_rng(0)
    states = np.repeat([0, 1, 0, 1], 60)
    x = np.where(states == 0, rng.normal(0, 0.5, 240),
                 2 + 0.5 * rng.standard_t(4, 240))
    model = EmissionHMM([GaussianStateEmission(), StudentTStateEmission()])
    model.init_from_data(x)
    history = model.fit(x, n_steps=80)
    assert history[-1] > history[0]
    grid = torch.linspace(-60, 60, 120001)[:, None]
    with torch.no_grad():
        for em in model.emissions:
            area = em.log_prob(grid).exp().sum() * (grid[1] - grid[0])
            assert abs(area.item() - 1) < 1e-2
    assert model.posterior(x).shape == (240, 2)
    path = model.viterbi(x).numpy()
    assert max((path == states).mean(), (path != states).mean()) > 0.9


def test_flow_identity_init_and_penalty():
    em = FlowEmission(count_bins=6, identity_init=True)
    x = torch.linspace(-4, 4, 9)[:, None]
    assert (em.spline(x) - x).abs().max() < 1e-4
    assert em.penalty().item() < 1e-8
    assert FlowEmission(count_bins=6).penalty().item() > 0.1


def test_regularization_shrinks_flow_toward_identity():
    rng = np.random.default_rng(1)
    x = rng.standard_t(3, 200)
    penalties = []
    for reg in (0.0, 50.0):
        torch.manual_seed(0)
        model = HMMFlow(2, identity_init=True)
        model.init_from_data(x)
        model.fit(x, n_steps=60, regularization=reg)
        penalties.append(model.penalty().item())
    assert penalties[1] < penalties[0]


def test_mixed_flow_and_gaussian():
    model = EmissionHMM([FlowEmission(identity_init=True), GaussianStateEmission()])
    x = torch.randn(50, 1)
    assert model.log_emissions(x).shape == (50, 2)
    assert np.isfinite(model.score(x))
