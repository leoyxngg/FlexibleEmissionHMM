import itertools

import numpy as np
import torch

from flexible_emission_hmm.hmmflow import HMMFlow


def make_model(seed=0):
    torch.manual_seed(seed)
    model = HMMFlow(2, seed=seed)
    with torch.no_grad():
        model.emissions[0].loc.fill_(-1.0)
        model.emissions[1].loc.fill_(1.0)
    return model


def test_forward_matches_brute_force():
    model = make_model()
    x = torch.tensor([[-1.2], [0.3], [1.1], [0.9]])
    logb = model.log_emissions(x).detach()
    a = model.transition_matrix.detach()
    pi = model.initial_probs.detach()
    total = 0.0
    for path in itertools.product(range(2), repeat=len(x)):
        p = pi[path[0]] * logb[0, path[0]].exp()
        for t in range(1, len(x)):
            p = p * a[path[t - 1], path[t]] * logb[t, path[t]].exp()
        total += p.item()
    assert np.isclose(model.log_likelihood(x).item(), np.log(total), atol=1e-4)


def test_flow_density_integrates_to_one():
    model = make_model()
    with torch.no_grad():
        grid = torch.linspace(-30, 30, 60001)[:, None]
        for em in model.emissions:
            area = em.log_prob(grid).exp().sum() * (grid[1] - grid[0])
            assert torch.isfinite(em.log_prob(grid)).all()
            assert abs(area.item() - 1) < 1e-2


def test_training_increases_likelihood_and_decodes():
    rng = np.random.default_rng(0)
    states = np.repeat([0, 1, 0, 1], 40)
    x = np.where(states == 0, rng.normal(-2, .5, 160), rng.normal(2, .5, 160))
    model = HMMFlow(2)
    model.init_from_data(x)
    history = model.fit(x, n_steps=60)
    assert history[-1] > history[0]
    assert model.posterior(x).shape == (160, 2)
    path = model.viterbi(x).numpy()
    acc = max((path == states).mean(), (path != states).mean())
    assert acc > 0.95
