"""Replay saved spline-flow HMM backtests and compute state diagnostics.

The helpers rebuild flow models at scheduled refit dates, which reproduces
the saved backtest because every refit reseeds and rebuilds the scaler, PCA
and model. They also evaluate exact PC1 marginals and spline maps, and
sampled same-day return densities.
"""

import hashlib
import json
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "notebooks/sp500/results"
for _path in (ROOT / "src", ROOT / "notebooks/sp500"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_backtest import (  # noqa: E402
    FlowEngine, _atomic_bytes, _fit_preprocessing, _schedule, bull_state,
)

CACHE = RESULTS / "flow_replay_cache"


def load_prices():
    """Return the S&P 500 price index used by every backtest."""
    frame = pd.read_csv(ROOT / "data/processed/sp500_index_price.csv",
                        parse_dates=["YYYYMMDD"])
    return frame.set_index("YYYYMMDD")["DlyPrcInd"]


def load_run(folder):
    """Return the settings and saved signals of a flow backtest run."""
    path = RESULTS / folder
    settings = json.loads((path / "settings.json").read_text())
    signals = pd.read_csv(path / "signals.csv", parse_dates=["date"])
    return dict(folder=folder, settings=settings, signals=signals)


def _engine(settings):
    return FlowEngine(settings["seed"], settings["max_iter"],
                      settings["update_iter"], n_states=settings["n_states"],
                      device="cpu", **settings["flow"])


def _history(prices, settings, date):
    _, start, features, returns, initial, _ = _schedule(
        prices, settings["start"], date, settings["years"])
    date = pd.Timestamp(date)
    history = initial if date == start else features.loc[initial[0]:date].index
    return features, returns, history


def snapshot(model, scaler, pca, features, returns, history):
    """Return the observations, posteriors and bull index for one model."""
    X = pca.transform(scaler.transform(features.loc[history]))
    r = returns.loc[history].to_numpy()
    _, filtered, _ = model.filter(X)
    bull = bull_state(filtered, r)
    next_prior = filtered[-1] @ model.transmat_
    return dict(model=model, scaler=scaler, pca=pca, X=X, returns=r,
                dates=history, bull=bull, smoothed=model.predict_proba(X),
                next_bull=float(next_prior[bull]))


def refit_at(settings, date):
    """Refit a flow run at one scheduled refit date and return its snapshot."""
    features, returns, history = _history(load_prices(), settings, date)
    engine = _engine(settings)
    model, scaler, pca = engine.refresh(features, returns, history)[engine.spec]
    return snapshot(model, scaler, pca, features, returns, history)


def em_snapshots(settings, date, total, steps=(1, 3, 10, 30)):
    """Refit at ``date`` step by step; return snapshots and log-likelihoods.

    The model is initialized like ``FlowEngine.refresh`` and advanced with
    warm-start fits. Snapshot 0 is the initialization; the last runs to
    ``total`` iterations, the count of the saved refit.
    """
    from hmmlearn_flow_hmm import FlowHMM
    features, returns, history = _history(load_prices(), settings, date)
    scaler, pca, standardized = _fit_preprocessing(features, history)
    X = pca.transform(standardized)
    engine = _engine(settings)
    model = FlowHMM(n_components=engine.n_states, device="cpu", dtype="float64",
                    random_state=int(engine.seed), n_iter=1, tol=0.0,
                    convergence_patience=5, **engine.options)
    model._init(X, np.asarray([len(X)]))
    model.set_params(init_params="")  # Continue from this initialization.

    def take():
        copy = pickle.loads(pickle.dumps(model))
        return snapshot(copy, scaler, pca, features, returns, history)

    snaps, log_likelihood, done = {0: take()}, [], 0
    for step in [*(s for s in steps if s < total), total]:
        model.set_params(n_iter=step - done)
        model.fit(X)
        log_likelihood.extend(model.history_[1:] if done else model.history_)
        done = step
        snaps[step] = take()
    return snaps, np.asarray(log_likelihood)


def _limited(function, threads, *args):
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(threads)
    with threadpool_limits(threads):
        return function(*args)


def _cache_path(cache, key, job, threads):
    """Name a cache file by its job key and a digest of everything it depends on.

    The digest covers the function, its arguments (run settings, dates), the
    thread count, which changes results in the last digits, and the prices.
    """
    prices = load_prices()
    digest = hashlib.sha256()
    digest.update(np.asarray(prices.index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(prices.to_numpy(), dtype="<f8").tobytes())
    digest.update(json.dumps([job[0].__name__, job[1:], threads],
                             sort_keys=True, default=str).encode())
    parts = key if isinstance(key, tuple) else (key,)
    label = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(map(str, parts))).strip("_")
    return cache / f"{job[0].__name__}_{label}_{digest.hexdigest()[:12]}.pkl"


def run_jobs(jobs, workers=4, threads=4, cache=None):
    """Run ``{key: (function, *args)}`` in spawn processes; return results by key.

    With ``cache``, reuse a saved result whose inputs match and save new ones
    there. Cache files are trusted pickles written by this function.
    """
    paths = ({key: _cache_path(Path(cache), key, job, threads)
              for key, job in jobs.items()} if cache is not None else {})
    results = {}
    for key, path in paths.items():
        if path.exists():
            with path.open("rb") as handle:
                results[key] = pickle.load(handle)
    pending = {key: job for key, job in jobs.items() if key not in results}
    if cache is not None:
        print(f"Cache {cache}: {len(results)} reused, {len(pending)} to compute")
    if pending:
        with ProcessPoolExecutor(min(workers, len(pending)),
                                 mp_context=get_context("spawn")) as pool:
            futures = {key: pool.submit(_limited, job[0], threads, *job[1:])
                       for key, job in pending.items()}
            for key, future in futures.items():
                results[key] = future.result()
                if cache is not None:
                    _atomic_bytes(paths[key], pickle.dumps(
                        results[key], protocol=pickle.HIGHEST_PROTOCOL))
    return {key: results[key] for key in jobs}


def pc1_sign(pca, feature_names):
    """Return +1 or -1 so that oriented PC1 rises as downside volatility falls."""
    downside = [i for i, name in enumerate(feature_names)
                if name.startswith("downside")]
    return 1.0 if pca.components_[0, downside].sum() <= 0 else -1.0


def base_log_prob(emission, z):
    """Return the unit-variance base log density at ``z`` for one coordinate."""
    if emission.base_distribution == "normal":
        return -0.5 * (z**2 + np.log(2 * np.pi))
    nu = emission.base_df
    scale = np.sqrt((nu - 2) / nu)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi) - np.log(scale)
            - (nu + 1) / 2 * np.log1p((z / scale) ** 2 / nu))


def pc1_flow(model, state, grid, sign=1.0):
    """Return the exact PC1 spline map and marginal density on oriented ``grid``.

    PC1 is first in the autoregressive order, so its transform depends on no
    other coordinate. The result holds u, z, dz/du, log density, location,
    scale and spline knots, all on the oriented axis.
    """
    import torch
    from pyro.distributions.transforms.spline import _calculate_knots
    emission = model.emission_
    flow = emission.flows[state]
    location = flow.location.detach()
    log_scale = flow.log_scale.detach().clamp(min=float(np.log(emission.min_scale)))
    scale0 = float(log_scale[0].exp())
    x1 = torch.as_tensor(sign * np.asarray(grid), dtype=location.dtype)
    u = torch.zeros(len(x1), len(location), dtype=location.dtype)
    u[:, 0] = (x1 - location[0]) / scale0
    u.requires_grad_(True)
    z = u
    for transform in flow.transforms:
        transform.clear_cache()
        z = transform(z)
    slope, = torch.autograd.grad(z[:, 0].sum(), u)
    u0 = u[:, 0].detach().numpy()
    z0 = z[:, 0].detach().numpy()
    slope0 = slope[:, 0].numpy()
    log_density = base_log_prob(emission, z0) + np.log(slope0) - np.log(scale0)
    bound = emission.bound
    with torch.no_grad():
        spline = flow.transforms[0].spline.condition(u[:1].detach())
        widths, heights, _, _ = (spline._params() if callable(spline._params)
                                 else spline._params)
        _, knots_u = _calculate_knots(widths[0, 0], -bound, bound)
        _, knots_z = _calculate_knots(heights[0, 0], -bound, bound)
    return dict(u=sign * u0, z=sign * z0, slope=slope0, log_density=log_density,
                location=sign * float(location[0]), scale=scale0,
                knots_u=sign * knots_u.numpy(), knots_z=sign * knots_z.numpy(),
                bound=bound)


def return_samples(snap, state, n=20000, seed=0):
    """Sample same-day returns (%) from one state, seen through the 95% PCA."""
    draws = snap["model"].emission_.sample(state, n_samples=n, random_state=seed)
    features = snap["scaler"].inverse_transform(snap["pca"].inverse_transform(draws))
    return 100 * features[:, 0]


def reconstructed_returns(snap):
    """Return observed same-day returns (%) after the PCA round trip."""
    features = snap["scaler"].inverse_transform(
        snap["pca"].inverse_transform(snap["X"]))
    return 100 * features[:, 0]
