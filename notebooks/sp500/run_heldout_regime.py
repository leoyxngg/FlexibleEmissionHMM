"""Resumable expanding-window held-out HMM predictive scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import sys

import numpy as np
import pandas as pd

try:
    from .run_paper_regime import (
        DEFAULT_MODELS, HALF_LIVES, MODEL_KEYS, WINDOWS, _atomic_bytes,
        _prepare, _refresh, build_observations, parse_specs,
    )
except ImportError:
    from run_paper_regime import (
        DEFAULT_MODELS, HALF_LIVES, MODEL_KEYS, WINDOWS, _atomic_bytes,
        _prepare, _refresh, build_observations, parse_specs,
    )


def score_block(model, training, test):
    """Score unseen observations conditional on the filtered training history.

    Args:
        model: Fitted HMM, kept fixed throughout scoring.
        training: Transformed training observations.
        test: Transformed subsequent observations.

    Returns:
        Training and test arrays of sequential predictive log densities.
    """
    _, filtered, train_scores = model.filter(training)
    prior = filtered[-1] @ model.states.transitions
    _, _, test_scores = model.filter(test, initial_prior=prior)
    return train_scores, test_scores


def run(prices, output_dir, start, end, years=10, seed=0, max_iter=100,
        test_days=63, models=DEFAULT_MODELS, command=None):
    """Fit before each chronological block and save fixed-parameter test scores.

    Args:
        prices: Positive price series indexed by trading date.
        output_dir: Dedicated result folder; resume only trusted checkpoints.
        start: First held-out observation date, excluded from initial training.
        end: Last held-out date, included in the experiment fingerprint.
        years: Initial trailing calendar years; history then expands.
        seed: Shared fitting seed.
        max_iter: Maximum EM iterations per fit.
        test_days: Number of observations per nonoverlapping test block.
        models: Branch keys accepted by ``run_paper_regime``.
        command: Optional command-line provenance.

    Returns:
        Daily predictive scores and per-block summary tables.

    Raises:
        ValueError: Data, dates, settings, or resumed checkpoint are incompatible.
    """
    if any(type(v) is not int or v < 1 for v in (years, max_iter, test_days)):
        raise ValueError("years, max_iter, and test_days must be positive integers")
    specs = parse_specs(models)
    prices = prices.astype(float).sort_index()
    if (not isinstance(prices.index, pd.DatetimeIndex)
            or not prices.index.is_unique or prices.index.hasnans
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("Prices must be positive and finite with unique dates")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start or end > prices.index[-1]:
        raise ValueError("Invalid held-out date range")
    features = build_observations(
        prices, windows=WINDOWS, half_lives=HALF_LIVES
    ).dropna()
    initial = features.loc[
        (features.index >= start - pd.DateOffset(years=years))
        & (features.index < start)
    ].index
    dates = features.loc[start:end].index
    if len(initial) < years * 200 or len(dates) == 0 or dates[0] != start:
        raise ValueError("Insufficient training data or invalid test start")
    returns = prices.pct_change(fill_method=None)
    settings = dict(version=1, start=str(start), end=str(end), years=years,
                    seed=seed, max_iter=max_iter, test_days=test_days,
                    models=[s.key for s in specs], windows=WINDOWS,
                    half_lives=HALF_LIVES)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(prices.index.asi8, dtype='<i8').tobytes())
    digest.update(np.asarray(prices, dtype='<f8').tobytes())
    fingerprint = digest.hexdigest()
    output = Path(output_dir)
    checkpoint = output / 'checkpoint.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as handle:
            state = pickle.load(handle)
        if state['fingerprint'] != fingerprint:
            raise ValueError("Checkpoint settings/data differ; use a new folder")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty output folder or a compatible checkpoint")
        state = dict(fingerprint=fingerprint, completed=0, daily=[], folds=[])
    completed = state['completed']
    if (not 0 <= completed <= len(dates)
            or (completed < len(dates) and completed % test_days)
            or len(state['daily']) != completed * len(specs)):
        raise ValueError("Incomplete checkpoint")
    if command is not None:
        _atomic_bytes(output / 'command.txt', command.encode())
    _atomic_bytes(output / 'settings.json', json.dumps(settings, indent=2).encode())

    for offset in range(completed, len(dates), test_days):
        test_dates = dates[offset:offset + test_days]
        train_dates = features.loc[
            (features.index >= initial[0]) & (features.index < test_dates[0])
        ].index
        print(f"Fit before {test_dates[0].date()}: {len(train_dates)} train, "
              f"{len(test_dates)} test observations", flush=True)
        branches = _refresh(features, returns, train_dates, seed, max_iter, specs)
        daily, folds = [], []
        for spec in specs:
            model, scaler, pca = branches[spec.key]
            training = _prepare(features, train_dates, scaler, pca)
            test = _prepare(features, test_dates, scaler, pca)
            train_scores, test_scores = score_block(model, training, test)
            metadata = dict(
                fold=offset // test_days, spec=spec.key, model=spec.emission,
                pca=spec.use_pca, gmm_components=spec.gmm_components,
                train_end=train_dates[-1], training_days=len(training),
                n_features=training.shape[1],
            )
            folds.append(dict(
                **metadata, test_start=test_dates[0], test_end=test_dates[-1],
                test_days=len(test), train_mean_log_score=float(train_scores.mean()),
                test_mean_log_score=float(test_scores.mean()),
                test_total_log_score=float(test_scores.sum()),
                converged=bool(model.trainer.converged),
                iterations=int(model.trainer.n_iter),
            ))
            daily.extend(dict(**metadata, date=date, log_predictive=float(score))
                         for date, score in zip(test_dates, test_scores))
            print(f"  {spec.key}: test mean log score {test_scores.mean():.6f}",
                  flush=True)
        state.update(completed=offset + len(test_dates), branches=branches)
        state['daily'].extend(daily)
        state['folds'].extend(folds)
        _atomic_bytes(checkpoint, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        _save_tables(output, state)
    # Regenerate CSV views even when resuming an already completed experiment.
    return _save_tables(output, state)


def _save_tables(output, state):
    daily, folds = pd.DataFrame(state['daily']), pd.DataFrame(state['folds'])
    summary = daily.groupby(['spec', 'pca'])['log_predictive'].agg(
        observations='size', mean_log_score='mean', total_log_score='sum'
    ).reset_index()
    for name, frame in [('daily_scores', daily), ('fold_scores', folds),
                        ('summary', summary)]:
        _atomic_bytes(output / f'{name}.csv', frame.to_csv(index=False).encode())
    return daily, folds


def main():
    """Run chronological held-out scoring on the existing SPX price data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', default='2004-01-02')
    parser.add_argument('--end', default='2025-12-30')
    parser.add_argument('--years', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--test-days', type=int, default=63)
    parser.add_argument('--models', nargs='+', choices=MODEL_KEYS,
                        default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / 'data/processed/sp500_index_price.csv'
    prices = pd.read_csv(source, parse_dates=['YYYYMMDD']).set_index('YYYYMMDD')
    run(prices['DlyPrcInd'], args.output, args.start, args.end, years=args.years,
        seed=args.seed, max_iter=args.max_iter, test_days=args.test_days,
        models=args.models, command=shlex.join([sys.executable, *sys.argv]) + '\n')


if __name__ == '__main__':
    main()
