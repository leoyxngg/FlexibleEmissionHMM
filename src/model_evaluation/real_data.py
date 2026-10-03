"""Causal, annual walk-forward evaluation for scalar daily index returns."""

import warnings
from math import erf, sqrt

import numpy as np
import pandas as pd

from flexible_emission_hmm import BaumWelchTrainer, GaussianEmission, HMM


def annual_splits(index, first_year=1990, last_year=2015, training_years=10):
    """Yield ten-calendar-year training windows and following evaluation years."""
    for year in range(first_year, last_year + 1):
        train = (index.year >= year - training_years) & (index.year < year)
        evaluate = index.year == year
        if not train.any() or not evaluate.any():
            raise ValueError(f"Missing training or evaluation observations for {year}")
        yield year, train, evaluate


def fit_gaussian_starts(values, seeds, n_states=3, max_iter=300, tol_per_observation=1e-6):
    """Fit independent starts; select the highest finite training likelihood."""
    observations = np.asarray(values, dtype=float).reshape(-1, 1)
    center = float(observations.mean())
    scale = float(observations.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training returns must have nonzero finite variance")
    standardized = (observations - center) / scale
    if not seeds:
        raise ValueError("At least one initialization seed is required")
    attempts = []
    best = None
    for seed in seeds:
        trainer = BaumWelchTrainer(max_iter=max_iter, tol=tol_per_observation * len(observations))
        model = HMM(GaussianEmission(n_states, min_variance=1e-4, random_state=int(seed)), trainer=trainer)
        record = {"seed": int(seed), "converged": False, "iterations": 0, "likelihood": np.nan,
                  "history": [], "warnings": [], "error": None, "parameters": None}
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(standardized)
            record.update(converged=trainer.converged, iterations=trainer.n_iter,
                          likelihood=float(model.history[-1] - len(observations) * np.log(scale)),
                          history=[float(item) for item in model.history],
                          warnings=[str(item.message) for item in caught],
                          parameters={"start": model.states.start.tolist(),
                                      "transitions": model.states.transitions.tolist(),
                                      "means": (model.emission.means[:, 0] * scale + center).tolist(),
                                      "variances": (model.emission.covariances[:, 0, 0] * scale**2).tolist()})
            if np.isfinite(record["likelihood"]) and (best is None or record["likelihood"] > best[0]):
                best = (record["likelihood"], model, int(seed))
        except Exception as error:
            record["error"] = str(error)
            record["warnings"] = [str(item.message) for item in caught]
        attempts.append(record)
    if best is None:
        raise ValueError("Every Gaussian initialization failed")
    return best[1], center, scale, best[2], attempts


def mixture_moments(probabilities, means, variances):
    mean = probabilities @ means
    variance = probabilities @ (variances + means**2) - mean**2
    return mean, np.maximum(variance, 0)


def lagged_state_forecasts(train_filtered, evaluation_filtered, transitions):
    """Predict each evaluated return from information two closes earlier."""
    if len(train_filtered) < 2:
        raise ValueError("At least two training observations are needed for execution lag")
    historical = np.vstack([train_filtered[-2:], evaluation_filtered])
    return historical[:len(evaluation_filtered)] @ np.linalg.matrix_power(transitions, 2)


def mixture_cdf(value, probabilities, means, standard_deviations):
    standardized = (value - means) / standard_deviations
    return float(np.dot(probabilities, [0.5 * (1 + erf(item / sqrt(2))) for item in standardized]))


def mixture_quantile(level, probabilities, means, standard_deviations):
    lower = float(np.min(means - 12 * standard_deviations))
    upper = float(np.max(means + 12 * standard_deviations))
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if mixture_cdf(midpoint, probabilities, means, standard_deviations) < level:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def ewma_variance(train_returns, evaluation_returns, decay=0.94):
    """Forecast variance before each evaluation return using past returns only."""
    variance = float(np.var(train_returns, ddof=0))
    for value in train_returns:
        variance = decay * variance + (1 - decay) * float(value) ** 2
    forecast = []
    for value in evaluation_returns:
        forecast.append(variance)
        variance = decay * variance + (1 - decay) * float(value) ** 2
    return np.asarray(forecast)


def ewma_lagged_signal_variance(train_returns, evaluation_returns, decay=0.94):
    """Variance known two closes before each evaluated return."""
    history = np.r_[train_returns, evaluation_returns]
    variance = float(np.var(train_returns, ddof=0))
    states = []
    for value in history:
        variance = decay * variance + (1 - decay) * float(value) ** 2
        states.append(variance)
    return np.asarray(states)[len(train_returns) - 2:-2]


def walk_forward_gaussian(frame, first_year=1990, last_year=2015, training_years=10,
                          n_starts=10, master_seed=42, max_iter=300):
    """Return causal forecasts, fit diagnostics, and selected parameters.

    `frame` must be date indexed and contain r_t, simple_return, sigma_t, m_t.
    Returns are decimal units. Forecasts refer to the return on their row date.
    """
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("frame needs unique, increasing dates")
    if frame[["r_t", "simple_return", "sigma_t", "m_t"]].isna().any().any():
        raise ValueError("frame contains missing inputs")
    rows, diagnostics, parameters = [], [], []
    for year, train_mask, eval_mask in annual_splits(frame.index, first_year, last_year, training_years):
        train = frame.loc[train_mask]
        evaluate = frame.loc[eval_mask]
        seed_sequence = np.random.SeedSequence([master_seed, year, training_years])
        seeds = [int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(n_starts)]
        model, center, scale, selected_seed, attempts = fit_gaussian_starts(
            train.r_t.to_numpy(), seeds, max_iter=max_iter)
        means = model.emission.means[:, 0] * scale + center
        variances = model.emission.covariances[:, 0, 0] * scale**2
        deviations = np.sqrt(variances)
        train_x = ((train.r_t.to_numpy() - center) / scale).reshape(-1, 1)
        eval_x = ((evaluate.r_t.to_numpy() - center) / scale).reshape(-1, 1)
        _, train_filtered, _ = model.filter(train_x)
        prior = train_filtered[-1] @ model.states.transitions
        predicted, filtered, log_scores = model.filter(eval_x, prior)
        signals = lagged_state_forecasts(train_filtered, filtered, model.states.transitions)
        _, signal_variances = mixture_moments(signals, means, variances)
        target_weights = np.zeros_like(signal_variances)
        valid_signal = np.isfinite(signal_variances) & (signal_variances > 0)
        target_weights[valid_signal] = np.minimum(1, 0.10 / np.sqrt(252 * signal_variances[valid_signal]))
        mean_benchmark = float(train.r_t.mean())
        std_benchmark = float(train.r_t.std(ddof=0))
        ewma = ewma_variance(train.r_t.to_numpy(), evaluate.r_t.to_numpy())
        ewma_signal = ewma_lagged_signal_variance(train.r_t.to_numpy(), evaluate.r_t.to_numpy())
        ewma_weights = np.zeros_like(ewma_signal)
        valid_ewma = np.isfinite(ewma_signal) & (ewma_signal > 0)
        ewma_weights[valid_ewma] = np.minimum(1, 0.10 / np.sqrt(252 * ewma_signal[valid_ewma]))
        threshold = float(train.sigma_t.median())
        reference = frame[["sigma_t", "m_t"]].shift(1).loc[evaluate.index]
        for index, (date, actual) in enumerate(zip(evaluate.index, evaluate.r_t.to_numpy())):
            probabilities = predicted[index]
            quantiles = {str(level): mixture_quantile(level, probabilities, means, deviations)
                         for level in (0.01, 0.025, 0.05, 0.25, 0.75, 0.95, 0.975)}
            rows.append({
                "date": date, "year": year, "training_cutoff": train.index[-1],
                "r_t": float(actual), "simple_return": float(evaluate.simple_return.iloc[index]),
                "hmm_log_density": float(log_scores[index] - np.log(scale)),
                "gaussian_log_density": float(-0.5 * np.log(2 * np.pi * std_benchmark**2)
                                              - 0.5 * ((actual - mean_benchmark) / std_benchmark)**2),
                "ewma_log_density": float(-0.5 * np.log(2 * np.pi * ewma[index])
                                          - 0.5 * ((actual - mean_benchmark)**2 / ewma[index])),
                "pit": mixture_cdf(actual, probabilities, means, deviations),
                "predicted_state": probabilities.tolist(), "filtered_state": filtered[index].tolist(),
                "signal_state": signals[index].tolist(), "forecast_annual_volatility": float(np.sqrt(252 * signal_variances[index])),
                "target_weight": float(target_weights[index]), "quantiles": quantiles,
                "emission_means": means.tolist(), "emission_variances": variances.tolist(),
                "ewma_target_weight": float(ewma_weights[index]),
                "reference_group": ("high" if reference.sigma_t.iloc[index] > threshold else "low")
                                   + " volatility / " + ("positive" if reference.m_t.iloc[index] > 0 else "negative")
                                   + " momentum",
            })
        diagnostics.extend({"year": year, **attempt} for attempt in attempts)
        parameters.append({"year": year, "selected_seed": selected_seed, "center": center, "scale": scale,
                           "start": model.states.start.tolist(), "transitions": model.states.transitions.tolist(),
                           "means": means.tolist(), "variances": variances.tolist(), "training_rows": len(train),
                           "evaluation_rows": len(evaluate)})
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(diagnostics), pd.DataFrame(parameters)


def predictive_summary(forecasts):
    """Summarize common-date density scores and Gaussian-mixture calibration."""
    records = []
    for period, group in [("combined", forecasts), *[(str(year), group) for year, group in forecasts.groupby("year")]]:
        record = {"period": period, "observations": len(group)}
        for model in ("hmm", "gaussian", "ewma"):
            record[f"{model}_mean_nll"] = -group[f"{model}_log_density"].mean()
        for level, lower, upper in ((0.50, "0.25", "0.75"), (0.90, "0.05", "0.95"),
                                    (0.95, "0.025", "0.975")):
            bounds = np.array([list(item.values()) for item in group.quantiles])
            levels = list(group.quantiles.iloc[0])
            left, right = bounds[:, levels.index(lower)], bounds[:, levels.index(upper)]
            record[f"coverage_{int(level * 100)}"] = np.mean((group.r_t >= left) & (group.r_t <= right))
            record[f"width_{int(level * 100)}"] = np.mean(right - left)
        for level in ("0.01", "0.05"):
            quantile = np.array([item[level] for item in group.quantiles])
            record[f"downside_exceedance_{level}"] = np.mean(group.r_t.to_numpy() < quantile)
        records.append(record)
    return pd.DataFrame(records).set_index("period")


def backtest(forecasts, cash_return=0.0, cost_bps=5, weight_column="target_weight"):
    """Apply lagged target weights with drift-aware one-way turnover."""
    result = forecasts.copy()
    cash = np.broadcast_to(np.asarray(cash_return, dtype=float), len(result))
    if not np.isfinite(cash).all() or not np.isfinite(result[weight_column]).all():
        raise ValueError("Cash returns and target weights must be finite")
    prior_weight = 0.0
    cost_rate = cost_bps / 10000
    if not np.isfinite(cost_rate) or cost_rate < 0 or cost_rate >= 1:
        raise ValueError("cost_bps must be finite and between 0 and 10000")
    turnovers, gross, costs, net = [], [], [], []
    for target, equity_return, cash_day in zip(result[weight_column], result.simple_return, cash):
        if target < 0 or target > 1:
            raise ValueError("target weights must be between zero and one")
        if target >= prior_weight:
            cost = cost_rate * (target - prior_weight) / (1 + cost_rate * target)
        else:
            cost = cost_rate * (prior_weight - target) / (1 - cost_rate * target)
        turnover = abs(target * (1 - cost) - prior_weight)
        gross_return = target * equity_return + (1 - target) * cash_day
        net_return = (1 - cost) * (1 + gross_return) - 1
        if 1 + gross_return <= 0:
            raise ValueError("Portfolio wealth became nonpositive")
        prior_weight = target * (1 + equity_return) / (1 + gross_return)
        turnovers.append(turnover)
        gross.append(gross_return)
        costs.append(cost)
        net.append(net_return)
    result["cash_return"] = cash
    result["turnover"] = turnovers
    result["cost"] = costs
    result["gross_return"] = gross
    result["net_return"] = net
    return result


def portfolio_summary(returns, cash_returns, turnover=None, exposure=None):
    """Compute stitched-series trading statistics in decimal return units."""
    returns = np.asarray(returns, dtype=float)
    excess = returns - np.asarray(cash_returns, dtype=float)
    deviation = excess.std(ddof=1) if len(excess) > 1 else np.nan
    wealth = np.cumprod(1 + returns)
    return {
        "observations": len(returns),
        "sharpe": np.sqrt(252) * excess.mean() / deviation if deviation > 0 else np.nan,
        "cagr": wealth[-1] ** (252 / len(returns)) - 1,
        "annual_volatility": returns.std(ddof=1) * np.sqrt(252),
        "max_drawdown": np.min(wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1),
        "mean_turnover": np.mean(turnover) if turnover is not None else np.nan,
        "mean_exposure": np.mean(exposure) if exposure is not None else np.nan,
    }
