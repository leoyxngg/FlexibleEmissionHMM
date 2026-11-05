"""Preview: Figure 1 variant with one curve per fitted component, per state."""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm
from paper_regime_experiment import build_observations, label_bull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

RES = "notebooks/sp500/results/gmm_sweep_2004_2013"
prices = pd.read_csv("data/processed/sp500_index_price.csv", parse_dates=["YYYYMMDD"])
prices = prices.set_index("YYYYMMDD")["DlyPrcInd"].sort_index().astype(float)
features = build_observations(prices).dropna()
returns = prices.pct_change(fill_method=None)
state = pickle.load(open(f"{RES}/checkpoint.pkl", "rb"))
signals = pd.read_csv(f"{RES}/signals.csv", parse_dates=["date"])
start = pd.Timestamp(signals.loc[signals.refresh, "date"].min())
initial = features.loc[(features.index > start - pd.DateOffset(years=5))
                       & (features.index <= start)].index
hist = features.loc[initial[0]:signals.date.max()].index
SPEC_ORDER = sorted(state["branches"], key=lambda key: (int(key[3]), key))

fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
for ax, key in zip(axes.ravel(), SPEC_ORDER):
    model, scaler, pca = state["branches"][key]
    X = scaler.transform(features.loc[hist])
    if pca is not None:
        X = pca.transform(X)
    emission = model.emission
    _, filtered, _ = model.filter(X)
    bull = label_bull(model, X, returns.reindex(hist).to_numpy())
    obs = X[:, 0]
    lo, hi = np.quantile(obs, 0.004), np.quantile(obs, 0.996)
    bins = np.linspace(lo, hi, 120)
    dx = bins[1] - bins[0]
    grid = np.linspace(lo, hi, 1200)
    peak = 0.0
    for k in (0, 1):
        is_bull = (k == bull)
        base = np.array(matplotlib.colors.to_rgb("tab:blue" if is_bull else "tab:red"))
        w = emission.mixture_weights[k]
        mu = emission.means[k, :, 0]
        sd = np.sqrt(emission.covariances[k, :, 0, 0])
        M = len(w)
        # Shades of the state colour, ordered by component mean (dark = center).
        order = np.argsort(np.abs(mu - (w * mu).sum()))
        shades = cm.Blues(np.linspace(0.95, 0.40, M)) if is_bull else \
               cm.Reds(np.linspace(0.95, 0.40, M))
        label = f"state {k} ({'bull' if is_bull else 'bear'}) mixture"
        y = np.histogram(np.clip(obs, lo, hi), bins=bins,
                         weights=filtered[:, k])[0]
        y = y / (y.sum() * dx)
        peak = max(peak, y.max())
        ax.step(0.5 * (bins[1:] + bins[:-1]), y, where="mid", color=base,
                alpha=0.15, lw=0.8)
        mix = sum(w[j] * norm.pdf(grid, mu[j], sd[j]) for j in range(M))
        peak = max(peak, mix.max())
        ax.plot(grid, mix, color=base, lw=1.9,
                linestyle="-" if is_bull else "--", label=label)
        for rank, j in enumerate(order):
            curve = w[j] * norm.pdf(grid, mu[j], sd[j])
            peak = max(peak, curve.max())
            shade = shades[rank][:3]
            ax.plot(grid, curve, color=shade, lw=1.0, alpha=0.85,
                    linestyle=":" if not is_bull else "-")
            ax.plot([mu[j]], [0.0], "|", color=base, ms=9, mew=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, peak * 1.15)
    ax.set_title(key, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.65)
axes[0, 0].set_ylabel("density")
for ax in axes[1]:
    ax.set_xlabel("standardized daily return (no_pca) / PC1 coordinate (pca)",
                  fontsize=7)
handles = [
    Line2D([0], [0], color="tab:blue", lw=1.9, label="bull state mixture"),
    Line2D([0], [0], color="tab:red", lw=1.9, ls="--", label="bear state mixture"),
    Line2D([0], [0], color=cm.Blues(0.85), lw=1.0,
           label="bull components (darker = closer to state mean)"),
    Line2D([0], [0], color=cm.Reds(0.85), lw=1.0, ls=":",
           label="bear components (darker = closer to state mean)"),
    Line2D([0], [0], color="0.6", lw=0.9, label="posterior-weighted histogram"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.suptitle("Figure 1 variant: each fitted component as its own curve "
             "(scaled by its mixture weight w_j)\n"
             f"final refresh checkpoint, window {hist[0].date()} to {hist[-1].date()}; "
             "bold = state mixture, | ticks = component means", fontsize=10)
fig.tight_layout(rect=[0, 0.05, 1, 0.91])
fig.savefig(f"{RES}/artifacts/preview_fig1_component_curves.png", dpi=150)
print("saved preview_fig1_component_curves.png")
