"""Preview: one row per spec; bull components | bear components | aggregate."""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm
from paper_regime_experiment import build_observations, label_bull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
SPEC_ORDER = sorted(state["branches"],
                    key=lambda key: (key.endswith("_pca"), int(key[3]), key))

fig, axes = plt.subplots(len(SPEC_ORDER), 3, figsize=(13.5, 2.5 * len(SPEC_ORDER)))
for row, key in zip(axes, SPEC_ORDER):
    model, scaler, pca = state["branches"][key]
    X = scaler.transform(features.loc[hist])
    if pca is not None:
        X = pca.transform(X)
    emission = model.emission
    _, filtered, _ = model.filter(X)
    bull = label_bull(model, X, returns.reindex(hist).to_numpy())
    obs = X[:, 0]
    lo, hi = np.quantile(obs, 0.004), np.quantile(obs, 0.996)
    grid = np.linspace(lo, hi, 1200)
    bins = np.linspace(lo, hi, 120)
    dx = bins[1] - bins[0]
    per_state = {}
    for k in (0, 1):
        is_bull = (k == bull)
        base = np.array(matplotlib.colors.to_rgb("tab:blue" if is_bull else "tab:red"))
        w = emission.mixture_weights[k]
        mu = emission.means[k, :, 0]
        sd = np.sqrt(emission.covariances[k, :, 0, 0])
        mix = sum(w[j] * norm.pdf(grid, mu[j], sd[j]) for j in range(len(w)))
        per_state[k] = dict(base=base, w=w, mu=mu, sd=sd, mix=mix,
                            is_bull=is_bull,
                            hist=np.histogram(np.clip(obs, lo, hi), bins=bins,
                                              weights=filtered[:, k])[0]
                                 / (filtered[:, k].sum() * dx))
    m = emission.n_components
    ylim = 1.0  # fixed density axis across every panel
    for col, k in ((0, int(bull)), (1, int(1 - bull))):
        ax = row[col]
        s = per_state[k]
        cmap = cm.Blues if s["is_bull"] else cm.Reds
        order = np.argsort(np.abs(s["mu"] - (s["w"] * s["mu"]).sum()))
        shades = cmap(np.linspace(0.95, 0.40, m))
        ax.step(0.5 * (bins[1:] + bins[:-1]), s["hist"], where="mid",
                color=s["base"], alpha=0.15, lw=0.8)
        for rank, j in enumerate(order):
            curve = s["w"][j] * norm.pdf(grid, s["mu"][j], s["sd"][j])
            ax.plot(grid, curve, color=shades[rank], lw=1.2, alpha=0.9,
                    label=f"w={s['w'][j]:.2f}  mu={s['mu'][j]:+.2f}  sd={s['sd'][j]:.2f}")
            ax.plot([s["mu"][j]], [0.0], "|", color=s["base"], ms=8, mew=1.3)
        ax.set_ylim(0, ylim)
        ax.set_xlim(lo, hi)
        ax.set_title(f"{'bull' if s['is_bull'] else 'bear'} state components "
                     f"(M={m}, darker = closer to state mean)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.65)
        ax.tick_params(labelsize=6.5)
    ax = row[2]
    for k in (0, 1):
        s = per_state[k]
        ax.plot(grid, s["mix"], color=s["base"], lw=1.8,
                linestyle="-" if s["is_bull"] else "--",
                label=f"state {k} ({'bull' if s['is_bull'] else 'bear'}) mixture")
        ax.step(0.5 * (bins[1:] + bins[:-1]), s["hist"], where="mid",
                color=s["base"], alpha=0.22, lw=0.8)
    ax.set_ylim(0, ylim)
    ax.set_xlim(lo, hi)
    ax.set_title("aggregate mixtures vs state-weighted histograms", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.65)
    ax.tick_params(labelsize=6.5)
    row[0].set_ylabel(f"{key}\ndensity", fontsize=8)
    for ax in row[1:]:
        ax.tick_params(axis="y", labelleft=False, left=False)
for ax in axes[-1]:
    ax.set_xlabel("standardized daily return (no_pca) / PC1 coordinate (pca)",
                  fontsize=7)
for lo, hi, text in ((0, 3, "no_pca leg (x: standardized daily return)"),
                     (4, 7, "pca leg (x: PC1 score)")):
    mid = (lo + hi) // 2
    axes[mid, 0].annotate(text, xy=(0, 1.12), xycoords="axes fraction",
                          ha="left", va="bottom", fontsize=9, fontweight="bold")
axes[3, 0].axhline(1.0, color="0.6", lw=0.8, clip_on=False)
fig.suptitle("Per-spec component decomposition (no_pca legs, then pca legs): bull components | bear components | "
             "aggregate mixtures\n"
             f"component curves are w_j * N(mu_j, sd_j) (they sum to the state "
             f"mixture); final refresh checkpoint, window {hist[0].date()} to "
             f"{hist[-1].date()}", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"{RES}/artifacts/preview_fig1_rows.png", dpi=150)
print("saved preview_fig1_rows.png")
