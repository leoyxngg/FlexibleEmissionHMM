"""Preview: two figures, 4x3 each - no_pca leg and pca leg."""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm
from paper_regime_experiment import build_observations, label_bull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
daily = returns.reindex(hist).to_numpy()

for leg, xlabel in (("no_pca", "standardized daily return"),
                    ("pca", "PC1 score")):
    keys = sorted([k for k in state["branches"]
                   if (k.endswith("_no_pca")) == (leg == "no_pca")],
                  key=lambda key: int(key[3]))
    fig, axes = plt.subplots(len(keys), 3, figsize=(13.5, 2.5 * len(keys)))
    for row, key in zip(axes, keys):
        model, scaler, pca = state["branches"][key]
        X = scaler.transform(features.loc[hist])
        if pca is not None:
            X = pca.transform(X)
        emission = model.emission
        _, filtered, _ = model.filter(X)
        bull = label_bull(model, X, daily)
        obs = X[:, 0]
        lo, hi = np.quantile(obs, 0.004), np.quantile(obs, 0.996)
        bins = np.linspace(lo, hi, 120)
        dx = bins[1] - bins[0]
        grid = np.linspace(np.quantile(obs, 2e-3), np.quantile(obs, 0.998), 1200)
        per_state = {}
        for k in (0, 1):
            w = emission.mixture_weights[k]
            mu = emission.means[k, :, 0]
            sd = np.sqrt(emission.covariances[k, :, 0, 0])
            per_state[k] = dict(
                base=np.array(mcolors.to_rgb("tab:blue" if k == bull else "tab:red")),
                bull=bool(k == bull),
                mix=sum(w[j] * norm.pdf(grid, mu[j], sd[j]) for j in range(len(w))),
                hist=np.histogram(np.clip(obs, lo, hi), bins=bins,
                                  weights=filtered[:, k])[0]
                     / (filtered[:, k].sum() * dx),
                w=w, mu=mu, sd=sd)
        ylim = max(max(s["mix"].max(), s["hist"].max())
                   for s in per_state.values()) * 1.25
        bull_k = int(bull)
        m = emission.n_components
        for col, k in ((0, bull_k), (1, 1 - bull_k)):
            ax, s = row[col], per_state[k]
            cmap = cm.Blues if s["bull"] else cm.Reds
            order = np.argsort(np.abs(s["mu"] - (s["w"] * s["mu"]).sum()))
            shades = cmap(np.linspace(0.95, 0.40, m))
            ax.step(0.5 * (bins[1:] + bins[:-1]), s["hist"], where="mid",
                    color=s["base"], alpha=0.15, lw=0.8)
            for rank, j in enumerate(order):
                ax.plot(grid, s["w"][j] * norm.pdf(grid, s["mu"][j], s["sd"][j]),
                        color=shades[rank], lw=1.2, alpha=0.9,
                        label=f"w={s['w'][j]:.2f}  mu={s['mu'][j]:+.2f}  sd={s['sd'][j]:.2f}")
                ax.plot([s["mu"][j]], [0.0], "|", color=s["base"], ms=8, mew=1.3)
            ax.set_ylim(0, ylim)
            ax.set_xlim(lo, hi)
            ax.set_title(f"{'bull' if s['bull'] else 'bear'} state components "
                         f"(M={m}, darker = closer to state mean)", fontsize=8)
            ax.legend(fontsize=6, loc="upper right", framealpha=0.65)
            ax.tick_params(labelsize=6.5)
        ax = row[2]
        for k in (0, 1):
            s = per_state[k]
            ax.plot(grid, s["mix"], color=s["base"], lw=1.8,
                    linestyle="-" if s["bull"] else "--",
                    label=f"state {k} ({'bull' if s['bull'] else 'bear'}) mixture")
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
        ax.set_xlabel(xlabel, fontsize=7)
    fig.suptitle(f"{leg} leg: bull components | bear components | aggregate mixtures\n"
                 f"component curves are w_j * N(mu_j, sd_j) (they sum to the state "
                 f"mixture); each row shares one density axis; checkpoint, window "
                 f"{hist[0].date()} to {hist[-1].date()}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    name = f"{RES}/artifacts/preview_fig1_rows_{leg}.png"
    fig.savefig(name, dpi=150)
    plt.close(fig)
    print("saved", name)
