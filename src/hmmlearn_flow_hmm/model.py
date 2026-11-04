"""Train Pyro flow emissions using hmmlearn's compiled HMM inference."""

from numbers import Integral

from hmmlearn import _hmmc
from hmmlearn.base import BaseHMM
import numpy as np
from scipy.special import logsumexp
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from flexible_emission_hmm import FlowEmission

class FlowHMM(BaseHMM):
    """Fit state-specific spline emissions with generalized Baum–Welch.

    Use hmmlearn's CPU inference and transition updates, with PyTorch emission
    evaluation and optimization on ``device``. Accept observations shaped (T, D)
    and optional sequence ``lengths``. Parameter codes are ``s`` for initial
    probabilities, ``t`` for transitions, and ``e`` for emissions. Excluding a
    code from ``params`` freezes it; excluding it from ``init_params`` preserves
    its existing initialization. Install the project's ``flows`` extra.
    """

    def __init__(
        self,
        n_components=3,
        *,
        count_bins=8,
        bound=5.0,
        num_layers=1,
        hidden_dims=(32, 32),
        learning_rate=1e-3,
        m_step_steps=5,
        batch_size=None,
        min_scale=1e-3,
        base_distribution="normal",
        base_df=5.0,
        device="cpu",
        dtype=None,
        startprob_prior=1.0,
        transmat_prior=1.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=100,
        tol=1e-4,
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
    ):
        """Configure flow emissions and hmmlearn training/inference options."""
        super().__init__(
            n_components=n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        # Preserve constructor arguments unchanged for sklearn estimator cloning.
        self.count_bins = count_bins
        self.bound = bound
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.m_step_steps = m_step_steps
        self.batch_size = batch_size
        self.min_scale = min_scale
        self.base_distribution = base_distribution
        self.base_df = base_df
        self.device = device
        self.dtype = dtype

    def _make_emission(self, seed):
        return FlowEmission(
            self.n_components,
            count_bins=self.count_bins,
            bound=self.bound,
            num_layers=self.num_layers,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            m_step_steps=self.m_step_steps,
            batch_size=self.batch_size,
            min_scale=self.min_scale,
            base_distribution=self.base_distribution,
            base_df=self.base_df,
            device=self.device,
            dtype=self.dtype,
            random_state=seed,
        )

    def _init(self, X, lengths=None):
        for name, value in (("n_components", self.n_components),
                            ("n_iter", self.n_iter)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        for name, codes in (("params", self.params), ("init_params", self.init_params)):
            if not isinstance(codes, str) or set(codes) - set("ste"):
                raise ValueError(f"{name} may contain only 's', 't', and 'e'")
        if self.implementation not in ("log", "scaling"):
            raise ValueError("implementation must be 'log' or 'scaling'")
        self._check_and_set_n_features(X)
        if self._needs_init("e", "emission_"):
            seed = (int(self.random_state) if isinstance(self.random_state, Integral)
                    else int(check_random_state(self.random_state).randint(2**32 - 1)))
            self.emission_ = self._make_emission(seed)
            self.emission_.initialize(X)
        # Initialize the flows before BaseHMM counts their trainable parameters.
        super()._init(X, lengths)

    def _check(self):
        super()._check()
        for values in (self.startprob_, self.transmat_):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("State probabilities must be finite and nonnegative")
        if not hasattr(self, "emission_") or self.emission_.flows is None:
            raise ValueError("Initialize flow emissions before using this model")
        if (self.emission_.n_states != self.n_components
                or self.emission_.n_features != self.n_features):
            raise ValueError("Flow dimensions must match n_components and n_features")

    def _get_n_fit_scalars_per_param(self):
        emission_count = sum(p.numel() for p in self.emission_.flows.parameters())
        return {"s": self.n_components - 1,
                "t": self.n_components * (self.n_components - 1),
                "e": emission_count}

    def _compute_log_likelihood(self, X):
        # Compiled hmmlearn kernels consume CPU double-precision emission scores.
        return np.asarray(self.emission_.log_prob(X), dtype=np.float64)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats.update(observations=[], responsibilities=[])
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )
        if "e" in self.params:
            # Preserve every sequence's own posterior weights without creating
            # transitions between sequence boundaries. Flow updates are pointwise.
            stats["observations"].append(X)
            stats["responsibilities"].append(posteriors.copy())

    def _do_mstep(self, stats):
        previous_transitions = self.transmat_.copy()
        super()._do_mstep(stats)
        if "t" in self.params:
            # Singleton sequences or unsupported states can produce empty rows.
            # Preserve their prior transition rows rather than returning an invalid HMM.
            unsupported = self.transmat_.sum(axis=1) == 0
            self.transmat_[unsupported] = previous_transitions[unsupported]
        if "e" in self.params:
            self.emission_.m_step(
                np.concatenate(stats["observations"]),
                np.concatenate(stats["responsibilities"]),
            )

    def _generate_sample_from_state(self, state, random_state=None):
        rng = check_random_state(random_state)
        seed = int(rng.randint(2**32 - 1))
        return self.emission_.sample(state, n_samples=1, random_state=seed)[0]

    def filter(self, X, initial_prior=None):
        """Return causal priors, filtered probabilities, and predictive log densities.

        Process one sequence using hmmlearn's compiled log-space forward pass.
        An optional state prior allows validation to continue from a training
        prefix without resetting the hidden chain. Unlike predict_proba, these
        probabilities do not condition on future observations.
        """
        check_is_fitted(self, "startprob_")
        self._check()
        X = check_array(X)
        prior = np.asarray(
            self.startprob_ if initial_prior is None else initial_prior,
            dtype=np.float64,
        )
        if (prior.shape != (self.n_components,) or not np.isfinite(prior).all()
                or (prior < 0).any() or not np.isclose(prior.sum(), 1)):
            raise ValueError("initial_prior must be a normalized state distribution")
        _, alpha = _hmmc.forward_log(
            prior, self.transmat_, self._compute_log_likelihood(X)
        )
        log_evidence = logsumexp(alpha, axis=1)
        if not np.isfinite(log_evidence).all():
            raise ValueError("An observation has zero or nonfinite predictive probability")
        filtered = np.exp(alpha - log_evidence[:, None])
        predicted = np.empty_like(filtered)
        predicted[0] = prior
        predicted[1:] = filtered[:-1] @ self.transmat_
        log_predictive = np.diff(log_evidence, prepend=0.0)
        return predicted, filtered, log_predictive

    def __getstate__(self):
        """Exclude imported module handles when pickling fitted estimators."""
        state = super().__getstate__().copy()
        emission = state.pop("emission_", None)
        if emission is not None:
            state["_saved_emission_state"] = {
                name: value for name, value in vars(emission).items()
                if name not in {"_torch", "_spline", "_autoregressive", "_arn"}
            }
        return state

    def __setstate__(self, state):
        """Restore runtime imports and fitted emission parameters after unpickling."""
        state = state.copy()
        emission_state = state.pop("_saved_emission_state", None)
        super().__setstate__(state)
        if emission_state is not None:
            self.emission_ = self._make_emission(0)
            vars(self.emission_).update(emission_state)
