"""Provide Pyro rational-quadratic spline emissions for generalized EM."""

from copy import deepcopy
from numbers import Integral

import numpy as np

from .emissions import EmissionModel
from .validation import as_sequence

class FlowEmission(EmissionModel):
    """Model each state's joint density with rational-quadratic splines.

    Use scalar splines in 1D and autoregressive splines otherwise. Parameters
    and emission optimization live on ``device``; the existing NumPy HMM
    performs state inference on CPU. Install the ``flows`` project extra.
    Set ``dtype`` to float32 or float64, as a string or torch dtype. By default,
    use float32 on MPS and float64 elsewhere. MPS cannot use float64.
    Choose a normal or unit-variance Student-t base; ``base_df`` must exceed two.
    """

    def __init__(
        self,
        n_states,
        *,
        count_bins=8,
        bound=3.0,
        num_layers=2,
        hidden_dims=(32, 32),
        learning_rate=1e-3,
        m_step_steps=20,
        batch_size=None,
        min_scale=1e-3,
        device="cpu",
        dtype=None,
        random_state=None,
        base_distribution="normal",
        base_df=5.0,
    ):
        """Configure state-specific flows and weighted likelihood optimization."""
        super().__init__(n_states)
        try:
            import torch
            from pyro.distributions.transforms import Spline, SplineAutoregressive
            from pyro.nn import AutoRegressiveNN
        except ImportError as exc:
            raise ImportError(
                "FlowEmission requires Pyro and PyTorch; install with "
                "`uv sync --extra flows` or `pip install '.[flows]'`."
            ) from exc
        for name, value in (
            ("count_bins", count_bins),
            ("num_layers", num_layers),
            ("m_step_steps", m_step_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if count_bins < 2 or count_bins >= 1000:
            raise ValueError("count_bins must be between 2 and 999")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be None or a positive integer")
        for name, value in (
            ("bound", bound),
            ("learning_rate", learning_rate),
            ("min_scale", min_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 1
            for v in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, Integral)
            or not 0 <= random_state < 2**32
        ):
            raise ValueError("random_state must be None or an integer in [0, 2**32)")
        self.count_bins = count_bins
        self.bound = bound
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.m_step_steps = m_step_steps
        self.batch_size = batch_size
        self.min_scale = min_scale
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float32 if self.device.type == "mps" else torch.float64
        if isinstance(dtype, str):
            dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64 (string or torch dtype)")
        if self.device.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS does not support float64; use dtype='float32'")
        self.dtype = dtype
        if base_distribution not in ("normal", "student_t"):
            raise ValueError("base_distribution must be 'normal' or 'student_t'")
        if not np.isfinite(base_df) or base_df <= 2:
            raise ValueError("base_df must be finite and greater than two")
        self.base_distribution = base_distribution
        self.base_df = float(base_df)
        self.random_state = random_state
        self._batch_rng = np.random.default_rng(random_state)
        self.flows = None
        self._torch = torch
        self._spline = Spline
        self._autoregressive = SplineAutoregressive
        self._arn = AutoRegressiveNN

    def initialize(self, X):
        """Initialize fresh flows with data-scaled, state-specific locations."""
        X = as_sequence(X)
        dimension = X.shape[1]
        if dimension > 1 and min(self.hidden_dims) < dimension:
            raise ValueError("Each hidden dimension must be at least n_features")
        torch = self._torch
        rng = np.random.default_rng(self.random_state)
        self._batch_rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_states, replace=len(X) < self.n_states)
        scale = np.maximum(X.std(axis=0), self.min_scale)
        flows = torch.nn.ModuleList()
        # Build on CPU inside a local RNG context without changing caller RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(rng.integers(2**31)))
            for index in indices:
                flow = torch.nn.Module()
                flow.location = torch.nn.Parameter(
                    torch.tensor(X[index], dtype=self.dtype)
                )
                flow.log_scale = torch.nn.Parameter(
                    torch.tensor(np.log(scale), dtype=self.dtype)
                )
                flow.transforms = torch.nn.ModuleList()
                for layer in range(self.num_layers):
                    if dimension == 1:
                        transform = self._spline(
                            dimension,
                            count_bins=self.count_bins,
                            bound=self.bound,
                            order="quadratic",
                        )
                        with torch.no_grad():
                            transform.unnormalized_widths.zero_()
                            transform.unnormalized_heights.zero_()
                            transform.unnormalized_derivatives.fill_(
                                float(np.log(np.expm1(1.0 - 1e-3)))
                            )
                    else:
                        permutation = torch.arange(dimension)
                        if layer % 2:
                            permutation = permutation.flip(0)
                        network = self._arn(
                            dimension,
                            list(self.hidden_dims),
                            param_dims=[
                                self.count_bins,
                                self.count_bins,
                                self.count_bins - 1,
                            ],
                            permutation=permutation,
                        )
                        transform = self._autoregressive(
                            dimension,
                            network,
                            count_bins=self.count_bins,
                            bound=self.bound,
                            order="quadratic",
                        )
                    flow.transforms.append(transform)
                flows.append(flow)
        self.n_features = dimension
        self.flows = flows.to(device=self.device, dtype=self.dtype)

    def _data(self, X):
        if self.flows is None:
            raise ValueError("Initialize the emission model before using it")
        X = as_sequence(X, self.n_features)
        return self._torch.tensor(X, dtype=self.dtype, device=self.device)

    def _density(self, flow, x):
        torch = self._torch
        log_scale = flow.log_scale.clamp(min=float(np.log(self.min_scale)))
        z = (x - flow.location) * torch.exp(-log_scale)
        log_det = -log_scale.sum().expand(len(x))
        # Transforms map observations to the base, making AR density evaluation fast.
        for transform in flow.transforms:
            transform.clear_cache()
            output = transform(z)
            jacobian = transform.log_abs_det_jacobian(z, output)
            if jacobian.ndim == z.ndim:
                jacobian = jacobian.sum(-1)
            log_det = log_det + jacobian
            z = output
        if self.base_distribution == "normal":
            base_log_prob = -0.5 * (z.square() + np.log(2 * np.pi))
        else:
            # Standardize the t base to unit variance, matching normal initialization.
            base = torch.distributions.StudentT(
                z.new_tensor(self.base_df),
                loc=z.new_tensor(0.0),
                scale=z.new_tensor(np.sqrt((self.base_df - 2) / self.base_df)),
            )
            base_log_prob = base.log_prob(z)
        return base_log_prob.sum(-1) + log_det

    def log_prob(self, X):
        """Return joint emission log densities as a NumPy array shaped (T, K)."""
        x = self._data(X)
        with self._torch.no_grad():
            values = self._torch.stack([self._density(f, x) for f in self.flows], -1)
        result = values.cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Flow emission log densities must be finite")
        return result

    def m_step(self, X, responsibilities):
        """Improve weighted likelihood with optional sampled emission batches.

        Fix full-sequence posterior weights during the numerical M-step. With
        batches, sample uniformly with replacement and check the full objective
        before and after all updates, restoring the start if it worsens. With
        full batches, retain the best iterate. None or batch_size >= len(X)
        uses the original full-batch method. Count Adam updates with m_step_steps,
        rather than epochs. Keep batch sampling local and reproducible.
        """
        x = self._data(X)
        weights = np.asarray(responsibilities, dtype=float)
        if (
            weights.shape != (len(x), self.n_states)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights.sum(axis=1), 1)
        ):
            raise ValueError("responsibilities must be normalized with shape (T, K)")
        torch = self._torch
        use_batches = self.batch_size is not None and self.batch_size < len(x)
        for state, flow in enumerate(self.flows):
            mass = weights[:, state].sum()
            if mass <= np.finfo(float).eps:
                continue
            w = torch.tensor(
                weights[:, state] / mass, dtype=self.dtype, device=self.device
            )
            optimizer = torch.optim.Adam(flow.parameters(), lr=self.learning_rate)
            best = deepcopy(flow.state_dict())
            with torch.no_grad():
                best_loss = float(-(w * self._density(flow, x)).sum())
            if not np.isfinite(best_loss):
                raise FloatingPointError("Initial weighted flow loss must be finite")
            try:
                for _ in range(self.m_step_steps):
                    optimizer.zero_grad()
                    if use_batches:
                        indices = torch.tensor(
                            self._batch_rng.integers(len(x), size=self.batch_size),
                            dtype=torch.long,
                            device=self.device,
                        )
                        # Preserve the full-state normalization. This is an
                        # unbiased estimate under uniform sampling, even when
                        # different batches have different posterior mass.
                        loss = -(len(x) / self.batch_size) * (
                            w[indices] * self._density(flow, x[indices])
                        ).sum()
                    else:
                        loss = -(w * self._density(flow, x)).sum()
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in flow.parameters()
                    ):
                        break
                    optimizer.step()
                    if use_batches:
                        continue
                    with torch.no_grad():
                        candidate = float(-(w * self._density(flow, x)).sum())
                    if not np.isfinite(candidate):
                        break
                    if candidate < best_loss:
                        best_loss, best = candidate, deepcopy(flow.state_dict())
                if use_batches:
                    # Avoid a full-data evaluation after every batch update.
                    # Accept only a finite improvement over the starting model.
                    with torch.no_grad():
                        candidate = float(-(w * self._density(flow, x)).sum())
                    if np.isfinite(candidate) and candidate < best_loss:
                        best = deepcopy(flow.state_dict())
            finally:
                flow.load_state_dict(best)
                for transform in flow.transforms:
                    transform.clear_cache()

    def sample(self, state, n_samples=1, random_state=None):
        """Draw observations from one initialized state as an (N, D) array."""
        if self.flows is None:
            raise ValueError("Initialize the emission model before sampling")
        if (
            isinstance(state, bool)
            or not isinstance(state, Integral)
            or not 0 <= state < self.n_states
        ):
            raise ValueError("state must index an existing state")
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, Integral)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        torch = self._torch
        rng = np.random.default_rng(random_state)
        shape = (n_samples, self.n_features)
        if self.base_distribution == "normal":
            draws = rng.normal(size=shape)
        else:
            draws = rng.standard_t(self.base_df, size=shape)
            draws *= np.sqrt((self.base_df - 2) / self.base_df)
        z = torch.tensor(draws, dtype=self.dtype, device=self.device)
        flow = self.flows[state]
        with torch.no_grad():
            for transform in reversed(flow.transforms):
                transform.clear_cache()
                z = transform.inv(z)
            scale = flow.log_scale.clamp(min=float(np.log(self.min_scale))).exp()
            samples = z * scale + flow.location
        return samples.cpu().numpy()
