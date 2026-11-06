"""HMM with normalizing-flow emissions, implemented with Pyro and PyTorch.

Each hidden state owns a small flow (base Normal -> rational-quadratic spline ->
learnable affine map). The marginal likelihood is computed exactly with the
forward algorithm in log space and maximized by gradient descent. The flow can be
regularized toward a plain Gaussian by penalizing the spline's distance from the
identity map.
"""

import torch
from torch import nn
import pyro.distributions as dist
import pyro.distributions.transforms as T

from .torch_hmm import EmissionHMM, StateEmission

# Unconstrained derivative value giving slope 1 (min_derivative is 1e-3).
_IDENTITY_DERIVATIVE = torch.tensor(1 - 1e-3).expm1().log().item()


class FlowEmission(StateEmission):
    """One state's emission density: a 1-D Pyro flow on a standard Normal."""

    def __init__(self, dim=1, count_bins=8, bound=3.0, identity_init=False):
        super().__init__(dim)
        self.spline = T.spline(dim, count_bins=count_bins, bound=bound)
        self.loc = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("base_loc", torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim))
        if identity_init:
            self.reset_spline_to_identity()

    def reset_spline_to_identity(self):
        """Set the spline to the identity map, so the flow starts Gaussian."""
        with torch.no_grad():
            self.spline.unnormalized_widths.zero_()
            self.spline.unnormalized_heights.zero_()
            self.spline.unnormalized_derivatives.fill_(_IDENTITY_DERIVATIVE)

    def penalty(self):
        """Return the squared distance of the spline from the identity map."""
        s = self.spline
        return (s.unnormalized_widths.square().sum()
                + s.unnormalized_heights.square().sum()
                + (s.unnormalized_derivatives - _IDENTITY_DERIVATIVE)
                .square().sum())

    def distribution(self):
        """Return the flow as a Pyro TransformedDistribution."""
        base = dist.Normal(self.base_loc, self.base_scale).to_event(1)
        affine = T.AffineTransform(self.loc, self.log_scale.exp(), event_dim=1)
        return dist.TransformedDistribution(base, [self.spline, affine])

    def log_prob(self, x):
        """Return log density of x with shape (T, dim) as shape (T,)."""
        return self.distribution().log_prob(x)

    def init_from_data(self, x):
        """Match the affine map to the mean and standard deviation of x."""
        with torch.no_grad():
            self.loc.copy_(x.mean(0))
            self.log_scale.copy_(x.std(0).clamp_min(1e-2).log())


class HMMFlow(EmissionHMM):
    """K-state HMM whose emissions are all normalizing flows."""

    def __init__(self, n_states, dim=1, count_bins=8, bound=3.0, seed=0,
                 identity_init=False):
        super().__init__(
            [FlowEmission(dim, count_bins, bound, identity_init)
             for _ in range(n_states)], seed=seed)
        self.dim = dim
