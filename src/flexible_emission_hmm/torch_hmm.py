"""Gradient-trained HMM whose per-state emissions are pluggable PyTorch modules.

An emission is any :class:`StateEmission`. :class:`EmissionHMM` takes a list with
one emission per state, so families can be mixed freely, for example one
Student-t state next to one Gaussian state. New families only need to implement
``log_prob`` (and optionally ``init_from_data`` and ``penalty``).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, StudentT


class StateEmission(nn.Module):
    """Base class for one state's emission density over x of shape (T, dim)."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def log_prob(self, x):
        """Return the log density of x with shape (T, dim) as shape (T,)."""
        raise NotImplementedError

    def init_from_data(self, x):
        """Initialise parameters from the observations x assigned to this state."""

    def penalty(self):
        """Return a scalar regularisation penalty (zero unless overridden)."""
        return torch.zeros((), device=next(self.parameters()).device)


class GaussianStateEmission(StateEmission):
    """Independent Gaussian emission per feature."""

    def __init__(self, dim=1):
        super().__init__(dim)
        self.loc = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def log_prob(self, x):
        """Return the Gaussian log density summed over features."""
        return Normal(self.loc, self.log_scale.exp()).log_prob(x).sum(-1)

    def init_from_data(self, x):
        """Match the mean and standard deviation of x."""
        with torch.no_grad():
            self.loc.copy_(x.mean(0))
            self.log_scale.copy_(x.std(0).clamp_min(1e-2).log())


class StudentTStateEmission(StateEmission):
    """Independent Student-t emission per feature with learnable degrees of freedom."""

    MIN_DF = 2.1

    def __init__(self, dim=1, init_df=5.0):
        super().__init__(dim)
        self.loc = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        raw = torch.tensor(float(init_df) - self.MIN_DF).expm1().log()
        self.raw_df = nn.Parameter(raw)

    @property
    def df(self):
        """Return the degrees of freedom, constrained above MIN_DF."""
        return self.MIN_DF + F.softplus(self.raw_df)

    def log_prob(self, x):
        """Return the Student-t log density summed over features."""
        return StudentT(self.df, self.loc, self.log_scale.exp()).log_prob(x).sum(-1)

    def init_from_data(self, x):
        """Match the mean and spread of x, scaled for the initial df."""
        with torch.no_grad():
            self.loc.copy_(x.mean(0))
            shrink = ((self.df - 2) / self.df).sqrt()
            self.log_scale.copy_((x.std(0).clamp_min(1e-2) * shrink).log())


class EmissionHMM(nn.Module):
    """HMM with one pluggable emission module per state.

    The marginal likelihood is exact (log-space forward algorithm) and is
    maximised by gradient descent, optionally with an emission penalty.
    """

    def __init__(self, emissions, seed=0):
        super().__init__()
        if len(emissions) < 1:
            raise ValueError("Provide one emission per state")
        self.n_states = len(emissions)
        gen = torch.Generator().manual_seed(seed)
        k = self.n_states
        self.trans_logits = nn.Parameter(
            0.1 * torch.randn(k, k, generator=gen) + 2.0 * torch.eye(k))
        self.init_logits = nn.Parameter(torch.zeros(k))
        self.emissions = nn.ModuleList(emissions)

    @property
    def transition_matrix(self):
        """Return the row-stochastic transition matrix."""
        return torch.softmax(self.trans_logits, dim=-1)

    @property
    def initial_probs(self):
        """Return the initial state distribution."""
        return torch.softmax(self.init_logits, dim=-1)

    def _prepare(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(len(x), -1)
        return x.to(self.trans_logits.device)

    def init_from_data(self, x):
        """Initialise each emission on one quantile chunk of the first feature."""
        x = self._prepare(x)
        chunks = torch.chunk(x[x[:, 0].argsort()], self.n_states)
        for emission, chunk in zip(self.emissions, chunks):
            emission.init_from_data(chunk)

    def penalty(self):
        """Return the summed regularisation penalty of all emissions."""
        return sum(em.penalty() for em in self.emissions)

    def log_emissions(self, x):
        """Return per-state log densities with shape (T, K)."""
        return torch.stack([em.log_prob(x) for em in self.emissions], dim=-1)

    def log_likelihood(self, x):
        """Return the exact log p(x) via the log-space forward algorithm."""
        return self._forward(self.log_emissions(x))[0]

    def _forward(self, logb):
        log_a = torch.log_softmax(self.trans_logits, dim=-1)
        alpha = torch.log_softmax(self.init_logits, dim=-1) + logb[0]
        alphas = [alpha]
        for t in range(1, len(logb)):
            alpha = torch.logsumexp(alpha[:, None] + log_a, dim=0) + logb[t]
            alphas.append(alpha)
        return torch.logsumexp(alpha, dim=0), torch.stack(alphas)

    def fit(self, x, n_steps=300, lr=0.02, clip=10.0, regularization=0.0,
            verbose=False):
        """Maximize log-likelihood minus ``regularization * penalty`` with Adam.

        Return the log-likelihood history.
        """
        x = self._prepare(x)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        for step in range(n_steps):
            opt.zero_grad()
            ll = self.log_likelihood(x)
            loss = -ll + regularization * self.penalty()
            (loss / len(x)).backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            opt.step()
            history.append(ll.item())
            if verbose and step % 50 == 0:
                print(f"step {step}: loglik {ll.item():.2f}")
        return history

    @torch.no_grad()
    def posterior(self, x):
        """Return smoothed state probabilities with shape (T, K)."""
        x = self._prepare(x)
        logb = self.log_emissions(x)
        log_a = torch.log_softmax(self.trans_logits, dim=-1)
        _, alphas = self._forward(logb)
        beta = torch.zeros(self.n_states, device=logb.device)
        betas = [beta]
        for t in range(len(x) - 1, 0, -1):
            beta = torch.logsumexp(log_a + (logb[t] + beta)[None, :], dim=1)
            betas.append(beta)
        betas = torch.stack(betas[::-1])
        return torch.softmax(alphas + betas, dim=-1)

    @torch.no_grad()
    def viterbi(self, x):
        """Return the most likely state path as a LongTensor of shape (T,)."""
        x = self._prepare(x)
        logb = self.log_emissions(x)
        log_a = torch.log_softmax(self.trans_logits, dim=-1)
        delta = torch.log_softmax(self.init_logits, dim=-1) + logb[0]
        back = []
        for t in range(1, len(x)):
            delta, arg = (delta[:, None] + log_a).max(dim=0)
            delta = delta + logb[t]
            back.append(arg)
        path = [int(delta.argmax())]
        for arg in reversed(back):
            path.append(int(arg[path[-1]]))
        return torch.tensor(path[::-1])

    @torch.no_grad()
    def score(self, x):
        """Return the total log-likelihood of x as a float."""
        return self.log_likelihood(self._prepare(x)).item()

    @torch.no_grad()
    def filter(self, x, prior=None):
        """Run the forward filter with fixed parameters.

        ``prior`` is the state distribution before the first observation and
        defaults to the initial distribution. Return the one-step-ahead log
        p(x_t | x_<t) with shape (T,), the filtered state probabilities with
        shape (T, K) and the next-step state prior with shape (K,).
        """
        x = self._prepare(x)
        logb = self.log_emissions(x)
        trans = self.transition_matrix
        belief = self.initial_probs if prior is None else torch.as_tensor(
            prior, dtype=torch.float32, device=x.device)
        scores, filtered = [], []
        for t in range(len(x)):
            joint = belief.log() + logb[t]
            scores.append(torch.logsumexp(joint, dim=0))
            posterior = torch.softmax(joint, dim=0)
            filtered.append(posterior)
            belief = posterior @ trans
        return (torch.stack(scores).cpu().numpy(),
                torch.stack(filtered).cpu().numpy(), belief.cpu().numpy())
