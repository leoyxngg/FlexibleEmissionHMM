from abc import ABC, abstractmethod
import numpy as np
import csv

class MarkovProcessDataGenerator(ABC):
    """
    - emission_params: the list of parameters for different states in order as a np array of np arrays
    - states: the labels of hidden states/regimes, passed as a np array of numbers 
      starting at 0
    - initial_probs: the initial probabilities for the Markov chain
    - trans_matrix: the transitional matrix for the Markov chain
    - data_count: the number of data points to be generated
    """
    def __init__(self, emission_params, states, initial_probs, trans_matrix, data_count,
                 random_state=None):
        self.emission_params = emission_params
        self.states = states
        self.initial_probs = initial_probs
        self.trans_matrix = trans_matrix
        self.data_count = data_count
        self.rng = np.random if random_state is None else np.random.default_rng(random_state)

    def sample_categorical(self, curr_state=0, init=False):
        if not init:
            trans_probs = self.trans_matrix[curr_state]
            sample_state = self.rng.choice(self.states,p=trans_probs)
            return sample_state
        return self.rng.choice(self.states,p=self.initial_probs)

    @abstractmethod
    def sample_dist(self, parameters):
        pass

    def sample_state_path(self):
        """Draw only the chain, so experiments can share the same hidden path."""
        states = np.asarray(self.states)
        start = np.asarray(self.initial_probs, dtype=float)
        transitions = np.asarray(self.trans_matrix, dtype=float)
        K = len(states)
        if (K == 0 or not np.array_equal(states, np.arange(K))
                or start.shape != (K,) or transitions.shape != (K, K)):
            raise ValueError("States must be 0,...,K-1 with probabilities shaped (K,) and (K,K)")
        for probabilities in (start, transitions):
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or not np.allclose(probabilities.sum(axis=-1), 1)):
                raise ValueError("Probabilities must be finite, nonnegative and normalized")
        if not isinstance(self.data_count, (int, np.integer)) or self.data_count < 1:
            raise ValueError("data_count must be a positive integer")
        path = np.empty(self.data_count, dtype=int)
        path[0] = self.sample_categorical(init=True)
        for t in range(1, len(path)):
            path[t] = self.sample_categorical(curr_state=path[t - 1])
        return path

    def sample_emissions(self, path, *, return_components=False):
        """Draw emissions on a supplied path; returns observations shaped (N,D).

        Optional component labels distinguish independent within-state mixture
        draws, not HMM states. Non-mixture emissions have component label -1.
        This leaves the legacy generate_data()/write_csv() API unchanged.
        """
        path = np.asarray(path)
        if (path.ndim != 1 or len(path) == 0 or not np.issubdtype(path.dtype, np.integer)
                or (path < 0).any() or (path >= len(self.states)).any()):
            raise ValueError("path must be a nonempty vector of valid integer state indices")
        values, components = [], []
        for state in path:
            values.append(np.atleast_1d(self.sample_dist(self.emission_params[state])))
            components.append(getattr(self, "last_component", -1))
        X = np.stack(values)
        return (X, np.asarray(components, dtype=int)) if return_components else X
    
    def generate_data(self):
        rows = []
        current_state = self.sample_categorical(init=True)
        for t in range(self.data_count):
            if t > 0:
                current_state = self.sample_categorical(
                    curr_state=current_state
                )
            parameters = self.emission_params[current_state]
            data = self.sample_dist(parameters)
            rows.append((t, current_state, data))
        return rows

    def write_csv(self, filename, rows):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "state", "data"])
            writer.writerows(rows)

class GaussianEmissionGenerator(MarkovProcessDataGenerator):
    def __init__(self, emission_params, states, initial_probs, trans_matrix, data_count,
                 random_state=None):
        super().__init__(emission_params, states, initial_probs, trans_matrix, data_count,
                         random_state=random_state)

    def sample_dist(self, parameters):
        mean, std = parameters
        return self.rng.normal(loc=mean,scale=std)


class _MomentMatchedEmissionGenerator(MarkovProcessDataGenerator):
    """Emissions parameterized by population mean and standard deviation.

    Shape parameters follow ``(mean, std)`` in each state's parameter tuple.
    Standardization uses population moments, never the generated sample.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameters in self.emission_params:
            self._validate_parameters(parameters)

    def _validate_parameters(self, parameters):
        mean, std = parameters[:2]
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("Emission mean must be finite and std must be positive and finite")


class StudentTEmissionGenerator(_MomentMatchedEmissionGenerator):
    """Parameters: ``(mean, std, df)``; df > 2 ensures finite variance."""

    def _validate_parameters(self, parameters):
        mean, std, df = parameters
        super()._validate_parameters(parameters)
        if not np.isfinite(df) or df <= 2:
            raise ValueError("Student-t df must be finite and greater than 2")

    def sample_dist(self, parameters):
        mean, std, df = parameters
        standardized = self.rng.standard_t(df) * np.sqrt((df - 2) / df)
        return mean + std * standardized


class MultimodalEmissionGenerator(_MomentMatchedEmissionGenerator):
    """Gaussian mixture with parameters ``(mean, std, weights, means, stds)``.

    The component parameters describe the unstandardized mixture. Its output
    is shifted and scaled to the requested overall population mean and std.
    Components are emission modes, not additional hidden Markov states.
    """

    def _validate_parameters(self, parameters):
        mean, std, weights, means, stds = parameters
        super()._validate_parameters(parameters)
        weights, means, stds = [np.asarray(x, dtype=float) for x in (weights, means, stds)]
        if (
            weights.ndim != 1 or len(weights) < 2
            or means.shape != weights.shape or stds.shape != weights.shape
            or not all(np.isfinite(x).all() for x in (weights, means, stds))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, rtol=1e-12, atol=1e-12)
            or np.any(stds <= 0)
        ):
            raise ValueError("Mixture requires matching finite vectors, positive weights summing to 1, and positive stds")

    def sample_dist(self, parameters):
        mean, std, weights, means, stds = parameters
        weights, means, stds = [np.asarray(x, dtype=float) for x in (weights, means, stds)]
        center = np.dot(weights, means)
        variance = np.dot(weights, stds ** 2 + (means - center) ** 2)
        component = self.rng.choice(len(weights), p=weights)
        self.last_component = int(component)
        value = self.rng.normal(means[component], stds[component])
        return mean + std * (value - center) / np.sqrt(variance)


class SkewNormalEmissionGenerator(_MomentMatchedEmissionGenerator):
    """Parameters: ``(mean, std, alpha)``; alpha controls signed skewness."""

    def _validate_parameters(self, parameters):
        mean, std, alpha = parameters
        super()._validate_parameters(parameters)
        if not np.isfinite(alpha):
            raise ValueError("Skew-normal alpha must be finite")

    def sample_dist(self, parameters):
        mean, std, alpha = parameters
        delta = alpha / np.hypot(1, alpha)
        value = delta * abs(self.rng.normal()) + self.rng.normal() / np.hypot(1, alpha)
        center = delta * np.sqrt(2 / np.pi)
        scale = np.sqrt(1 - 2 * delta ** 2 / np.pi)
        return mean + std * (value - center) / scale


class GammaEmissionGenerator(_MomentMatchedEmissionGenerator):
    """Parameters: ``(mean, std, shape)`` for a shifted/scaled gamma.

    Support starts at mean - std * sqrt(shape), which may be negative.
    """

    def _validate_parameters(self, parameters):
        mean, std, shape = parameters
        super()._validate_parameters(parameters)
        if not np.isfinite(shape) or shape <= 0:
            raise ValueError("Gamma shape must be positive and finite")

    def sample_dist(self, parameters):
        mean, std, shape = parameters
        return mean + std * (self.rng.gamma(shape) - shape) / np.sqrt(shape)


class ContaminatedGaussianEmissionGenerator(_MomentMatchedEmissionGenerator):
    """Parameters: ``(mean, std, contamination, outlier_scale)``.

    Mix N(0, 1) with N(0, outlier_scale**2), then match population moments.
    """

    def _validate_parameters(self, parameters):
        mean, std, contamination, outlier_scale = parameters
        super()._validate_parameters(parameters)
        if (
            not np.isfinite([contamination, outlier_scale]).all()
            or not 0 <= contamination <= 1 or outlier_scale <= 1
        ):
            raise ValueError("Contamination must be in [0, 1] and outlier_scale must be finite and > 1")

    def sample_dist(self, parameters):
        mean, std, contamination, outlier_scale = parameters
        self.last_component = int(self.rng.random() < contamination)
        scale = outlier_scale if self.last_component else 1
        variance = (1 - contamination) + contamination * outlier_scale ** 2
        return mean + std * self.rng.normal(scale=scale) / np.sqrt(variance)


class MultivariateGaussianEmissionGenerator(MarkovProcessDataGenerator):
    """Parameters per state: (mean_vector, full_covariance_matrix)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dimensions = set()
        for mean, covariance in self.emission_params:
            mean, covariance = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
            if (mean.ndim != 1 or len(mean) == 0 or not np.isfinite(mean).all()
                    or covariance.shape != (len(mean), len(mean))
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T)):
                raise ValueError("Provide a finite mean vector and symmetric (D,D) covariance")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Covariance must be positive definite") from exc
            dimensions.add(len(mean))
        if len(dimensions) != 1:
            raise ValueError("All states must have the same observation dimension")

    def sample_dist(self, parameters):
        mean, covariance = parameters
        return self.rng.multivariate_normal(mean, covariance)


class StateDependentEmissionGenerator(MarkovProcessDataGenerator):
    """Allow different emission families in different hidden states.

    Each emission specification is ``(family, parameters)``. Parameters are the
    same tuples accepted by the corresponding existing generator. Example:
    [("gaussian", (0, 1)),
     ("multimodal", (0, 1, [.5, .5], [-.9, .9], [sqrt(.19)] * 2))].
    Use sample_state_path()/sample_emissions() for independently seeded paths
    and emissions; generate_data() remains available for legacy row output.
    """

    def __init__(self, emission_params, states, initial_probs, trans_matrix,
                 data_count, random_state=None):
        registry = {
            "gaussian": GaussianEmissionGenerator,
            "multivariate_gaussian": MultivariateGaussianEmissionGenerator,
            "student_t": StudentTEmissionGenerator,
            "skew_normal": SkewNormalEmissionGenerator,
            "multimodal": MultimodalEmissionGenerator,
            "gamma": GammaEmissionGenerator,
            "contaminated_gaussian": ContaminatedGaussianEmissionGenerator,
        }
        super().__init__(emission_params, states, initial_probs, trans_matrix, data_count,
                         random_state=np.random.default_rng(random_state))
        if len(emission_params) != len(states):
            raise ValueError("Provide one emission specification per state")
        self._samplers = {}
        dimensions = set()
        for family, parameters in emission_params:
            if family not in registry:
                raise ValueError(f"Unknown emission family: {family}")
            if family == "gaussian":
                mean, std = parameters
                if not np.isfinite([mean, std]).all() or std <= 0:
                    raise ValueError("Gaussian mean must be finite and std finite and positive")
            dimensions.add(len(parameters[0]) if family == "multivariate_gaussian" else 1)
            same_family = [params for name, params in emission_params if name == family]
            if family not in self._samplers:
                count = len(same_family)
                self._samplers[family] = registry[family](
                    same_family, np.arange(count), np.full(count, 1 / count),
                    np.full((count, count), 1 / count), data_count, random_state=self.rng)
        if len(dimensions) != 1:
            raise ValueError("All states must have the same observation dimension")

    def sample_dist(self, parameters):
        family, values = parameters
        sampler = self._samplers[family]
        value = sampler.sample_dist(values)
        self.last_component = getattr(sampler, "last_component", -1)
        return value
