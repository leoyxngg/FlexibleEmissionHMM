import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal

from flexible_emission_hmm import (
    BaumWelchTrainer, GaussianEmission, GaussianMixtureEmission, HMM,
)


def sample_sequence(seed, n_samples=160):
    rng = np.random.default_rng(seed)
    transitions = np.array([[0.92, 0.08], [0.12, 0.88]])
    means = np.array([[[-4., -2.], [-1., 1.]], [[2., -1.], [5., 2.]]])
    covariance = np.array([[0.7, 0.2], [0.2, 0.5]])
    mixture_weights = np.array([[0.7, 0.3], [0.4, 0.6]])
    X = np.empty((n_samples, 2))
    state = 0
    for t in range(n_samples):
        component = rng.choice(2, p=mixture_weights[state])
        X[t] = rng.multivariate_normal(means[state, component], covariance)
        state = rng.choice(2, p=transitions[state])
    return X


class GaussianMixtureEmissionTests(unittest.TestCase):
    def configured_emission(self):
        emission = GaussianMixtureEmission(2, 2, random_state=7)
        emission.initialize(np.array([[-2.], [0.], [1.], [3.]]))
        emission.mixture_weights = np.array([[0.3, 0.7], [0.6, 0.4]])
        emission.means = np.array([[[-2.], [0.5]], [[1.], [3.]]])
        emission.covariances = np.array([[[[0.8]], [[1.2]]], [[[0.6]], [[1.5]]]])
        return emission

    def test_density_matches_explicit_mixture(self):
        emission = GaussianMixtureEmission(2, 3, random_state=7)
        X = sample_sequence(8)
        emission.initialize(X)
        emission.mixture_weights = np.array([[0.1, 0.3, 0.6], [0.8, 0.2, 0.]])
        expected = np.zeros((len(X), 2))
        for k in range(2):
            for m in range(3):
                expected[:, k] += emission.mixture_weights[k, m] * multivariate_normal.pdf(
                    X, mean=emission.means[k, m], cov=emission.covariances[k, m],
                )
        assert_allclose(emission.log_prob(X), np.log(expected), atol=1e-12)

    def test_joint_update_matches_enumerated_latent_assignments(self):
        X = np.array([[-1.7], [0.2], [1.3], [2.8]])
        emission = self.configured_emission()
        model = HMM(emission)
        model.states.start = np.array([0.7, 0.3])
        model.states.transitions = np.array([[0.85, 0.15], [0.25, 0.75]])

        # Enumerate every complete (state, component) path independently of
        # forward/backward and the implementation's conditional responsibilities.
        joint = np.zeros((len(X), 2, 2))
        total = 0.
        for path in itertools.product(range(4), repeat=len(X)):
            states = np.array(path) // 2
            components = np.array(path) % 2
            probability = model.states.start[states[0]]
            for t, (k, m) in enumerate(zip(states, components)):
                if t:
                    probability *= model.states.transitions[states[t - 1], k]
                probability *= emission.mixture_weights[k, m] * multivariate_normal.pdf(
                    X[t], mean=emission.means[k, m], cov=emission.covariances[k, m],
                )
            total += probability
            for t, (k, m) in enumerate(zip(states, components)):
                joint[t, k, m] += probability
        joint /= total
        result = model.inference.infer(model.states, emission.log_prob(X))
        assert_allclose(result.log_likelihood, np.log(total), atol=1e-12)
        assert_allclose(result.responsibilities, joint.sum(axis=2), atol=1e-12)

        masses = joint.sum(axis=0)
        expected_weights = masses / masses.sum(axis=1, keepdims=True)
        expected_means = np.empty((2, 2, 1))
        expected_covariances = np.empty((2, 2, 1, 1))
        for k in range(2):
            for m in range(2):
                expected_means[k, m, 0] = np.average(X[:, 0], weights=joint[:, k, m])
                expected_covariances[k, m, 0, 0] = np.average(
                    (X[:, 0] - expected_means[k, m, 0]) ** 2, weights=joint[:, k, m],
                )
        emission.m_step(X, result.responsibilities)
        assert_allclose(emission.mixture_weights, expected_weights, atol=1e-12)
        assert_allclose(emission.means, expected_means, atol=1e-12)
        assert_allclose(emission.covariances, expected_covariances, atol=1e-12)

    def test_single_component_matches_gaussian_hmm(self):
        for D in (1, 2):
            with self.subTest(n_features=D):
                sequences = [sample_sequence(seed)[:, :D] for seed in (10, 11)]
                gaussian = HMM(
                    GaussianEmission(2, random_state=42),
                    trainer=BaumWelchTrainer(max_iter=8, tol=0),
                ).fit(sequences)
                mixture = HMM(
                    GaussianMixtureEmission(2, 1, random_state=42),
                    trainer=BaumWelchTrainer(max_iter=8, tol=0),
                ).fit(sequences)
                assert_allclose(mixture.history, gaussian.history, atol=1e-10)
                assert_allclose(mixture.states.start, gaussian.states.start, atol=1e-12)
                assert_allclose(mixture.states.transitions, gaussian.states.transitions, atol=1e-12)
                assert_allclose(mixture.emission.means[:, 0], gaussian.emission.means, atol=1e-12)
                assert_allclose(mixture.emission.covariances[:, 0], gaussian.emission.covariances,
                                atol=1e-12)
                assert_array_equal(mixture.emission.mixture_weights, np.ones((2, 1)))
                X = sample_sequence(12)[:, :D]
                assert_allclose(mixture.score(X), gaussian.score(X), atol=1e-10)
                assert_allclose(mixture.predict_proba(X), gaussian.predict_proba(X), atol=1e-12)
                assert_array_equal(mixture.predict(X), gaussian.predict(X))
                for actual, expected in zip(mixture.filter(X), gaussian.filter(X)):
                    assert_allclose(actual, expected, atol=1e-12)

    def test_multicomponent_training_and_inference(self):
        sequences = [sample_sequence(seed) for seed in (20, 21)]
        model = HMM(
            GaussianMixtureEmission(2, 2, random_state=42),
            trainer=BaumWelchTrainer(max_iter=20, tol=0),
        ).fit(sequences)
        self.assertTrue(np.isfinite(model.history).all())
        self.assertTrue((np.diff(model.history) >= -1e-9).all())
        self.assertGreater(model.history[-1], model.history[0] + 1)
        assert_allclose(model.emission.mixture_weights.sum(axis=1), 1)
        assert_allclose(model.states.transitions.sum(axis=1), 1)
        assert_allclose(model.states.start.sum(), 1)
        self.assertTrue((np.linalg.eigvalsh(model.emission.covariances) > 0).all())
        assert_allclose(model.history[-1], sum(model.score(X) for X in sequences))
        X = sample_sequence(22)
        assert_allclose(model.predict_proba(X).sum(axis=1), 1)
        predicted, filtered, log_scores = model.filter(X)
        assert_allclose(predicted.sum(axis=1), 1)
        assert_allclose(filtered.sum(axis=1), 1)
        assert_allclose(log_scores.sum(), model.score(X))
        path = model.predict(X)
        self.assertEqual(path.shape, (len(X),))
        self.assertTrue(np.isin(path, [0, 1]).all())

    def test_degenerate_data_and_reproducible_refit(self):
        for X in (np.ones((1, 2)), np.ones((8, 2)), np.column_stack([np.arange(8.)] * 2)):
            with self.subTest(shape=X.shape, constant=np.ptp(X) == 0):
                model = HMM(
                    GaussianMixtureEmission(2, 3, min_variance=1e-4, random_state=1),
                    trainer=BaumWelchTrainer(max_iter=5, tol=0),
                ).fit(X)
                self.assertTrue(np.isfinite(model.history).all())
                self.assertTrue((np.diff(model.history) >= -1e-7).all())
                eigenvalues = np.linalg.eigvalsh(model.emission.covariances)
                self.assertGreaterEqual(eigenvalues.min(), 1e-4 - 1e-12)
                history = model.history.copy()
                means = model.emission.means.copy()
                model.fit(X)
                assert_allclose(model.history, history)
                assert_array_equal(model.emission.means, means)

    def test_unsupported_state_and_component_preserve_gaussians(self):
        emission = self.configured_emission()
        emission.mixture_weights[0] = [1., 0.]
        means, covariances = emission.means.copy(), emission.covariances.copy()
        state_weights = emission.mixture_weights[1].copy()
        X = np.array([[-1.], [0.], [1.]])
        emission.m_step(X, np.tile([1., 0.], (3, 1)))
        assert_array_equal(emission.means[0, 1], means[0, 1])
        assert_array_equal(emission.covariances[0, 1], covariances[0, 1])
        assert_array_equal(emission.means[1], means[1])
        assert_array_equal(emission.covariances[1], covariances[1])
        assert_array_equal(emission.mixture_weights[1], state_weights)
        assert_array_equal(emission.mixture_weights[0], [1., 0.])
        self.assertTrue(np.isfinite(emission.log_prob(X)).all())

    def test_log_density_is_stable_in_far_tails(self):
        emission = self.configured_emission()
        X = np.array([[-1e4], [1e4]])
        self.assertTrue(np.isfinite(emission.log_prob(X)).all())
        emission.m_step(X, np.full((2, 2), 0.5))
        self.assertTrue(np.isfinite(emission.log_prob(X)).all())
        assert_allclose(emission.mixture_weights.sum(axis=1), 1)

    def test_input_validation(self):
        for value in (0, -1, 1.5, True, np.nan):
            with self.subTest(n_components=value), self.assertRaises(ValueError):
                GaussianMixtureEmission(2, value)
        for value in (0, -1, np.inf, np.nan):
            with self.subTest(min_variance=value), self.assertRaises(ValueError):
                GaussianMixtureEmission(2, min_variance=value)
        emission = GaussianMixtureEmission(2)
        with self.assertRaises(ValueError):
            emission.log_prob([1.])
        with self.assertRaises(ValueError):
            emission.initialize([np.nan])
        emission.initialize([0., 1.])
        with self.assertRaises(ValueError):
            emission.log_prob(np.zeros((2, 3)))
        for weights in (
            np.ones((2, 1)), np.ones((2, 2)),
            np.array([[-1., 2.], [0.5, 0.5]]), np.full((2, 2), np.nan),
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                emission.m_step([0., 1.], weights)


if __name__ == "__main__":
    unittest.main()
