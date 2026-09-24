"""Compare our Gaussian HMM with hmmlearn on reproducible synthetic data.

Run from the repository root:
    uv sync --dev
    uv run python -m unittest discover -s tests -v

Both models use identical initial parameters and maximum-likelihood updates.
Compare after a fixed number of EM updates because the implementations check
convergence at different points. Nondegenerate data keeps our covariance floor
inactive; hmmlearn's min_covar is not an equivalent per-update eigenvalue floor.
"""

import unittest

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.testing import assert_allclose, assert_array_equal

from flexible_emission_hmm import BaumWelchTrainer, GaussianEmission, HMM


SEED = 42
N_STATES = 3
MIN_VARIANCE = 1e-6
RTOL = 1e-7
ATOL = 1e-8


def sample_sequence(seed, n_samples=300, n_features=1):
    """Sample recurring regimes, including correlated multivariate emissions."""
    rng = np.random.default_rng(seed)
    transitions = np.array([
        [0.90, 0.07, 0.03],
        [0.05, 0.90, 0.05],
        [0.03, 0.07, 0.90],
    ])
    means = np.array([[-3.0, 1.0], [0.0, -2.0], [3.0, 2.0]])[:, :n_features]
    covariance = np.array([[1.0, 0.35], [0.35, 0.8]])[:n_features, :n_features]
    X = np.empty((n_samples, n_features))
    state = rng.integers(N_STATES)
    for t in range(n_samples):
        X[t] = rng.multivariate_normal(means[state], covariance)
        state = rng.choice(N_STATES, p=transitions[state])
    return X


class GaussianHMMComparisonTests(unittest.TestCase):
    def assert_models_match(self, sequences, X_test, n_iter):
        X_train = np.concatenate(sequences)
        lengths = [len(sequence) for sequence in sequences]
        model = HMM(
            GaussianEmission(N_STATES, min_variance=MIN_VARIANCE, random_state=SEED),
            trainer=BaumWelchTrainer(max_iter=n_iter, tol=0.0),
        )

        # The same seed alone does not produce the same initialization across
        # libraries. Copy our initial parameters and disable hmmlearn's init.
        # Our fit() repeats this initialization deterministically with SEED.
        model.emission.initialize(X_train)
        reference = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            algorithm="viterbi",
            implementation="log",
            random_state=SEED,
            n_iter=n_iter,
            tol=0.0,
            init_params="",
            params="stmc",
            min_covar=MIN_VARIANCE,
            startprob_prior=1.0,
            transmat_prior=1.0,
            means_prior=0.0,
            means_weight=0.0,
            covars_prior=0.0,
            covars_weight=0.0,
        )
        reference.startprob_ = model.states.start.copy()
        reference.transmat_ = model.states.transitions.copy()
        reference.means_ = model.emission.means.copy()
        reference.covars_ = model.emission.covariances.copy()

        model.fit(sequences[0] if len(sequences) == 1 else sequences)
        reference.fit(X_train, lengths=lengths)
        self.assertEqual(model.trainer.n_iter, n_iter)
        self.assertEqual(reference.monitor_.iter, n_iter)

        for name, actual, expected in (
            ("start probabilities", model.states.start, reference.startprob_),
            ("transition matrix", model.states.transitions, reference.transmat_),
            ("means", model.emission.means, reference.means_),
            ("covariances", model.emission.covariances, reference.covars_),
            # Our history includes the final post-update likelihood; hmmlearn
            # records the likelihood before each update.
            ("likelihood history", model.history[:-1], list(reference.monitor_.history)),
        ):
            with self.subTest(output=name):
                assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

        self.assertGreater(np.linalg.eigvalsh(model.emission.covariances).min(), MIN_VARIANCE)
        for i, X in enumerate([*sequences, X_test]):
            with self.subTest(sequence=i, held_out=i == len(sequences)):
                assert_allclose(model.score(X), reference.score(X), rtol=RTOL, atol=ATOL)
                assert_allclose(
                    model.predict_proba(X), reference.predict_proba(X), rtol=RTOL, atol=ATOL,
                )
                # Shared initialization preserves state labels, so no label
                # permutation is needed for this controlled comparison.
                assert_array_equal(model.predict(X), reference.predict(X))

    def test_one_em_update_matches_hmmlearn(self):
        self.assert_models_match([sample_sequence(SEED)], sample_sequence(SEED + 1), n_iter=1)

    def test_univariate_fit_matches_hmmlearn(self):
        self.assert_models_match([sample_sequence(SEED)], sample_sequence(SEED + 1), n_iter=20)

    def test_full_covariance_fit_matches_hmmlearn(self):
        self.assert_models_match(
            [sample_sequence(SEED, n_features=2)],
            sample_sequence(SEED + 1, n_features=2),
            n_iter=20,
        )

    def test_multiple_sequences_fit_matches_hmmlearn(self):
        self.assert_models_match(
            [sample_sequence(SEED, n_samples=180), sample_sequence(SEED + 1, n_samples=220)],
            sample_sequence(SEED + 2),
            n_iter=20,
        )


if __name__ == "__main__":
    unittest.main()
