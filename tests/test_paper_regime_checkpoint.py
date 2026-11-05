"""Check model continuation and crash-safe checkpoint resume semantics."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from flexible_emission_hmm import GaussianEmission, HMM, BaumWelchTrainer
from notebooks.sp500.run_paper_regime import run


class ContinuationTests(unittest.TestCase):
    """Verify that continuation preserves existing trained parameters."""

    def test_continue_does_not_reinitialize_emission(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        model = HMM(GaussianEmission(2, random_state=0),
                    trainer=BaumWelchTrainer(max_iter=2, tol=0)).fit(X)
        before = model.emission.means.copy()
        original_initialize = model.emission.initialize
        def forbidden_initialize(_):
            raise AssertionError("Warm start reinitialized the emission")
        model.emission.initialize = forbidden_initialize
        model.continue_fit(np.vstack((X, [[0.2, -0.1]])), max_iter=1, tol=0)
        self.assertEqual(model.trainer.n_iter, 1)
        self.assertEqual(len(model.history), 2)
        self.assertFalse(np.array_equal(model.emission.means, before))
        model.emission.initialize = original_initialize

    def test_resume_matches_uninterrupted_run(self):
        dates = pd.bdate_range("2000-01-03", periods=550)
        prices = pd.Series(100 * np.cumprod(1 + 0.01 * np.sin(np.arange(550))),
                           index=dates)
        start, middle, end = dates[500], dates[501], dates[503]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            arguments = dict(years=1, seed=0, max_iter=2,
                             update_iter=1, refresh_days=2)
            first = run(prices, output / "resume", start, middle,
                        command="python run_paper_regime.py --end 2001-12-04\n",
                        **arguments)
            self.assertEqual(len(first), 12)
            command_path = output / "resume" / "command.txt"
            self.assertEqual(command_path.read_text(),
                             "python run_paper_regime.py --end 2001-12-04\n")
            resumed = run(prices, output / "resume", start, end,
                          command="python run_paper_regime.py --end 2001-12-06\n",
                          **arguments)
            self.assertEqual(command_path.read_text(),
                             "python run_paper_regime.py --end 2001-12-06\n")
            full = run(prices, output / "full", start, end, **arguments)
            pd.testing.assert_frame_equal(resumed, full)
            with (output / "resume" / "checkpoint.pkl").open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.assertEqual(len(checkpoint["branches"]), 6)
            self.assertEqual(len(checkpoint["rows"]), 24)
            self.assertEqual(len(pd.read_csv(output / "resume" / "signals.csv")), 24)
            with self.assertRaisesRegex(ValueError, "Checkpoint data or settings differ"):
                run(prices, output / "resume", start, end, **{**arguments, "seed": 1})


if __name__ == "__main__":
    unittest.main()
