import csv
import numpy as np


class GeometricBrownianMotionGenerator:
    """
    - emission_params[k] = [mu, sigma] gives regime k's annual drift and
    volatility
    - regime_schedule contains (state, number_of_steps) segments,
    e.g. [(0, 200), (1, 100), (0, 200)]. Labels are parameter-row indices.
    - dt is the amount of years each interval lasts 
    """

    def __init__(self, emission_params, regime_schedule,
                 initial_price=100.0, dt=1 / 252, random_state=None):
        emission_params = np.asarray(emission_params, dtype=float)
        schedule = np.asarray(regime_schedule)
        self.emission_params = emission_params.copy()
        self.regime_schedule = [(int(state), int(steps)) for state, steps in schedule]
        self.data_count = sum(steps for state, steps in self.regime_schedule)
        self.initial_price = float(initial_price)
        self.dt = float(dt)
        self.rng = np.random.default_rng(random_state)

    def generate_data(self):
        states = np.concatenate([
            np.full(steps, state, dtype=int) for state, steps in self.regime_schedule
        ])
        mu, sigma = self.emission_params[states].T
        log_returns = self.rng.normal((mu - 0.5 * sigma**2) * self.dt,
                                     sigma * np.sqrt(self.dt))
        prices = np.exp(np.log(self.initial_price) + log_returns.cumsum())
        simple_returns = np.expm1(log_returns)
        return [
            ((t + 1) * self.dt, int(state), float(log_return), float(price), float(simple_return))
            for t, (state, log_return, price, simple_return)
            in enumerate(zip(states, log_returns, prices, simple_returns))
        ]

    def write_csv(self, filename, rows):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "state", "data", "price", "simple_return"])
            writer.writerows(rows)