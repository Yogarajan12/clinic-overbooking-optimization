"""Expected cost of an overbooked session, and the k that minimises it.

Attendance is a sum of independent Bernoulli variables with unequal success
probabilities, so the number of arrivals follows a Poisson-Binomial
distribution rather than a Binomial one. The exact PMF is built by a dynamic
program in O(n^2), which is cheap at session scale (n is about 30), and the
expected cost is the PMF-weighted average of the realised cost. A normal
approximation takes over above 100 booked patients.

Ported without change from stage 3 of the study.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from noshow_overbooking.config import DEFAULT_CONFIG, CostConfig


class CostFunction:
    """Session cost as a function of arrivals, and its expectation over attendance.

    Realised cost for a session with ``shows`` arrivals is

        idle     = max(0, capacity - shows) * service_time * c_idle
        overflow = max(0, shows - capacity) * c_overflow
        waiting  = overflow * (overflow * service_time / 2) * c_wait

    The waiting term is quadratic in overflow: the n-th overflow patient waits
    behind the n-1 before them, so congestion costs grow faster than headcount.
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def calculate_cost(self, shows: int) -> dict[str, float]:
        """Realised cost decomposition for a known number of arrivals."""
        c = self.config

        idle_slots = max(0, c.capacity - shows)
        idle_cost = idle_slots * c.service_time * c.c_idle

        overflow = max(0, shows - c.capacity)
        overflow_cost = overflow * c.c_overflow

        avg_wait_time = overflow * c.service_time / 2
        wait_cost = overflow * avg_wait_time * c.c_wait

        return {
            "idle_cost": idle_cost,
            "overflow_cost": overflow_cost,
            "wait_cost": wait_cost,
            "total_cost": idle_cost + overflow_cost + wait_cost,
            "idle_slots": idle_slots,
            "overflow": overflow,
            "shows": shows,
        }

    def expected_cost(self, show_probs: np.ndarray) -> dict[str, float]:
        """Expected cost over the Poisson-Binomial attendance distribution."""
        show_probs = np.asarray(show_probs, dtype=float)
        n = len(show_probs)
        pmf = self.poisson_binomial_pmf(show_probs)

        expected = {
            "idle_cost": 0.0,
            "overflow_cost": 0.0,
            "wait_cost": 0.0,
            "total_cost": 0.0,
            "expected_shows": 0.0,
            "P_overflow": 0.0,
            "variance_shows": 0.0,
        }

        for s in range(n + 1):
            if pmf[s] <= 1e-10:
                continue
            costs = self.calculate_cost(s)
            expected["idle_cost"] += pmf[s] * costs["idle_cost"]
            expected["overflow_cost"] += pmf[s] * costs["overflow_cost"]
            expected["wait_cost"] += pmf[s] * costs["wait_cost"]
            expected["total_cost"] += pmf[s] * costs["total_cost"]
            expected["expected_shows"] += pmf[s] * s
            if s > self.config.capacity:
                expected["P_overflow"] += pmf[s]

        expected["variance_shows"] = float(
            sum(pmf[s] * (s - expected["expected_shows"]) ** 2 for s in range(n + 1))
        )
        return expected

    @staticmethod
    def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
        """PMF of a sum of independent Bernoulli variables with unequal probabilities."""
        p = np.asarray(p, dtype=float)
        n = len(p)

        if n == 0:
            return np.array([1.0])

        if n <= 100:
            pmf = np.zeros(n + 1)
            pmf[0] = 1.0
            for i, pi in enumerate(p):
                new_pmf = np.zeros(n + 1)
                for k in range(i + 2):
                    if k > 0:
                        new_pmf[k] += pmf[k - 1] * pi
                    new_pmf[k] += pmf[k] * (1 - pi)
                pmf = new_pmf
            return pmf

        mu = float(np.sum(p))
        sigma = float(np.sqrt(np.sum(p * (1 - p))))
        pmf = np.array([stats.norm.pdf(k, mu, sigma) for k in range(n + 1)])
        return pmf / pmf.sum()

    def find_optimal_k(
        self, base_probs: np.ndarray, max_extra: int = 15
    ) -> tuple[int, dict[int, dict[str, float]]]:
        """Exhaustive search for the cost-minimising number of extra bookings.

        ``base_probs`` are no-show probabilities for the patients already
        holding a slot. Each extra booking is modelled as a patient drawn at
        the session's average risk, which is the assumption the simulation
        inherits and one of the study's stated limitations.
        """
        base_probs = np.asarray(base_probs, dtype=float)
        base_show_probs = 1 - base_probs
        results: dict[int, dict[str, float]] = {}

        for k in range(max_extra + 1):
            if k == 0:
                show_probs = base_show_probs
            else:
                avg_noshow = base_probs.mean()
                extra_show_probs = np.array([1 - avg_noshow] * k)
                show_probs = np.concatenate([base_show_probs, extra_show_probs])
            results[k] = self.expected_cost(show_probs)

        optimal_k = min(results, key=lambda k: results[k]["total_cost"])
        return optimal_k, results

    def prove_convexity(self, base_probs: np.ndarray, max_k: int = 15) -> dict[str, object]:
        """Second differences of expected cost in k.

        Non-negative second differences mean the discrete problem is convex, so
        the enumerated minimum is global and the KKT point of the continuous
        relaxation is the optimum rather than a stationary point of unknown type.
        """
        _, results = self.find_optimal_k(base_probs, max_k)
        costs = [results[k]["total_cost"] for k in range(max_k + 1)]
        first_diff = np.diff(costs)
        second_diff = np.diff(first_diff)

        return {
            "costs": costs,
            "first_differences": first_diff.tolist(),
            "second_differences": second_diff.tolist(),
            "is_convex": bool(np.all(second_diff >= -1e-6)),
            "optimal_k": int(np.argmin(costs)),
        }
