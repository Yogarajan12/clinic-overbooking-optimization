"""The twelve overbooking policies compared in the study.

A policy is a rule that turns the risk profile of a session into k, the number
of patients booked beyond capacity. The set spans three families:

* blind rules that ignore the model entirely (``no_overbooking``,
  ``fixed_10pct``, ``fixed_20pct``, ``historical_rate``);
* risk-aware rules that use the calibrated per-patient probabilities
  (``predictive_*``, ``cost_optimal``, ``risk_threshold_*``);
* equity-aware rules that damp overbooking where vulnerable patients are
  concentrated (``fairness_constrained``, ``group_adjusted``).

These are the definitions used in the stage 4 simulation, so they are the ones
behind every reported cost. ``exact_cost_optimal_k`` is the separate,
Poisson-Binomial optimum from stage 3; see ``docs/limitations.md`` on why the
two differ and why that matters when reading the headline number.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

POLICY_NAMES: tuple[str, ...] = (
    "no_overbooking",
    "fixed_10pct",
    "fixed_20pct",
    "historical_rate",
    "predictive_simple",
    "predictive_conservative",
    "predictive_aggressive",
    "cost_optimal",
    "risk_threshold_low",
    "risk_threshold_high",
    "fairness_constrained",
    "group_adjusted",
)

HISTORICAL_NOSHOW_RATE = 0.285  # cleaned Kaggle cohort, all periods pooled


def policy_k(
    probs: Sequence[float],
    policy_name: str,
    capacity: int = 20,
    c_idle: float = 150.0,
    c_overflow: float = 75.0,
    age: Sequence[float] | None = None,
    scholarship: Sequence[float] | None = None,
) -> int:
    """Number of extra patients to book into one session under ``policy_name``.

    Parameters
    ----------
    probs:
        Predicted no-show probability for each patient already in the session.
    age, scholarship:
        Per-patient attributes, needed only by the two equity-aware policies.
        When absent those policies fall back to ``predictive_simple``.

    Returns
    -------
    int
        k, clipped to ``[0, capacity]``.
    """
    if policy_name not in POLICY_NAMES:
        raise ValueError(f"Unknown policy: {policy_name}")

    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError("probs must not be empty")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must lie in [0, 1]")

    expected_no_shows = float(probs.sum())
    std_no_shows = float(np.sqrt(np.sum(probs * (1 - probs))))

    if policy_name == "no_overbooking":
        k = 0

    elif policy_name == "fixed_10pct":
        k = round(capacity * 0.10)

    elif policy_name == "fixed_20pct":
        k = round(capacity * 0.20)

    elif policy_name == "historical_rate":
        k = round(capacity * HISTORICAL_NOSHOW_RATE)

    elif policy_name == "predictive_simple":
        # Book exactly the number of no-shows the model expects.
        k = int(round(expected_no_shows))

    elif policy_name == "predictive_conservative":
        # One standard deviation below the expectation: protects against overflow.
        k = int(max(0, round(expected_no_shows - std_no_shows)))

    elif policy_name == "predictive_aggressive":
        # Half a standard deviation above: buys utilisation with overflow risk.
        k = int(round(expected_no_shows + 0.5 * std_no_shows))

    elif policy_name == "cost_optimal":
        # Tilts the expectation by the cost asymmetry: when idle time is dearer
        # than overflow the rule leans long, otherwise it leans short.
        if c_overflow > c_idle:
            k = int(round(expected_no_shows * 0.8))
        else:
            k = int(round(expected_no_shows * 1.2))

    elif policy_name == "risk_threshold_low":
        # Only the clearly high-risk patients (>40%) earn a replacement booking.
        k = int(round(int((probs > 0.40).sum()) * 0.8))

    elif policy_name == "risk_threshold_high":
        # A wider net (>25%) but a smaller replacement fraction.
        k = int(round(int((probs > 0.25).sum()) * 0.5))

    elif policy_name == "fairness_constrained":
        base_k = int(round(expected_no_shows))
        vulnerable = False
        if age is not None:
            vulnerable |= bool(np.mean(np.asarray(age) < 30) > 0.5)
        if scholarship is not None:
            vulnerable |= bool(np.mean(np.asarray(scholarship)) > 0.5)
        k = int(round(base_k * 0.8)) if vulnerable else base_k

    else:  # group_adjusted
        if age is None:
            k = int(round(expected_no_shows))
        else:
            age_arr = np.asarray(age, dtype=float)
            weights = np.ones_like(probs)
            weights[age_arr < 30] = 0.7
            weights[age_arr > 50] = 1.1
            k = int(round(float(np.sum(probs * weights))))

    return int(max(0, min(k, capacity)))


def exact_cost_optimal_k(
    probs: Sequence[float],
    config=None,
    max_extra: int = 15,
) -> int:
    """Poisson-Binomial expected-cost minimiser from stage 3.

    This is the optimisation the theory section is about: enumerate k, take the
    exact expected cost under the Poisson-Binomial attendance distribution, and
    return the argmin. It is *not* the rule labelled ``cost_optimal`` in the
    simulation, which is the cost-ratio heuristic above.
    """
    from noshow_overbooking.cost import CostFunction

    optimal_k, _ = CostFunction(config).find_optimal_k(np.asarray(probs, dtype=float), max_extra)
    return int(optimal_k)
