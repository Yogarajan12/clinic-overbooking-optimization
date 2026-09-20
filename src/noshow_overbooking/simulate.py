"""Monte Carlo evaluation of an overbooking policy on held-out sessions.

The simulator takes held-out appointments with calibrated no-show
probabilities, partitions them into fixed-size sessions, applies a policy to
get k, samples attendance from the predicted probabilities and prices the
outcome. Repeating that many times gives a cost distribution per policy rather
than a point estimate, which is what makes confidence intervals and paired
significance tests possible.

Two modelling choices are worth stating plainly, because they shape every
number that comes out. Extra bookings are drawn from the same session
population, so overbooked patients carry the same risk profile as the patients
already there. And the simulator prices a session with

    cost = max(0, capacity - shows) * c_idle + max(0, shows - capacity) * c_overflow

which omits the service-time scaling and the quadratic waiting term used by
the stage 3 cost model. See ``docs/limitations.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from noshow_overbooking.policies import policy_k

PROB_COLUMN = "pred_noshow_prob"


def create_slots(
    data: pd.DataFrame, slot_size: int = 20, n_slots: int | None = None
) -> list[pd.DataFrame]:
    """Partition appointments into consecutive fixed-size sessions.

    The data is assumed to be in chronological order already, so sessions are
    contiguous blocks rather than random samples.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    if n_slots is None:
        n_slots = len(data) // slot_size
    needed = n_slots * slot_size
    if needed > len(data):
        raise ValueError(f"need {needed} rows for {n_slots} slots, got {len(data)}")

    block = data.head(needed)
    return [
        block.iloc[i * slot_size : (i + 1) * slot_size].reset_index(drop=True)
        for i in range(n_slots)
    ]


def slot_cost(
    shows: int, capacity: int, c_idle: float, c_overflow: float
) -> tuple[float, int, int]:
    """Price one session outcome. Returns (total cost, idle slots, overflow patients)."""
    idle_slots = max(0, capacity - shows)
    overflow_patients = max(0, shows - capacity)
    total = idle_slots * c_idle + overflow_patients * c_overflow
    return float(total), int(idle_slots), int(overflow_patients)


def simulate_slot(
    slot: pd.DataFrame,
    policy_name: str,
    capacity: int = 20,
    c_idle: float = 150.0,
    c_overflow: float = 75.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """One policy, one session, one sampled attendance realisation."""
    rng = rng or np.random.default_rng()
    probs = slot[PROB_COLUMN].to_numpy(dtype=float)

    k = policy_k(
        probs,
        policy_name,
        capacity=capacity,
        c_idle=c_idle,
        c_overflow=c_overflow,
        age=slot["Age"].to_numpy() if "Age" in slot.columns else None,
        scholarship=slot["Scholarship"].to_numpy() if "Scholarship" in slot.columns else None,
    )

    booked = capacity + k
    # Extra bookings inherit the session's own risk profile, by repetition.
    repeats = int(np.ceil(booked / len(probs)))
    booked_probs = np.tile(probs, repeats)[:booked]

    no_shows = rng.random(booked) <= booked_probs
    shows = int(booked - no_shows.sum())

    total_cost, idle_slots, overflow = slot_cost(shows, capacity, c_idle, c_overflow)

    return {
        "policy": policy_name,
        "k": k,
        "booked": booked,
        "shows": shows,
        "idle_slots": idle_slots,
        "overflow": overflow,
        "overflow_event": int(overflow > 0),
        "utilization": min(shows, capacity) / capacity,
        "total_cost": total_cost,
    }


def simulate_policy(
    slots: Sequence[pd.DataFrame],
    policy_name: str,
    n_iterations: int = 1000,
    capacity: int = 20,
    c_idle: float = 150.0,
    c_overflow: float = 75.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run ``n_iterations`` attendance draws over every session.

    Returns one row per iteration with session-level means, which is the unit
    the confidence intervals and paired tests are computed on.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for iteration in range(n_iterations):
        costs, utils, overflows = [], [], []
        for slot in slots:
            outcome = simulate_slot(
                slot, policy_name, capacity, c_idle, c_overflow, rng=rng
            )
            costs.append(outcome["total_cost"])
            utils.append(outcome["utilization"])
            overflows.append(outcome["overflow_event"])
        rows.append(
            {
                "iteration": iteration,
                "policy": policy_name,
                "mean_cost": float(np.mean(costs)),
                "mean_utilization": float(np.mean(utils)),
                "overflow_rate": float(np.mean(overflows)),
            }
        )

    return pd.DataFrame(rows)
