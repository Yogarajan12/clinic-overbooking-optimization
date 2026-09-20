"""Uncertainty quantification for simulated policy costs.

Simulation output is a distribution, so every comparison between policies has
to carry an interval. Two tools are enough here: a percentile bootstrap for
the mean cost of a single policy, and a paired bootstrap for the difference
between two policies evaluated on the same sessions. Pairing matters, because
session composition is a shared source of variance and differencing removes it.

The p-value below is a bootstrap tail proportion, not an analytic test
statistic. Its resolution is bounded by ``n_bootstrap``: with 10,000
replicates, "p = 0.0000" means only that no replicate crossed zero, so it
should be read as p < 1e-4 rather than as a vanishing probability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    data: Sequence[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 33,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a mean."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    means = np.array(
        [np.mean(rng.choice(data, size=data.size, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": float(np.mean(data)),
        "ci_lower": float(np.percentile(means, alpha * 100)),
        "ci_upper": float(np.percentile(means, (1.0 - alpha) * 100)),
    }


def paired_bootstrap_test(
    data1: Sequence[float],
    data2: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for the difference in means of two policies.

    Both inputs must be aligned session by session; resampling draws session
    indices, not values, so the pairing is preserved in every replicate.
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    if data1.shape != data2.shape:
        raise ValueError("data must be paired (same length)")

    rng = np.random.default_rng(seed)
    n = data1.size
    observed_diff = float(np.mean(data1) - np.mean(data2))

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = np.mean(data1[idx]) - np.mean(data2[idx])

    if observed_diff > 0:
        p_value = 2 * float(np.mean(boot_diffs <= 0))
    else:
        p_value = 2 * float(np.mean(boot_diffs >= 0))
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": observed_diff,
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "p_value": p_value,
        "p_value_floor": 1.0 / n_bootstrap,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


def disparity(rates_by_group: dict[str, float]) -> dict[str, float]:
    """Absolute and ratio disparity across group-level rates.

    Used for overflow rates by age band and by scholarship status. The ratio is
    the more honest summary when base rates are small: a 2 percentage point gap
    on a 3 percent base is a 1.67x burden, not a rounding error.
    """
    if len(rates_by_group) < 2:
        raise ValueError("need at least two groups")
    values = np.asarray(list(rates_by_group.values()), dtype=float)
    lo, hi = float(values.min()), float(values.max())
    return {
        "absolute": hi - lo,
        "ratio": hi / lo if lo > 0 else float("inf"),
        "max_group": max(rates_by_group, key=rates_by_group.get),
        "min_group": min(rates_by_group, key=rates_by_group.get),
    }
