"""Tests for the session cost model and the Poisson-Binomial optimiser."""

import numpy as np
import pytest
from scipy import stats

from noshow_overbooking.config import CostConfig
from noshow_overbooking.cost import CostFunction


@pytest.fixture
def cost_fn():
    return CostFunction(CostConfig())


def test_exact_capacity_costs_nothing(cost_fn):
    """Twenty arrivals into twenty slots is the one free outcome."""
    assert cost_fn.calculate_cost(20)["total_cost"] == 0.0


def test_idle_cost_scales_with_service_time(cost_fn):
    """Five empty slots at 20 minutes each cost five * 0.333h * $150."""
    result = cost_fn.calculate_cost(15)
    assert result["idle_slots"] == 5
    assert result["overflow"] == 0
    assert result["idle_cost"] == pytest.approx(5 * 0.333 * 150.0)


def test_waiting_cost_is_quadratic_in_overflow(cost_fn):
    """Doubling overflow more than doubles the waiting penalty."""
    one = cost_fn.calculate_cost(21)["wait_cost"]
    two = cost_fn.calculate_cost(22)["wait_cost"]
    assert two == pytest.approx(4 * one)


def test_poisson_binomial_pmf_is_a_distribution(cost_fn):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, size=25)
    pmf = cost_fn.poisson_binomial_pmf(probs)

    assert pmf.shape == (26,)
    assert pmf.sum() == pytest.approx(1.0)
    assert np.all(pmf >= 0)


def test_poisson_binomial_reduces_to_binomial_when_probs_are_equal(cost_fn):
    """With identical probabilities the distribution must be Binomial(n, p)."""
    n, p = 20, 0.3
    pmf = cost_fn.poisson_binomial_pmf(np.full(n, p))
    expected = stats.binom.pmf(np.arange(n + 1), n, p)
    assert np.allclose(pmf, expected, atol=1e-12)


def test_poisson_binomial_mean_matches_sum_of_probabilities(cost_fn):
    probs = np.array([0.1, 0.4, 0.55, 0.8, 0.25])
    pmf = cost_fn.poisson_binomial_pmf(probs)
    mean = float(np.dot(np.arange(len(probs) + 1), pmf))
    assert mean == pytest.approx(probs.sum())


def test_degenerate_probabilities_give_point_masses(cost_fn):
    assert cost_fn.poisson_binomial_pmf(np.zeros(10))[0] == pytest.approx(1.0)
    assert cost_fn.poisson_binomial_pmf(np.ones(10))[10] == pytest.approx(1.0)


def test_expected_cost_is_convex_in_k(cost_fn):
    """Convexity is what makes the enumerated optimum globally optimal."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.2, 0.6, size=20)
    analysis = cost_fn.prove_convexity(probs, max_k=15)

    assert analysis["is_convex"]
    assert 0 <= analysis["optimal_k"] <= 15


def test_optimal_k_grows_when_idle_time_is_priced_higher():
    """The cost ratio, not the risk profile alone, decides how far to push k."""
    probs = np.full(20, 0.35)

    cheap_overflow = CostFunction(CostConfig(c_idle=150.0, c_overflow=75.0))
    dear_overflow = CostFunction(CostConfig(c_idle=75.0, c_overflow=150.0))

    k_aggressive, _ = cheap_overflow.find_optimal_k(probs)
    k_cautious, _ = dear_overflow.find_optimal_k(probs)

    assert k_aggressive > k_cautious


def test_zero_no_show_risk_means_no_overbooking():
    """If everyone attends, any extra booking is pure overflow."""
    cost_fn = CostFunction(CostConfig())
    optimal_k, _ = cost_fn.find_optimal_k(np.zeros(20))
    assert optimal_k == 0


def test_config_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        CostConfig(capacity=0)
    with pytest.raises(ValueError):
        CostConfig(c_overflow=-1.0)
