"""Tests for the overbooking policies and the Monte Carlo simulator."""

import numpy as np
import pandas as pd
import pytest

from noshow_overbooking.metrics import bootstrap_ci, disparity, paired_bootstrap_test
from noshow_overbooking.policies import POLICY_NAMES, policy_k
from noshow_overbooking.simulate import create_slots, simulate_policy, simulate_slot, slot_cost


@pytest.fixture
def session():
    """Twenty patients at the test-period risk level, with demographics attached."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "pred_noshow_prob": rng.uniform(0.25, 0.60, size=20),
            "Age": rng.integers(18, 80, size=20),
            "Scholarship": rng.integers(0, 2, size=20),
        }
    )


def test_baseline_policy_never_overbooks(session):
    assert policy_k(session["pred_noshow_prob"], "no_overbooking") == 0


def test_fixed_rules_ignore_the_model(session):
    """Blind rules must return the same k whatever the risk profile is."""
    low = np.full(20, 0.05)
    high = np.full(20, 0.90)
    for name, expected in (("fixed_10pct", 2), ("fixed_20pct", 4), ("historical_rate", 6)):
        assert policy_k(low, name) == expected
        assert policy_k(high, name) == expected


def test_predictive_simple_books_the_expected_no_shows(session):
    probs = session["pred_noshow_prob"].to_numpy()
    assert policy_k(probs, "predictive_simple") == int(round(probs.sum()))


def test_conservative_never_exceeds_aggressive(session):
    probs = session["pred_noshow_prob"]
    assert (
        policy_k(probs, "predictive_conservative")
        <= policy_k(probs, "predictive_simple")
        <= policy_k(probs, "predictive_aggressive")
    )


def test_cost_optimal_responds_to_the_cost_ratio(session):
    """The rule leans long when idle time is dearer, short when overflow is."""
    probs = session["pred_noshow_prob"]
    idle_dear = policy_k(probs, "cost_optimal", c_idle=150.0, c_overflow=75.0)
    overflow_dear = policy_k(probs, "cost_optimal", c_idle=75.0, c_overflow=150.0)
    assert idle_dear > overflow_dear


def test_fairness_constrained_damps_overbooking_for_vulnerable_sessions():
    probs = np.full(20, 0.45)
    young = np.full(20, 22)
    older = np.full(20, 62)

    base = policy_k(probs, "predictive_simple")
    damped = policy_k(probs, "fairness_constrained", age=young, scholarship=np.ones(20))
    untouched = policy_k(probs, "fairness_constrained", age=older, scholarship=np.zeros(20))

    assert damped < base
    assert untouched == base


def test_group_adjusted_downweights_young_patients():
    probs = np.full(20, 0.45)
    young = policy_k(probs, "group_adjusted", age=np.full(20, 22))
    old = policy_k(probs, "group_adjusted", age=np.full(20, 62))
    assert young < old


def test_every_policy_returns_k_within_bounds(session):
    for name in POLICY_NAMES:
        k = policy_k(
            session["pred_noshow_prob"],
            name,
            age=session["Age"],
            scholarship=session["Scholarship"],
        )
        assert isinstance(k, int)
        assert 0 <= k <= 20


def test_unknown_policy_and_bad_probabilities_are_rejected(session):
    with pytest.raises(ValueError):
        policy_k(session["pred_noshow_prob"], "wishful_thinking")
    with pytest.raises(ValueError):
        policy_k(np.array([1.4, 0.2]), "predictive_simple")


def test_slot_cost_prices_both_failure_modes():
    assert slot_cost(20, 20, 150.0, 75.0) == (0.0, 0, 0)
    assert slot_cost(15, 20, 150.0, 75.0) == (750.0, 5, 0)
    assert slot_cost(23, 20, 150.0, 75.0) == (225.0, 0, 3)


def test_create_slots_partitions_without_overlap():
    frame = pd.DataFrame({"pred_noshow_prob": np.linspace(0.1, 0.9, 100)})
    slots = create_slots(frame, slot_size=20, n_slots=5)
    assert len(slots) == 5
    assert all(len(s) == 20 for s in slots)
    rebuilt = pd.concat(slots)["pred_noshow_prob"].to_numpy()
    assert np.allclose(rebuilt, frame["pred_noshow_prob"].to_numpy())


def test_create_slots_refuses_to_invent_rows():
    frame = pd.DataFrame({"pred_noshow_prob": np.full(10, 0.3)})
    with pytest.raises(ValueError):
        create_slots(frame, slot_size=20, n_slots=5)


def test_certain_attendance_fills_capacity_exactly(session):
    """With zero no-show risk the baseline policy leaves no idle slot."""
    certain = session.assign(pred_noshow_prob=0.0)
    outcome = simulate_slot(certain, "no_overbooking", rng=np.random.default_rng(1))
    assert outcome["shows"] == 20
    assert outcome["total_cost"] == 0.0
    assert outcome["utilization"] == 1.0


def test_certain_absence_wastes_the_whole_session(session):
    empty = session.assign(pred_noshow_prob=1.0)
    outcome = simulate_slot(empty, "no_overbooking", rng=np.random.default_rng(1))
    assert outcome["shows"] == 0
    assert outcome["idle_slots"] == 20
    assert outcome["total_cost"] == pytest.approx(20 * 150.0)


def test_simulated_attendance_matches_the_predicted_rate(session):
    """Law of large numbers: the sampler must reproduce the probabilities it is given."""
    flat = session.assign(pred_noshow_prob=0.40)
    rng = np.random.default_rng(11)
    shows = [
        simulate_slot(flat, "no_overbooking", rng=rng)["shows"] for _ in range(2000)
    ]
    assert np.mean(shows) / 20 == pytest.approx(0.60, abs=0.02)


def test_overbooking_beats_the_baseline_when_idle_time_is_dear(session):
    """The study's central claim, reproduced on a small synthetic run."""
    baseline = simulate_policy([session] * 10, "no_overbooking", n_iterations=60, seed=3)
    predictive = simulate_policy([session] * 10, "predictive_simple", n_iterations=60, seed=3)
    assert predictive["mean_cost"].mean() < baseline["mean_cost"].mean()
    assert predictive["mean_utilization"].mean() > baseline["mean_utilization"].mean()


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(5)
    data = rng.normal(500, 50, size=400)
    result = bootstrap_ci(data, n_bootstrap=400)
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]


def test_paired_bootstrap_detects_a_real_difference_and_ignores_none():
    rng = np.random.default_rng(6)
    a = rng.normal(500, 30, size=300)
    real = a - 40
    noise = a + rng.normal(0, 0.5, size=300)

    assert paired_bootstrap_test(a, real, n_bootstrap=500)["significant_01"]
    assert not paired_bootstrap_test(a, noise, n_bootstrap=500)["significant_05"]


def test_paired_bootstrap_reports_its_resolution_floor():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, size=50)
    result = paired_bootstrap_test(a, a + 5, n_bootstrap=1000)
    assert result["p_value_floor"] == pytest.approx(1e-3)


def test_disparity_reports_absolute_and_ratio_burden():
    result = disparity({"young": 0.031, "middle": 0.035, "older": 0.038})
    assert result["absolute"] == pytest.approx(0.007)
    assert result["ratio"] == pytest.approx(0.038 / 0.031)
    assert result["max_group"] == "older"
