"""Calibrated no-show risk and cost-aware clinic overbooking.

The library holds the decision layer of the study: the cost model, the
Poisson-Binomial expected-cost optimiser, the twelve overbooking policies and
the Monte Carlo simulator. It is deliberately free of I/O so that each piece
can be unit-tested in isolation; the end-to-end study lives in ``pipeline/``.
"""

from noshow_overbooking.config import CostConfig, load_cost_config
from noshow_overbooking.cost import CostFunction
from noshow_overbooking.metrics import bootstrap_ci, paired_bootstrap_test
from noshow_overbooking.policies import POLICY_NAMES, policy_k
from noshow_overbooking.simulate import (
    create_slots,
    simulate_policy,
    simulate_slot,
    slot_cost,
)

__all__ = [
    "CostConfig",
    "load_cost_config",
    "CostFunction",
    "POLICY_NAMES",
    "policy_k",
    "create_slots",
    "simulate_slot",
    "simulate_policy",
    "slot_cost",
    "bootstrap_ci",
    "paired_bootstrap_test",
]

__version__ = "1.0.0"
