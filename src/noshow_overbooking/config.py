"""Cost parameters for the overbooking decision.

Every headline number in this study is a function of these five values, so
they live in one place and are loaded from ``config/costs.yaml`` rather than
being redefined per stage. Changing ``c_idle`` and ``c_overflow`` changes
which policy wins; that is the point of the sensitivity analysis, not a bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "costs.yaml"


@dataclass
class CostConfig:
    """Clinic cost structure for a single appointment session.

    Attributes
    ----------
    c_idle:
        Cost of an hour of idle clinician time, in dollars.
    c_overflow:
        Cost of one patient who arrives with no slot left, in dollars. Covers
        rescheduling effort, dissatisfaction and delayed care.
    c_wait:
        Cost of an hour of patient waiting, in dollars.
    service_time:
        Hours of clinician time per patient. 0.333 is a twenty-minute slot.
    capacity:
        Patients the session can serve without overflow.
    """

    c_idle: float = 150.0
    c_overflow: float = 75.0
    c_wait: float = 30.0
    service_time: float = 0.333
    capacity: int = 20

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.service_time <= 0:
            raise ValueError("service_time must be positive")
        for field in ("c_idle", "c_overflow", "c_wait"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")

    @property
    def cost_ratio(self) -> float:
        """Idle cost over overflow cost, the quantity that decides how far to push k."""
        return self.c_idle / self.c_overflow if self.c_overflow else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = CostConfig()


def load_cost_config(path: Path | str | None = None, scenario: str = "default") -> CostConfig:
    """Load a named scenario from ``config/costs.yaml``.

    Falls back to the built-in defaults when the file or the scenario is
    missing, so the library stays usable without the repository around it.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return CostConfig()

    import yaml  # imported lazily: the library works without a config file

    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}

    scenarios = payload.get("scenarios", {})
    if scenario not in scenarios:
        raise KeyError(
            f"scenario {scenario!r} not in {path}; available: {sorted(scenarios)}"
        )
    return CostConfig(**scenarios[scenario])
