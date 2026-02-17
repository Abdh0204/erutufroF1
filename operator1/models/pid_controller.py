"""PID Controller for adaptive learning rate in the forward pass.

Implements a Proportional-Integral-Derivative controller that dynamically
adjusts model learning rates based on prediction error feedback.  This
replaces fixed learning rates with an adaptive mechanism that:

- **Proportional (P)**: Reacts to the current error magnitude.  Large
  errors -> aggressive correction.
- **Integral (I)**: Accumulates past errors to eliminate persistent bias.
  Handles regime drift where the model consistently under/over-predicts.
- **Derivative (D)**: Reacts to the *rate of change* of error.  Detects
  when errors are accelerating (market shock) vs decelerating (recovery).

The PID output is a **learning rate multiplier** that scales the model
update step in the forward pass.  During calm periods (small, stable
errors), the multiplier stays near 1.0.  During shocks (large, spiking
errors), it increases to force faster adaptation.

Integration point: ``operator1/models/forecasting.py:run_forward_pass``

Top-level entry points:
    ``PIDController`` -- stateful controller for a single variable.
    ``create_pid_bank`` -- creates controllers for all tier variables.
    ``compute_pid_adjustment`` -- convenience wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default PID gains (tuned for financial time-series error correction).
DEFAULT_KP: float = 0.5    # proportional gain
DEFAULT_KI: float = 0.1    # integral gain
DEFAULT_KD: float = 0.2    # derivative gain

# Clamp the output multiplier to prevent runaway corrections.
MIN_MULTIPLIER: float = 0.1
MAX_MULTIPLIER: float = 5.0

# Integral windup limit (prevents unbounded accumulation).
MAX_INTEGRAL: float = 10.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PIDState:
    """Snapshot of a PID controller's internal state."""

    variable: str = ""
    proportional: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    output: float = 1.0           # learning rate multiplier
    error_history: list[float] = field(default_factory=list)
    n_updates: int = 0


@dataclass
class PIDBankResult:
    """Aggregated result from all PID controllers."""

    states: dict[str, PIDState] = field(default_factory=dict)
    mean_multiplier: float = 1.0
    max_multiplier: float = 1.0
    min_multiplier: float = 1.0
    n_variables: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_multiplier": round(self.mean_multiplier, 4),
            "max_multiplier": round(self.max_multiplier, 4),
            "min_multiplier": round(self.min_multiplier, 4),
            "n_variables": self.n_variables,
            "per_variable": {
                var: {
                    "output": round(s.output, 4),
                    "proportional": round(s.proportional, 4),
                    "integral": round(s.integral, 4),
                    "derivative": round(s.derivative, 4),
                    "n_updates": s.n_updates,
                }
                for var, s in self.states.items()
            },
        }


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------


class PIDController:
    """Single-variable PID controller for adaptive error correction.

    Parameters
    ----------
    variable:
        Name of the variable being controlled (for logging).
    kp:
        Proportional gain.
    ki:
        Integral gain.
    kd:
        Derivative gain.
    min_output:
        Minimum output (learning rate multiplier).
    max_output:
        Maximum output (learning rate multiplier).
    max_integral:
        Anti-windup clamp for the integral term.
    setpoint:
        Target error (usually 0.0 -- we want zero prediction error).
    """

    def __init__(
        self,
        variable: str = "",
        kp: float = DEFAULT_KP,
        ki: float = DEFAULT_KI,
        kd: float = DEFAULT_KD,
        min_output: float = MIN_MULTIPLIER,
        max_output: float = MAX_MULTIPLIER,
        max_integral: float = MAX_INTEGRAL,
        setpoint: float = 0.0,
    ) -> None:
        self.variable = variable
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.max_integral = max_integral
        self.setpoint = setpoint

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._error_history: list[float] = []
        self._n_updates: int = 0

    def update(self, error: float) -> float:
        """Feed a new prediction error and get the updated multiplier.

        Parameters
        ----------
        error:
            Signed prediction error (actual - predicted).  Positive
            means the model under-predicted.

        Returns
        -------
        float
            Learning rate multiplier in [min_output, max_output].
            Values > 1.0 mean "learn faster", < 1.0 mean "slow down".
        """
        if np.isnan(error):
            return 1.0

        # Error relative to setpoint
        e = abs(error) - self.setpoint

        # Proportional term
        p_term = self.kp * e

        # Integral term (with anti-windup)
        self._integral += e
        self._integral = np.clip(self._integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self._integral

        # Derivative term
        d_term = self.kd * (e - self._prev_error) if self._n_updates > 0 else 0.0

        # PID output: base multiplier of 1.0 + correction
        raw_output = 1.0 + p_term + i_term + d_term

        # Clamp
        output = float(np.clip(raw_output, self.min_output, self.max_output))

        # Update state
        self._prev_error = e
        self._error_history.append(error)
        self._n_updates += 1

        return output

    def reset(self) -> None:
        """Reset the controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._error_history.clear()
        self._n_updates = 0

    @property
    def state(self) -> PIDState:
        """Return a snapshot of the current state."""
        e = abs(self._prev_error)
        return PIDState(
            variable=self.variable,
            proportional=self.kp * e,
            integral=self.ki * self._integral,
            derivative=self.kd * (e - abs(self._error_history[-2]) if len(self._error_history) >= 2 else 0.0),
            output=self.update(self._prev_error) if self._n_updates > 0 else 1.0,
            error_history=list(self._error_history[-50:]),  # last 50 for memory
            n_updates=self._n_updates,
        )


# ---------------------------------------------------------------------------
# PID Bank (one controller per variable)
# ---------------------------------------------------------------------------


def create_pid_bank(
    variables: list[str],
    *,
    kp: float = DEFAULT_KP,
    ki: float = DEFAULT_KI,
    kd: float = DEFAULT_KD,
    tier_weights: dict[str, float] | None = None,
) -> dict[str, PIDController]:
    """Create a bank of PID controllers, one per variable.

    Parameters
    ----------
    variables:
        List of variable names to control.
    kp, ki, kd:
        Base PID gains.
    tier_weights:
        Optional mapping of tier name -> weight.  Higher-tier
        variables get slightly more aggressive gains.

    Returns
    -------
    dict mapping variable name -> PIDController.
    """
    bank: dict[str, PIDController] = {}

    for var in variables:
        # Scale gains by tier importance (higher tier = more responsive)
        scale = 1.0
        if tier_weights:
            for tier_name, weight in tier_weights.items():
                # Higher weights = more important = more responsive PID
                if weight > 0.25:
                    scale = 1.2
                elif weight > 0.15:
                    scale = 1.0
                else:
                    scale = 0.8

        bank[var] = PIDController(
            variable=var,
            kp=kp * scale,
            ki=ki * scale,
            kd=kd * scale,
        )

    logger.info("Created PID bank with %d controllers", len(bank))
    return bank


def compute_pid_adjustment(
    bank: dict[str, PIDController],
    errors: dict[str, float],
) -> PIDBankResult:
    """Feed prediction errors to the PID bank and get adjustments.

    Parameters
    ----------
    bank:
        PID controller bank from ``create_pid_bank``.
    errors:
        Mapping of variable name -> prediction error for the current
        time step.

    Returns
    -------
    PIDBankResult with per-variable multipliers and summary stats.
    """
    states: dict[str, PIDState] = {}
    multipliers: list[float] = []

    for var, controller in bank.items():
        error = errors.get(var, 0.0)
        multiplier = controller.update(error)
        multipliers.append(multiplier)

        states[var] = PIDState(
            variable=var,
            proportional=controller.kp * abs(error),
            integral=controller.ki * controller._integral,
            derivative=controller.kd * (abs(error) - abs(controller._prev_error)),
            output=multiplier,
            error_history=list(controller._error_history[-20:]),
            n_updates=controller._n_updates,
        )

    if multipliers:
        return PIDBankResult(
            states=states,
            mean_multiplier=float(np.mean(multipliers)),
            max_multiplier=float(np.max(multipliers)),
            min_multiplier=float(np.min(multipliers)),
            n_variables=len(multipliers),
        )

    return PIDBankResult()
