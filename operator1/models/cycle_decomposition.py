"""C8 -- Wave / cycle decomposition (Fourier + Wavelet).

Extracts cyclical components from price and fundamental time series
to identify dominant periods (earnings cycles, seasonal patterns).
Improves prediction of periodic events.

Spec reference: The_Apps_core_idea.pdf Section E.2 Category 6.

Uses scipy.fft for Fourier analysis. Optionally uses pywt for
wavelet decomposition when installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


@dataclass
class DominantCycle:
    """A single detected dominant cycle."""
    period_days: float
    amplitude: float
    phase: float
    label: str = ""  # e.g., "quarterly", "annual"


@dataclass
class CycleResult:
    """Result from cycle decomposition."""
    dominant_cycles: list[DominantCycle] = field(default_factory=list)
    # Wavelet energy by scale/period
    wavelet_energy: dict[str, float] = field(default_factory=dict)
    # Trend component (low-frequency)
    trend_strength: float = 0.0
    # Noise ratio (high-frequency energy / total energy)
    noise_ratio: float = 0.0
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "dominant_cycles": [
                {"period_days": c.period_days, "amplitude": c.amplitude,
                 "phase": c.phase, "label": c.label}
                for c in self.dominant_cycles
            ],
            "wavelet_energy": self.wavelet_energy,
            "trend_strength": self.trend_strength,
            "noise_ratio": self.noise_ratio,
        }


def _label_period(period_days: float) -> str:
    """Assign a human-readable label to a cycle period."""
    if 3 <= period_days <= 8:
        return "weekly"
    elif 18 <= period_days <= 25:
        return "monthly"
    elif 55 <= period_days <= 70:
        return "quarterly"
    elif 110 <= period_days <= 140:
        return "semi-annual"
    elif 230 <= period_days <= 270:
        return "annual"
    else:
        return f"~{int(period_days)}d"


def _fourier_decomposition(
    series: np.ndarray,
    min_period: int = 5,
    max_period: int = 300,
    top_n: int = 5,
) -> list[DominantCycle]:
    """Extract dominant cycles using FFT."""
    n = len(series)
    if n < 2 * min_period:
        return []

    # Detrend (remove linear trend)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    detrended = series - (slope * x + intercept)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = detrended * window

    # FFT
    yf = fft(windowed)
    xf = fftfreq(n, d=1.0)  # frequency in cycles/day

    # Only positive frequencies
    pos_mask = xf > 0
    freqs = xf[pos_mask]
    amplitudes = 2.0 / n * np.abs(yf[pos_mask])

    # Convert to periods
    periods = 1.0 / freqs

    # Filter to valid range
    valid = (periods >= min_period) & (periods <= max_period)
    periods = periods[valid]
    amplitudes = amplitudes[valid]

    if len(periods) == 0:
        return []

    # Get phases
    phases_full = np.angle(yf[pos_mask])
    phases = phases_full[valid]

    # Sort by amplitude (descending)
    order = np.argsort(-amplitudes)
    top_indices = order[:top_n]

    cycles = []
    for idx in top_indices:
        if amplitudes[idx] > 0.01 * amplitudes.max():  # significance threshold
            cycles.append(DominantCycle(
                period_days=round(float(periods[idx]), 1),
                amplitude=round(float(amplitudes[idx]), 6),
                phase=round(float(phases[idx]), 4),
                label=_label_period(float(periods[idx])),
            ))

    return cycles


def _wavelet_decomposition(
    series: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 6,
) -> dict[str, float]:
    """Multi-resolution wavelet decomposition. Returns energy per level."""
    try:
        import pywt
    except ImportError:
        logger.info("pywt not installed, skipping wavelet decomposition")
        return {}

    n = len(series)
    if n < 2 ** (max_level + 1):
        max_level = max(1, int(np.log2(n)) - 2)

    # Detrend
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    detrended = series - (slope * x + intercept)

    # Wavelet decomposition
    coeffs = pywt.wavedec(detrended, wavelet, level=max_level)

    # Compute energy per level
    energy: dict[str, float] = {}
    total_energy = 0.0

    for i, c in enumerate(coeffs):
        level_energy = float(np.sum(c ** 2))
        total_energy += level_energy
        if i == 0:
            label = "approximation"
            period_approx = 2 ** max_level
        else:
            level = max_level - i + 1
            label = f"detail_level_{level}"
            period_approx = 2 ** level
        energy[f"{label} (~{period_approx}d)"] = round(level_energy, 4)

    # Normalise
    if total_energy > 0:
        energy = {k: round(v / total_energy, 4) for k, v in energy.items()}

    return energy


def run_cycle_decomposition(
    cache: pd.DataFrame,
    variable: str = "close",
) -> CycleResult:
    """Run Fourier + wavelet cycle decomposition.

    Parameters
    ----------
    cache:
        Daily feature table.
    variable:
        Column to decompose (default: close price).

    Returns
    -------
    CycleResult
    """
    try:
        return _run_cycle_impl(cache, variable)
    except Exception as exc:
        logger.warning("Cycle decomposition failed: %s", exc)
        return CycleResult(available=False, error=str(exc))


def _run_cycle_impl(
    cache: pd.DataFrame,
    variable: str,
) -> CycleResult:
    if variable not in cache.columns:
        return CycleResult(available=False, error=f"Variable '{variable}' not in cache")

    series = cache[variable].dropna().values
    if len(series) < 30:
        return CycleResult(available=False, error="Insufficient data for cycle analysis")

    # Fourier
    dominant_cycles = _fourier_decomposition(series)

    # Wavelet
    wavelet_energy = _wavelet_decomposition(series)

    # Trend strength: ratio of variance explained by linear trend
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    trend = slope * x + intercept
    residual = series - trend

    total_var = np.var(series)
    trend_var = np.var(trend)
    trend_strength = float(trend_var / total_var) if total_var > 0 else 0.0

    # Noise ratio: high-frequency energy (period < 5 days)
    n = len(series)
    yf = fft(series - trend)
    xf = fftfreq(n, d=1.0)
    power = np.abs(yf) ** 2
    total_power = power[1:n // 2].sum()
    hf_mask = (np.abs(xf) > 1 / 5) & (np.arange(n) < n // 2)
    hf_power = power[hf_mask].sum()
    noise_ratio = float(hf_power / total_power) if total_power > 0 else 0.0

    logger.info(
        "Cycle decomposition: %d dominant cycles, trend=%.2f, noise=%.2f",
        len(dominant_cycles), trend_strength, noise_ratio,
    )

    return CycleResult(
        dominant_cycles=dominant_cycles,
        wavelet_energy=wavelet_energy,
        trend_strength=round(trend_strength, 4),
        noise_ratio=round(noise_ratio, 4),
    )
