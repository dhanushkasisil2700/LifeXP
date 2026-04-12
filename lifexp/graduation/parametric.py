"""Parametric mortality model fitting: Gompertz, Makeham, and Weibull decay."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _qx_from_mu(mu: np.ndarray) -> np.ndarray:
    """Convert force-of-mortality to annual probability: qx = 1 - exp(-mu)."""
    return 1.0 - np.exp(-mu)


def _gompertz_mu(x: np.ndarray, B: float, c: float) -> np.ndarray:
    """Gompertz force of mortality: mu_x = B * c^x."""
    return B * np.power(c, x)


def _makeham_mu(x: np.ndarray, A: float, B: float, c: float) -> np.ndarray:
    """Makeham force of mortality: mu_x = A + B * c^x."""
    return A + B * np.power(c, x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_gompertz(
    ages: pd.Series,
    qx: pd.Series,
) -> dict:
    """Fit a Gompertz mortality model to observed rates.

    Uses the log-linear relationship::

        ln(mu_x) ≈ ln(B) + x * ln(c)

    with an OLS warm-start, then refines via scipy curve_fit on the
    non-linear model mu_x = B * c^x.  Rates are converted to forces via
    mu_x ≈ -ln(1 - qx) before fitting.

    Parameters
    ----------
    ages:
        Integer ages corresponding to each rate (need not start at 0).
    qx:
        Observed or graduated mortality rates (probabilities).

    Returns
    -------
    dict with keys:
        ``B``            Gompertz scale parameter (> 0).
        ``c``            Gompertz shape parameter (> 1 for increasing mortality).
        ``fitted_qx``    pd.Series of fitted rates on the same index as *qx*.
    """
    x = ages.to_numpy(dtype=float)
    q = qx.to_numpy(dtype=float)

    # Force of mortality: avoid log(0) by clipping
    mu = -np.log(np.clip(1.0 - q, 1e-15, 1.0))

    # OLS log-linear warm-start
    log_mu = np.log(np.clip(mu, 1e-15, None))
    coeffs = np.polyfit(x, log_mu, 1)
    B0 = float(np.exp(coeffs[1]))
    c0 = float(np.exp(coeffs[0]))

    # Clip to physically meaningful region
    B0 = max(B0, 1e-10)
    c0 = max(c0, 1.001)

    popt, _ = curve_fit(
        _gompertz_mu,
        x,
        mu,
        p0=[B0, c0],
        bounds=([0.0, 1.0], [np.inf, np.inf]),
        maxfev=10_000,
    )

    B_fit, c_fit = float(popt[0]), float(popt[1])
    mu_fit = _gompertz_mu(x, B_fit, c_fit)
    qx_fit = pd.Series(_qx_from_mu(mu_fit), index=qx.index, name=qx.name)

    return {"B": B_fit, "c": c_fit, "fitted_qx": qx_fit}


def fit_makeham(
    ages: pd.Series,
    qx: pd.Series,
) -> dict:
    """Fit a Makeham mortality model to observed rates.

    Model: mu_x = A + B * c^x, where A captures an age-independent
    accident/background hazard.

    Parameters
    ----------
    ages:
        Integer ages corresponding to each rate.
    qx:
        Observed or graduated mortality rates.

    Returns
    -------
    dict with keys:
        ``A``            Makeham accident hazard (≥ 0).
        ``B``            Gompertz scale parameter (> 0).
        ``c``            Gompertz shape parameter.
        ``fitted_qx``    pd.Series of fitted rates on the same index as *qx*.
    """
    x = ages.to_numpy(dtype=float)
    q = qx.to_numpy(dtype=float)

    # Force of mortality
    mu = -np.log(np.clip(1.0 - q, 1e-15, 1.0))

    # Warm-start: Gompertz fit for B, c; initialise A near a small fraction of mu
    gompertz = fit_gompertz(ages, qx)
    A0 = float(np.min(mu) * 0.1)
    B0 = gompertz["B"]
    c0 = gompertz["c"]

    popt, _ = curve_fit(
        _makeham_mu,
        x,
        mu,
        p0=[A0, B0, c0],
        bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, np.inf]),
        maxfev=10_000,
    )

    A_fit, B_fit, c_fit = float(popt[0]), float(popt[1]), float(popt[2])
    mu_fit = _makeham_mu(x, A_fit, B_fit, c_fit)
    qx_fit = pd.Series(_qx_from_mu(mu_fit), index=qx.index, name=qx.name)

    return {"A": A_fit, "B": B_fit, "c": c_fit, "fitted_qx": qx_fit}


# ---------------------------------------------------------------------------
# Weibull decay model (morbidity termination rates)
# ---------------------------------------------------------------------------

def _weibull_hazard(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Weibull hazard function: h(t) = (shape/scale) · (t/scale)^(shape−1).

    When shape < 1 the hazard is monotonically decreasing (declining claims
    termination rate).  When shape > 1 it increases (recovery accelerates).
    """
    return (shape / scale) * np.power(t / scale, shape - 1.0)


def fit_weibull_decay(
    durations: pd.Series,
    rates: pd.Series,
) -> dict:
    """Fit a Weibull hazard model to claim termination rates by duration.

    Parameterisation::

        h(t) = (k / λ) · (t / λ)^(k−1)

    where *k* is the shape parameter and *λ* is the scale parameter.
    A shape k < 1 corresponds to a declining (excess-mortality-style) termination
    hazard — typical for morbidity claim termination rates.

    A log-log OLS regression on h(t) provides the warm-start, then
    ``scipy.optimize.curve_fit`` refines the fit.

    Parameters
    ----------
    durations:
        Positive claim durations (e.g. months or quarters since claim onset).
        Must be strictly positive (no zeros).
    rates:
        Observed termination rates at each duration.

    Returns
    -------
    dict with keys:
        ``shape``       Fitted shape parameter k.
        ``scale``       Fitted scale parameter λ.
        ``r_squared``   Coefficient of determination on the rate scale.
        ``fitted``      pd.Series of fitted hazard values on the same index.
    """
    t = durations.to_numpy(dtype=float)
    r = rates.to_numpy(dtype=float)

    # Log-log OLS warm-start: log h(t) = log(k/λ^k) + (k-1)·log(t)
    log_t = np.log(np.clip(t, 1e-15, None))
    log_r = np.log(np.clip(r, 1e-15, None))
    coeffs = np.polyfit(log_t, log_r, 1)
    # slope = k - 1  →  k0 = slope + 1
    # intercept = log(k / λ^k)  →  λ^k = k / exp(intercept)  →  λ = (k/exp(intercept))^(1/k)
    k0 = float(coeffs[0]) + 1.0
    k0 = max(k0, 0.01)
    intercept = float(coeffs[1])
    lam0 = (k0 / np.exp(intercept)) ** (1.0 / k0)
    lam0 = max(lam0, 1e-6)

    popt, _ = curve_fit(
        _weibull_hazard,
        t,
        r,
        p0=[k0, lam0],
        bounds=([1e-6, 1e-6], [np.inf, np.inf]),
        maxfev=10_000,
    )

    k_fit, lam_fit = float(popt[0]), float(popt[1])
    r_fit = _weibull_hazard(t, k_fit, lam_fit)

    # R-squared on the original rate scale
    ss_res = float(np.sum((r - r_fit) ** 2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return {
        "shape": k_fit,
        "scale": lam_fit,
        "r_squared": r_squared,
        "fitted": pd.Series(r_fit, index=rates.index, name=rates.name),
    }
