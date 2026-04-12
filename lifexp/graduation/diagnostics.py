"""Graduation diagnostics: chi-squared test, signs test, summary report."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chi_squared_test(
    crude: pd.Series,
    graduated: pd.Series,
    etr: pd.Series,
) -> dict:
    """Chi-squared goodness-of-fit test for a graduated mortality table.

    Computes the test statistic::

        X² = Σ  (observed_deaths_i − expected_deaths_i)²
                 ────────────────────────────────────────
                         expected_deaths_i

    where observed_deaths = crude_rate × etr and expected_deaths =
    graduated_rate × etr.  Cells with expected_deaths = 0 are excluded.

    Parameters
    ----------
    crude:
        Crude (observed) mortality rates.
    graduated:
        Graduated (smoothed) mortality rates.
    etr:
        Central exposed-to-risk for each age cell.

    Returns
    -------
    dict with keys:
        ``statistic``   Chi-squared test statistic X².
        ``p_value``     P-value (upper tail of χ² distribution).
        ``df``          Degrees of freedom = number of cells included.
        ``pass``        True when p_value > 0.05 (graduation is adequate).
    """
    obs = crude.to_numpy(dtype=float) * etr.to_numpy(dtype=float)
    exp = graduated.to_numpy(dtype=float) * etr.to_numpy(dtype=float)

    # Exclude cells with zero expected deaths (no information)
    mask = exp > 0.0
    obs = obs[mask]
    exp = exp[mask]

    df = int(mask.sum())
    if df == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "df": 0, "pass": False}

    statistic = float(np.sum((obs - exp) ** 2 / exp))
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=df))

    return {
        "statistic": statistic,
        "p_value": p_value,
        "df": df,
        "pass": bool(p_value > 0.05),
    }


def signs_test(
    crude: pd.Series,
    graduated: pd.Series,
) -> dict:
    """Signs test for systematic bias in a graduated table.

    Counts the number of age cells where the crude rate *exceeds* the
    graduated rate (positive deviations, P) versus those where it is
    *below* (negative deviations, N).  Under a good graduation,
    P ≈ N ≈ n/2 by symmetry.

    The test statistic is the normal approximation::

        z = (P − n/2) / sqrt(n/4)

    where n = P + N (ties excluded).

    Parameters
    ----------
    crude:
        Crude mortality rates.
    graduated:
        Graduated mortality rates.

    Returns
    -------
    dict with keys:
        ``n_positive``      P: count of crude > graduated.
        ``n_negative``      N: count of crude < graduated.
        ``n_total``         n = P + N (ties excluded).
        ``z_statistic``     Normal approximation z-score.
        ``p_value``         Two-tailed p-value.
        ``pass``            True when p_value > 0.05.
    """
    diff = crude.to_numpy(dtype=float) - graduated.to_numpy(dtype=float)

    n_pos = int(np.sum(diff > 0.0))
    n_neg = int(np.sum(diff < 0.0))
    n = n_pos + n_neg  # ties excluded

    if n == 0:
        return {
            "n_positive": 0,
            "n_negative": 0,
            "n_total": 0,
            "z_statistic": float("nan"),
            "p_value": float("nan"),
            "pass": True,  # vacuously: nothing to test
        }

    z = (n_pos - n / 2.0) / math.sqrt(n / 4.0)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_total": n,
        "z_statistic": z,
        "p_value": p_value,
        "pass": bool(p_value > 0.05),
    }


def graduation_report(
    crude: pd.Series,
    graduated: pd.Series,
    etr: pd.Series,
) -> dict:
    """Produce a combined graduation diagnostic report.

    Parameters
    ----------
    crude:
        Crude (observed) mortality rates.
    graduated:
        Graduated (smoothed) mortality rates.
    etr:
        Central exposed-to-risk for each age cell.

    Returns
    -------
    dict with keys:
        ``chi_squared``   Result dict from :func:`chi_squared_test`.
        ``signs``         Result dict from :func:`signs_test`.
        ``sum_sq_reduction``
            Fractional reduction in sum-of-squared deviations from the crude
            rates relative to a flat (mean) predictor.  Positive values
            indicate the graduation improved fit; negative means worse.
    """
    chi2 = chi_squared_test(crude, graduated, etr)
    signs = signs_test(crude, graduated)

    # Sum-of-squared deviations: how much smoother is graduated vs crude?
    crude_arr = crude.to_numpy(dtype=float)
    grad_arr = graduated.to_numpy(dtype=float)
    ss_crude = float(np.sum(np.diff(crude_arr) ** 2))
    ss_grad = float(np.sum(np.diff(grad_arr) ** 2))

    if ss_crude > 0.0:
        ssr = (ss_crude - ss_grad) / ss_crude
    else:
        ssr = float("nan")

    return {
        "chi_squared": chi2,
        "signs": signs,
        "sum_sq_reduction": ssr,
    }
