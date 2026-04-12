"""Whittaker-Henderson 1-D graduation."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _diff_matrix(n: int, z: int) -> np.ndarray:
    """Return the z-th order finite-difference matrix of shape (n-z, n).

    Built by applying numpy.diff z times to the n×n identity matrix.
    For z=2 the result is the standard second-difference operator:
    row i = [0, …, 0, 1, -2, 1, 0, …, 0].
    """
    D = np.eye(n)
    for _ in range(z):
        D = np.diff(D, axis=0)
    return D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def whittaker_1d(
    crude_rates: pd.Series,
    weights: pd.Series,
    lam: float,
    z: int = 2,
) -> pd.Series:
    """Graduate crude mortality rates using the Whittaker-Henderson method.

    Solves the linear system::

        (W + λ DᵀD) g = W q

    where *W* is the diagonal weight matrix, *D* is the z-th order difference
    operator, *λ* is the smoothing parameter and *q* are the crude rates.

    Parameters
    ----------
    crude_rates:
        Observed (crude) mortality rates indexed by age.
    weights:
        Exposure weights for each age cell (ETR or death counts).
        A weight of 0 means the cell has no data; its graduated rate is
        determined entirely by interpolation from neighbours.
    lam:
        Smoothing parameter λ ≥ 0.
        λ = 0 → identity (graduated = crude).
        Large λ → heavily smoothed rates approaching a degree-(z-1) polynomial.
    z:
        Order of the difference penalty.  z=2 (default) penalises curvature,
        producing a smooth curve; z=1 penalises slope.

    Returns
    -------
    pd.Series
        Graduated rates with the same index as *crude_rates*.
    """
    n = len(crude_rates)

    # Edge case: single observation — nothing to smooth
    if n <= 1:
        return crude_rates.copy()

    q = crude_rates.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)

    # Degenerate: no lambda and no weights → return crude unchanged
    if lam == 0.0 and np.all(w == 0.0):
        return crude_rates.copy()

    # Build penalty matrix D'D
    D = _diff_matrix(n, z)
    DtD = D.T @ D           # shape (n, n)

    W = np.diag(w)
    A = W + lam * DtD       # symmetric positive (semi-)definite
    b = W @ q

    # Solve with lstsq for robustness (handles near-singular A, e.g. lam=0)
    g, *_ = np.linalg.lstsq(A, b, rcond=None)

    return pd.Series(g, index=crude_rates.index, name=crude_rates.name)
