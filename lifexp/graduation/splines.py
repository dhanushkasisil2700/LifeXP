"""P-spline graduation for 2-D mortality surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from lifexp.graduation.whittaker import _diff_matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bspline_design(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """Build a B-spline design matrix of shape (len(x), n_basis).

    Uses equally-spaced internal knots and clamped boundary knots
    (multiplicity = degree + 1 at each end).

    Parameters
    ----------
    x:
        Evaluation points (need not be integers).
    n_basis:
        Number of B-spline basis functions (= columns in the design matrix).
        Must satisfy ``n_basis >= degree + 1``.
    degree:
        Polynomial degree of the B-splines (default 3 = cubic).

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(len(x), n_basis)``.
    """
    x = np.asarray(x, dtype=float)
    if n_basis < degree + 1:
        raise ValueError(
            f"n_basis={n_basis} is too small for degree={degree}; "
            f"need n_basis >= {degree + 1}"
        )
    n_internal = n_basis - degree - 1
    x_min, x_max = float(x.min()), float(x.max())

    if n_internal > 0:
        internal_knots = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])

    knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        internal_knots,
        np.repeat(x_max, degree + 1),
    ])

    B = np.zeros((len(x), n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        B[:, j] = BSpline(knots, coef, degree)(x)
    return B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pspline_2d(
    crude_surface: pd.DataFrame,
    weights: pd.DataFrame,
    df_age: int = 10,
    df_dur: int = 6,
    lam_age: float = 1.0,
    lam_dur: float = 1.0,
    degree: int = 3,
) -> pd.DataFrame:
    """Graduate a 2-D mortality surface using P-splines.

    Fits the model::

        G ≈ B_age · C · B_durᵀ

    where *B_age* and *B_dur* are B-spline design matrices and *C* is a
    (``df_age`` × ``df_dur``) coefficient matrix estimated by penalised least
    squares with second-order difference penalties on the coefficients in each
    direction.

    The vectorised normal equations are::

        (B₂ᵀ W B₂ + λ_age · P_age + λ_dur · P_dur) α = B₂ᵀ W q

    where ``B₂ = B_age ⊗ B_dur`` and the penalty matrices are Kronecker products
    of difference-squared matrices on the coefficient indices.

    Parameters
    ----------
    crude_surface:
        DataFrame of crude rates; rows = age, columns = duration.
    weights:
        Exposure weights; same shape as *crude_surface*.
    df_age:
        Number of B-spline basis functions in the age direction.
    df_dur:
        Number of B-spline basis functions in the duration direction.
    lam_age:
        Roughness penalty on age-direction spline coefficients.
    lam_dur:
        Roughness penalty on duration-direction spline coefficients.
    degree:
        B-spline polynomial degree (default 3 = cubic).

    Returns
    -------
    pd.DataFrame
        Graduated surface with the same index and columns as *crude_surface*.
    """
    n_age, n_dur = crude_surface.shape

    Q = crude_surface.to_numpy(dtype=float)
    W_arr = weights.to_numpy(dtype=float)

    # Cap df to the number of data points in each dimension
    df_age = min(df_age, n_age)
    df_dur = min(df_dur, n_dur)

    # Ensure df is large enough for the requested degree
    df_age = max(df_age, degree + 1)
    df_dur = max(df_dur, degree + 1)

    # B-spline design matrices (evaluate at integer indices 0 … n-1)
    age_idx = np.arange(n_age, dtype=float)
    dur_idx = np.arange(n_dur, dtype=float)
    B_age = _bspline_design(age_idx, df_age, degree)   # (n_age, df_age)
    B_dur = _bspline_design(dur_idx, df_dur, degree)   # (n_dur, df_dur)

    # 2-D design matrix via Kronecker product (row-major: age outer, dur inner)
    B_2d = np.kron(B_age, B_dur)                       # (n_age*n_dur, df_age*df_dur)

    # Vectorised observations and weights
    q = Q.ravel()
    w = W_arr.ravel()
    W_diag = np.diag(w)

    # Penalty matrices on spline coefficients (second-order differences)
    D_age_c = _diff_matrix(df_age, 2)
    D_dur_c = _diff_matrix(df_dur, 2)
    P_age = np.kron(D_age_c.T @ D_age_c, np.eye(df_dur))
    P_dur = np.kron(np.eye(df_age), D_dur_c.T @ D_dur_c)

    # Normal equations
    BtWB = B_2d.T @ W_diag @ B_2d
    A = BtWB + lam_age * P_age + lam_dur * P_dur
    b = B_2d.T @ (W_diag @ q)

    alpha, *_ = np.linalg.lstsq(A, b, rcond=None)

    g = B_2d @ alpha
    G = g.reshape(n_age, n_dur)
    return pd.DataFrame(G, index=crude_surface.index, columns=crude_surface.columns)
