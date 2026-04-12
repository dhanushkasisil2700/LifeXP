"""Whittaker-Henderson 2-D graduation over an age × duration surface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lifexp.graduation.whittaker import _diff_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def whittaker_2d(
    crude_surface: pd.DataFrame,
    weights: pd.DataFrame,
    lam_age: float,
    lam_dur: float,
    z: int = 2,
) -> pd.DataFrame:
    """Graduate a 2-D mortality surface using the Whittaker-Henderson method.

    The surface is treated as a vectorised 1-D problem via Kronecker products.
    With row-major (age-outer, duration-inner) flattening the penalty matrices
    are:

    * Age penalty:      λ_age · (Dₐ ⊗ I_dur)ᵀ (Dₐ ⊗ I_dur)
    * Duration penalty: λ_dur · (I_age ⊗ D_d)ᵀ (I_age ⊗ D_d)

    This gives the system::

        (W + λ_age · DₐᵀDₐ ⊗ I_d + λ_dur · I_a ⊗ DdᵀDd) g = W q

    **Separability property**: when ``lam_dur = 0`` the system decouples into
    ``n_dur`` independent 1-D Whittaker problems — one per duration column —
    each identical to calling :func:`~lifexp.graduation.whittaker.whittaker_1d`
    on that column with ``lam = lam_age``.

    Parameters
    ----------
    crude_surface:
        DataFrame of crude rates; rows = age, columns = duration.
    weights:
        DataFrame of ETR or death-count weights; same shape as *crude_surface*.
    lam_age:
        Smoothing parameter in the age direction (λ ≥ 0).
    lam_dur:
        Smoothing parameter in the duration direction (λ ≥ 0).
    z:
        Order of the difference penalty (default 2 = curvature penalty).

    Returns
    -------
    pd.DataFrame
        Graduated surface with the same index and columns as *crude_surface*.
    """
    n_age, n_dur = crude_surface.shape

    # Trivial cases
    if n_age == 0 or n_dur == 0:
        return crude_surface.copy()
    if n_age == 1 and n_dur == 1:
        return crude_surface.copy()

    Q = crude_surface.to_numpy(dtype=float)
    W_arr = weights.to_numpy(dtype=float)

    # Row-major vectorisation: index k = i * n_dur + j  (age outer, dur inner)
    q = Q.ravel()
    w = W_arr.ravel()

    W_diag = np.diag(w)
    A = W_diag.copy()

    # Age-direction penalty (applies D_age between age rows, independently per duration)
    if lam_age > 0.0 and n_age > z:
        D_age = _diff_matrix(n_age, z)       # (n_age-z, n_age)
        I_dur = np.eye(n_dur)
        D_age_2d = np.kron(D_age, I_dur)     # (n_age-z)*n_dur × n_age*n_dur
        A = A + lam_age * (D_age_2d.T @ D_age_2d)

    # Duration-direction penalty (applies D_dur within each age block)
    if lam_dur > 0.0 and n_dur > z:
        I_age = np.eye(n_age)
        D_dur = _diff_matrix(n_dur, z)       # (n_dur-z, n_dur)
        D_dur_2d = np.kron(I_age, D_dur)     # n_age*(n_dur-z) × n_age*n_dur
        A = A + lam_dur * (D_dur_2d.T @ D_dur_2d)

    b = W_diag @ q
    g, *_ = np.linalg.lstsq(A, b, rcond=None)

    G = g.reshape(n_age, n_dur)
    return pd.DataFrame(G, index=crude_surface.index, columns=crude_surface.columns)
