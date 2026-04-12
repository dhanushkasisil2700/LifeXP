"""Tests for 2-D Whittaker-Henderson, P-splines, and Weibull decay fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifexp.graduation.whittaker import whittaker_1d
from lifexp.graduation.whittaker_2d import whittaker_2d
from lifexp.graduation.parametric import fit_weibull_decay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gompertz_surface(n_age: int, n_dur: int, B: float = 0.0001, c: float = 1.08) -> np.ndarray:
    """Smooth Gompertz × hyperbolic-decay surface."""
    ages = np.arange(n_age, dtype=float)
    durs = np.arange(1, n_dur + 1, dtype=float)
    mu_age = B * np.power(c, ages)
    qx_age = 1.0 - np.exp(-mu_age)
    decay = 1.0 / durs
    return np.outer(qx_age, decay)


# ---------------------------------------------------------------------------
# Normal: 2-D smoothing reduces surface noise
# ---------------------------------------------------------------------------

def test_2d_smoothing_reduces_surface_noise():
    """Adding noise then graduating recovers the original surface more closely than the noisy input."""
    rng = np.random.default_rng(42)
    n_age, n_dur = 25, 12
    ages = pd.Index(range(40, 40 + n_age))
    durs = pd.Index(range(1, n_dur + 1))

    true_arr = _gompertz_surface(n_age, n_dur)
    noise = rng.normal(0, 0.00015, size=(n_age, n_dur))
    noisy_arr = np.clip(true_arr + noise, 1e-7, 1.0)

    crude = pd.DataFrame(noisy_arr, index=ages, columns=durs)
    weights = pd.DataFrame(np.full((n_age, n_dur), 10_000.0), index=ages, columns=durs)

    graduated = whittaker_2d(crude, weights, lam_age=200.0, lam_dur=20.0)

    mse_noisy = float(np.mean((noisy_arr - true_arr) ** 2))
    mse_grad = float(np.mean((graduated.to_numpy() - true_arr) ** 2))

    assert mse_grad < mse_noisy, (
        f"Graduation did not reduce MSE: mse_crude={mse_noisy:.2e}, mse_grad={mse_grad:.2e}"
    )
    # Graduated shape must match input
    assert graduated.shape == crude.shape
    assert list(graduated.index) == list(ages)
    assert list(graduated.columns) == list(durs)


# ---------------------------------------------------------------------------
# Normal: independent dimension smoothing (lam_dur = 0 ↔ 1-D W-H per column)
# ---------------------------------------------------------------------------

def test_lam_dur_zero_matches_1d_whittaker_per_column():
    """lam_dur=0 → each duration column is graduated exactly as 1-D W-H."""
    rng = np.random.default_rng(7)
    n_age, n_dur = 20, 8
    ages = pd.Index(range(40, 40 + n_age))
    durs = pd.Index(range(1, n_dur + 1))
    lam = 75.0

    q_arr = rng.uniform(0.001, 0.05, size=(n_age, n_dur))
    w_arr = rng.uniform(2_000, 30_000, size=(n_age, n_dur))

    crude = pd.DataFrame(q_arr, index=ages, columns=durs)
    weights = pd.DataFrame(w_arr, index=ages, columns=durs)

    # 2-D graduation with no duration penalty
    grad_2d = whittaker_2d(crude, weights, lam_age=lam, lam_dur=0.0)

    # Column-by-column 1-D graduation
    for col in durs:
        col_1d = whittaker_1d(crude[col], weights[col], lam=lam)
        np.testing.assert_allclose(
            grad_2d[col].to_numpy(),
            col_1d.to_numpy(),
            rtol=1e-6,
            err_msg=f"Duration column {col}: 2-D (lam_dur=0) differs from 1-D W-H",
        )


# ---------------------------------------------------------------------------
# Normal: Weibull fits decreasing claim termination rates
# ---------------------------------------------------------------------------

def test_weibull_fits_decreasing_termination_rates():
    """fit_weibull_decay recovers shape < 1 and achieves R² > 0.90."""
    k_true, lam_true = 0.55, 6.0
    durs = pd.Series(np.arange(1, 25, dtype=float), name="duration")
    # Exact Weibull hazard values (no noise — pure fit recovery test)
    rates = pd.Series(
        (k_true / lam_true) * (durs / lam_true) ** (k_true - 1.0),
        index=durs.index,
        name="rate",
    )

    result = fit_weibull_decay(durs, rates)

    assert "shape" in result and "scale" in result
    assert "r_squared" in result and "fitted" in result

    # Declining hazard ↔ shape < 1
    assert result["shape"] < 1.0, f"Expected shape < 1, got {result['shape']:.4f}"
    # Close recovery of true parameters
    assert result["shape"] == pytest.approx(k_true, rel=0.05), (
        f"Shape recovery: expected ~{k_true}, got {result['shape']:.4f}"
    )
    assert result["r_squared"] > 0.90, (
        f"R² too low: {result['r_squared']:.4f}"
    )
    assert len(result["fitted"]) == len(durs)


# ---------------------------------------------------------------------------
# Edge: sparse 2-D surface (60 % zero-weight cells) → no NaN in output
# ---------------------------------------------------------------------------

def test_sparse_2d_surface_no_nan():
    """Graduating a surface where 60 % of cells have zero weight produces no NaN."""
    rng = np.random.default_rng(13)
    n_age, n_dur = 20, 10
    ages = pd.Index(range(40, 60))
    durs = pd.Index(range(1, 11))

    q_arr = rng.uniform(0.001, 0.05, size=(n_age, n_dur))
    w_arr = rng.uniform(1_000, 15_000, size=(n_age, n_dur))

    # Zero out 60 % of weights at random
    zero_mask = rng.random(size=(n_age, n_dur)) < 0.60
    w_arr[zero_mask] = 0.0

    crude = pd.DataFrame(q_arr, index=ages, columns=durs)
    weights = pd.DataFrame(w_arr, index=ages, columns=durs)

    graduated = whittaker_2d(crude, weights, lam_age=100.0, lam_dur=20.0)

    assert graduated.shape == crude.shape
    assert not graduated.isna().any().any(), (
        "Graduated surface contains NaN despite valid smoothing penalty"
    )


# ---------------------------------------------------------------------------
# Edge: single duration column → reduces to 1-D W-H over age
# ---------------------------------------------------------------------------

def test_single_duration_column_equals_1d():
    """A surface with exactly 1 duration column graduates identically to 1-D W-H."""
    rng = np.random.default_rng(99)
    n_age = 18
    ages = pd.Index(range(40, 40 + n_age))
    durs = pd.Index([1])
    lam = 60.0

    q_arr = rng.uniform(0.001, 0.05, size=(n_age, 1))
    w_arr = rng.uniform(2_000, 40_000, size=(n_age, 1))

    crude_2d = pd.DataFrame(q_arr, index=ages, columns=durs)
    weights_2d = pd.DataFrame(w_arr, index=ages, columns=durs)

    # 2-D graduation (duration penalty has no effect: n_dur=1 ≤ z=2)
    grad_2d = whittaker_2d(crude_2d, weights_2d, lam_age=lam, lam_dur=50.0)

    # 1-D graduation on the single column
    col_crude = pd.Series(q_arr[:, 0], index=ages)
    col_weights = pd.Series(w_arr[:, 0], index=ages)
    grad_1d = whittaker_1d(col_crude, col_weights, lam=lam)

    np.testing.assert_allclose(
        grad_2d[1].to_numpy(),
        grad_1d.to_numpy(),
        rtol=1e-6,
        err_msg="Single-column 2-D graduation differs from 1-D W-H",
    )
