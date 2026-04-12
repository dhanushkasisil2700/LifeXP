"""Tests for lifexp.mortality.projection: LeeCarter and apply_improvement_factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifexp.core.tables import MortalityTable, TableRegistry
from lifexp.mortality.projection import (
    InsufficientDataError,
    LeeCarter,
    apply_improvement_factors,
)


# ---------------------------------------------------------------------------
# Shared fixture: synthetic Lee-Carter matrix
#
# True model: ln(q_{x,t}) = a_x + b_x * k_t + ε
#   a_x  = Gompertz log-mortality at each age
#   b_x  = uniform sensitivity (sum = 1)
#   k_t  = linear declining trend → projected mortality falls over time
# ---------------------------------------------------------------------------

def _make_lc_matrix(
    n_ages: int = 30,
    n_years: int = 25,
    noise_std: float = 0.01,
    drift: float = -0.40,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic log-mortality matrix following the LC structure."""
    rng = np.random.default_rng(seed)
    ages = list(range(40, 40 + n_ages))
    years = list(range(1990, 1990 + n_years))

    # Age-specific constants (Gompertz trend)
    a_true = -5.0 + 0.06 * np.arange(n_ages)

    # Sensitivity: mildly increasing with age, normalised to sum = 1
    b_raw = 0.04 + 0.001 * np.arange(n_ages)
    b_true = b_raw / b_raw.sum()

    # Mortality index: linear declining trend (mortality improving)
    k_true = drift * np.arange(n_years)

    # Exact LC matrix + small noise
    ln_qx = a_true[:, np.newaxis] + np.outer(b_true, k_true)
    ln_qx += rng.normal(0, noise_std, (n_ages, n_years))

    return pd.DataFrame(ln_qx, index=ages, columns=years)


@pytest.fixture(scope="module")
def lc_matrix() -> pd.DataFrame:
    return _make_lc_matrix()


@pytest.fixture(scope="module")
def fitted_lc(lc_matrix) -> LeeCarter:
    lc = LeeCarter()
    lc.fit(lc_matrix)
    return lc


# ---------------------------------------------------------------------------
# 1. Normal: SVD reconstruction — ax + bx*kt ≈ original ln(qx) within 5%
# ---------------------------------------------------------------------------

def test_svd_reconstruction(fitted_lc, lc_matrix):
    """Mandatory checkpoint: reconstruction captures ≥ 95% of variance."""
    ax = fitted_lc.ax.to_numpy()
    bx = fitted_lc.bx.to_numpy()
    kt = fitted_lc.kt.to_numpy()

    # Reconstructed log-mortality surface
    reconstructed = ax[:, np.newaxis] + np.outer(bx, kt)

    residuals = reconstructed - lc_matrix.to_numpy()

    # Fraction of total variance unexplained
    centered = lc_matrix.to_numpy() - lc_matrix.to_numpy().mean()
    ss_residual = float(np.sum(residuals ** 2))
    ss_total = float(np.sum(centered ** 2))
    frac_error = ss_residual / ss_total

    assert frac_error < 0.05, (
        f"SVD reconstruction captures less than 95% of variance: "
        f"unexplained fraction = {frac_error:.4f}"
    )


def test_svd_reconstruction_cellwise(fitted_lc, lc_matrix):
    """All per-cell absolute residuals are small (< 0.10 in log scale)."""
    ax = fitted_lc.ax.to_numpy()
    bx = fitted_lc.bx.to_numpy()
    kt = fitted_lc.kt.to_numpy()

    reconstructed = ax[:, np.newaxis] + np.outer(bx, kt)
    max_abs_error = float(np.abs(reconstructed - lc_matrix.to_numpy()).max())

    assert max_abs_error < 0.10, (
        f"Max per-cell abs error in log space = {max_abs_error:.4f} (threshold 0.10)"
    )


def test_bx_sums_to_one(fitted_lc):
    """Normalisation constraint: sum(bx) = 1 after fitting."""
    assert fitted_lc.bx.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Normal: Projected mortality decreasing with declining kt trend
# ---------------------------------------------------------------------------

def test_projected_mortality_decreasing():
    """Project 20 years forward; q_60 in final projected year < base-year q_60."""
    # Use a strong declining drift so the effect is unambiguous
    matrix = _make_lc_matrix(drift=-0.80, noise_std=0.005)
    lc = LeeCarter()
    lc.fit(matrix)

    base_year = int(matrix.columns[-1])    # last fitted year
    proj_df = lc.project(20)
    last_proj_year = int(proj_df.columns[-1])

    # q_x at age 60 from the last fitted year (reconstruct from LC fit)
    ax_60 = float(lc.ax.loc[60])
    bx_60 = float(lc.bx.loc[60])
    kt_base = float(lc.kt.iloc[-1])

    qx_base_60 = float(np.exp(ax_60 + bx_60 * kt_base))
    qx_proj_60 = float(proj_df.loc[60, last_proj_year])

    assert qx_proj_60 < qx_base_60, (
        f"q_60 should decrease under declining kt: "
        f"base={qx_base_60:.6f}, projected={qx_proj_60:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Normal: Life expectancy increases with mortality improvement
# ---------------------------------------------------------------------------

def test_life_expectancy_increases_with_improvement():
    """e_65 in a future projected year exceeds e_65 in the base year."""
    matrix = _make_lc_matrix(drift=-0.50, noise_std=0.005)
    lc = LeeCarter()
    lc.fit(matrix)

    # Base year = last column of the matrix
    base_year = int(matrix.columns[-1])
    future_year = base_year + 20

    e65_base = lc.life_expectancy(from_age=60, projected_year=base_year)
    e65_future = lc.life_expectancy(from_age=60, projected_year=future_year)

    assert e65_future > e65_base, (
        f"Life expectancy at 60 should increase: "
        f"base={e65_base:.4f}, future={e65_future:.4f}"
    )


def test_life_expectancy_positive(fitted_lc, lc_matrix):
    """e_x is strictly positive for any age within the table range."""
    base_year = int(lc_matrix.columns[-1])
    for age in (40, 50, 60):
        ex = fitted_lc.life_expectancy(from_age=age, projected_year=base_year)
        assert ex > 0.0, f"e_{age} should be positive; got {ex}"


# ---------------------------------------------------------------------------
# 4. Edge: Single-year matrix raises InsufficientDataError
# ---------------------------------------------------------------------------

def test_single_year_raises_insufficient_data_error():
    """fit() with only 1 calendar year raises InsufficientDataError."""
    matrix_1yr = pd.DataFrame(
        {2000: [-6.0, -5.0, -4.0, -3.0]},
        index=[40, 50, 60, 70],
    )
    lc = LeeCarter()
    with pytest.raises(InsufficientDataError, match="2 calendar years"):
        lc.fit(matrix_1yr)


def test_two_years_is_sufficient():
    """fit() with exactly 2 years succeeds (minimum valid input)."""
    matrix_2yr = pd.DataFrame(
        {2000: [-6.0, -5.0, -4.0], 2001: [-6.1, -5.1, -4.1]},
        index=[40, 50, 60],
    )
    lc = LeeCarter()
    lc.fit(matrix_2yr)
    assert hasattr(lc, "ax") and hasattr(lc, "bx") and hasattr(lc, "kt")


# ---------------------------------------------------------------------------
# 5. Edge: Zero improvement factors → target table == base table
# ---------------------------------------------------------------------------

def test_zero_improvement_factors_identity():
    """All improvement factors = 0 → target qx equals base qx for every age."""
    reg = TableRegistry()
    base = reg.get("A_1967_70")

    test_ages = list(range(30, 71))
    years = list(range(2020, 2025))  # 5-year projection
    factors = pd.DataFrame(0.0, index=test_ages, columns=years)

    target = apply_improvement_factors(base, factors, base_year=2020, target_year=2025)

    for age in test_ages:
        assert target.qx(age) == pytest.approx(base.qx(age), rel=1e-9), (
            f"Age {age}: qx changed despite zero improvement factors"
        )


def test_positive_improvement_reduces_qx():
    """Positive improvement factors reduce qx relative to the base table."""
    reg = TableRegistry()
    base = reg.get("A_1967_70")

    ages = list(range(40, 71))
    years = [2020, 2021, 2022]
    # Constant 2% improvement per year
    factors = pd.DataFrame(0.02, index=ages, columns=years)

    target = apply_improvement_factors(base, factors, base_year=2020, target_year=2023)

    for age in ages:
        assert target.qx(age) < base.qx(age), (
            f"Age {age}: qx should decrease with positive improvement rate"
        )
        # After 3 years at 2%: target ≈ base × (0.98)^3
        expected = base.qx(age) * (0.98 ** 3)
        assert target.qx(age) == pytest.approx(expected, rel=1e-9)


def test_apply_improvement_target_before_base_raises():
    """target_year ≤ base_year raises ValueError."""
    base = TableRegistry().get("A_1967_70")
    factors = pd.DataFrame(0.01, index=[40], columns=[2020])
    with pytest.raises(ValueError, match="strictly greater"):
        apply_improvement_factors(base, factors, base_year=2025, target_year=2020)
