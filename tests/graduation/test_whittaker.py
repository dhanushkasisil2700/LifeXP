"""Tests for lifexp.graduation: Whittaker-Henderson, parametric models, diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lifexp.graduation.whittaker import whittaker_1d
from lifexp.graduation.parametric import fit_gompertz, fit_makeham
from lifexp.graduation.diagnostics import (
    chi_squared_test,
    graduation_report,
    signs_test,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ages_40_70():
    return pd.Series(range(40, 71), name="age")


@pytest.fixture
def gompertz_qx(ages_40_70):
    """Synthetic Gompertz rates: B=0.0001, c=1.10."""
    B, c = 0.0001, 1.10
    x = ages_40_70.to_numpy(dtype=float)
    mu = B * np.power(c, x)
    return pd.Series(1.0 - np.exp(-mu), index=ages_40_70.index, name="qx")


@pytest.fixture
def uniform_weights(ages_40_70):
    """Uniform weights of 10 000 for each age cell."""
    return pd.Series(10_000.0, index=ages_40_70.index, name="etr")


# ---------------------------------------------------------------------------
# 1. Whittaker: λ = 0 → identity (graduated == crude)
# ---------------------------------------------------------------------------

def test_whittaker_lambda_zero_returns_crude(ages_40_70, gompertz_qx, uniform_weights):
    """λ = 0 with equal weights → graduated rates equal the crude rates exactly."""
    graduated = whittaker_1d(gompertz_qx, uniform_weights, lam=0.0)
    pd.testing.assert_series_equal(graduated, gompertz_qx, check_names=False)


# ---------------------------------------------------------------------------
# 2. Whittaker: smoothing reduces sum of squared second differences
# ---------------------------------------------------------------------------

def test_whittaker_smoothing_reduces_roughness():
    """Graduated rates are smoother than crude rates under λ > 0."""
    rng = np.random.default_rng(42)
    ages = pd.Series(range(40, 71))
    # True Gompertz rates + noise
    B, c = 0.0001, 1.10
    x = ages.to_numpy(dtype=float)
    mu_true = B * np.power(c, x)
    noise = rng.normal(0, 0.0002, size=len(ages))
    crude = pd.Series(np.clip(1.0 - np.exp(-mu_true) + noise, 1e-6, 1.0), index=ages.index)
    weights = pd.Series(10_000.0, index=ages.index)

    graduated = whittaker_1d(crude, weights, lam=100.0)

    ss_crude = float(np.sum(np.diff(crude.to_numpy()) ** 2))
    ss_grad = float(np.sum(np.diff(graduated.to_numpy()) ** 2))
    assert ss_grad < ss_crude, (
        f"Graduated series is not smoother: ss_crude={ss_crude:.2e}, ss_grad={ss_grad:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Whittaker: zero-weight cells are interpolated from neighbours
# ---------------------------------------------------------------------------

def test_whittaker_zero_weight_interpolated():
    """Age cells with weight 0 are smoothed from neighbouring cells, not left at crude."""
    ages = pd.Series(range(40, 46))
    # Monotone crude rates; middle cell has weight 0
    crude = pd.Series([0.001, 0.002, 0.999, 0.004, 0.005, 0.006], index=ages.index)
    weights = pd.Series([1e4, 1e4, 0.0, 1e4, 1e4, 1e4], index=ages.index)

    graduated = whittaker_1d(crude, weights, lam=10.0)

    # The zero-weight cell (index 2) must NOT be close to its crude value (0.999)
    assert abs(graduated.iloc[2] - 0.999) > 0.01, (
        "Zero-weight cell was not smoothed away from its crude value"
    )
    # It should lie in a reasonable neighbourhood of its neighbours
    assert 0.0 < graduated.iloc[2] < 0.5


# ---------------------------------------------------------------------------
# 4. Gompertz fit: recovers B and c from synthetic data
# ---------------------------------------------------------------------------

def test_fit_gompertz_recovers_parameters(ages_40_70, gompertz_qx):
    """fit_gompertz recovers B and c within 5 % of the true values."""
    result = fit_gompertz(ages_40_70, gompertz_qx)

    assert "B" in result and "c" in result and "fitted_qx" in result

    B_true, c_true = 0.0001, 1.10
    assert result["B"] == pytest.approx(B_true, rel=0.05), (
        f"B recovery failed: expected ~{B_true}, got {result['B']:.6f}"
    )
    assert result["c"] == pytest.approx(c_true, rel=0.05), (
        f"c recovery failed: expected ~{c_true}, got {result['c']:.6f}"
    )
    assert len(result["fitted_qx"]) == len(ages_40_70)


# ---------------------------------------------------------------------------
# 5. Chi-squared test: well-graduated data passes; crude noisy data fails
# ---------------------------------------------------------------------------

def test_chi_squared_test_output_structure(ages_40_70, gompertz_qx, uniform_weights):
    """chi_squared_test returns a dict with statistic, p_value, df, pass keys."""
    result = chi_squared_test(gompertz_qx, gompertz_qx, uniform_weights)

    assert "statistic" in result
    assert "p_value" in result
    assert "df" in result
    assert "pass" in result

    # Graduated == crude → statistic = 0, p-value = 1.0
    assert result["statistic"] == pytest.approx(0.0, abs=1e-10)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)
    assert result["df"] == len(ages_40_70)
    assert result["pass"] is True


def test_chi_squared_test_noisy_data_may_fail():
    """A highly noisy graduation produces a large chi-squared statistic."""
    rng = np.random.default_rng(7)
    ages = pd.Series(range(40, 71))
    B, c = 0.0001, 1.10
    x = ages.to_numpy(dtype=float)
    mu = B * np.power(c, x)
    true_qx = pd.Series(1.0 - np.exp(-mu), index=ages.index)

    # Graduated rates with large systematic bias
    biased = true_qx * 2.0  # 100% over-estimation

    etr = pd.Series(10_000.0, index=ages.index)
    result = chi_squared_test(true_qx, biased, etr)

    # Huge bias → very large statistic, very small p-value
    assert result["statistic"] > 100.0
    assert result["p_value"] < 0.001
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# 6. graduation_report: sum-of-squares reduction is positive after graduation
# ---------------------------------------------------------------------------

def test_graduation_report_ssr_positive():
    """A Whittaker graduation of noisy data produces positive sum-sq reduction."""
    rng = np.random.default_rng(99)
    ages = pd.Series(range(40, 71))
    B, c = 0.0001, 1.10
    x = ages.to_numpy(dtype=float)
    mu = B * np.power(c, x)
    noise = rng.normal(0, 0.0003, size=len(ages))
    crude = pd.Series(np.clip(1.0 - np.exp(-mu) + noise, 1e-6, 1.0), index=ages.index)
    etr = pd.Series(10_000.0, index=ages.index)

    graduated = whittaker_1d(crude, etr, lam=100.0)
    report = graduation_report(crude, graduated, etr)

    assert "chi_squared" in report
    assert "signs" in report
    assert "sum_sq_reduction" in report

    assert report["sum_sq_reduction"] > 0.0, (
        f"Expected positive SSR, got {report['sum_sq_reduction']:.4f}"
    )
    # Chi-squared result must have all required keys
    chi2 = report["chi_squared"]
    assert all(k in chi2 for k in ("statistic", "p_value", "df", "pass"))
