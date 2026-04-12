"""Tests for lifexp.core.credibility and lifexp.core.segmentation."""

import math

import numpy as np
import pandas as pd
import pytest

from lifexp.core.credibility import (
    CredibilityConfig,
    apply_credibility,
    blend_rates,
    classical_credibility,
)
from lifexp.core.segmentation import segment
from lifexp.core.tables import TableRegistry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> CredibilityConfig:
    return CredibilityConfig()  # threshold = 1082


@pytest.fixture
def a1967_70():
    return TableRegistry().get("A_1967_70")


# ---------------------------------------------------------------------------
# Normal: full credibility at threshold
# ---------------------------------------------------------------------------

def test_classical_credibility_at_threshold(default_config):
    """deaths = 1082 (exactly at threshold) → Z = 1.0."""
    z = classical_credibility(1082, default_config)
    assert z == pytest.approx(1.0, abs=1e-12)


def test_blended_rate_at_full_credibility(default_config):
    """At Z = 1.0, blended_rate equals the experience rate exactly."""
    z = classical_credibility(1082, default_config)
    assert z == pytest.approx(1.0)
    blended = blend_rates(0.00250, 0.00179, z)
    assert blended == pytest.approx(0.00250, rel=1e-9)


# ---------------------------------------------------------------------------
# Normal: zero credibility
# ---------------------------------------------------------------------------

def test_classical_credibility_zero_deaths(default_config):
    """deaths = 0 → Z = 0.0."""
    z = classical_credibility(0, default_config)
    assert z == pytest.approx(0.0, abs=1e-12)


def test_blended_rate_at_zero_credibility(default_config):
    """At Z = 0.0, blended_rate equals the standard rate regardless of experience."""
    z = classical_credibility(0, default_config)
    blended = blend_rates(0.99, 0.00179, z)
    assert blended == pytest.approx(0.00179, rel=1e-9)


# ---------------------------------------------------------------------------
# Normal: partial credibility at ¼ threshold
# ---------------------------------------------------------------------------

def test_classical_credibility_partial(default_config):
    """deaths = 271 (≈ ¼ threshold) → Z ≈ 0.5 (sqrt(271/1082))."""
    z = classical_credibility(271, default_config)
    expected = math.sqrt(271 / 1082)
    assert z == pytest.approx(expected, rel=1e-9)
    # Approximately half-credibility
    assert z == pytest.approx(0.5, abs=0.005)


# ---------------------------------------------------------------------------
# Edge: deaths above threshold → Z capped at 1.0
# ---------------------------------------------------------------------------

def test_classical_credibility_cap(default_config):
    """deaths = 5000 (far above threshold) → Z capped at 1.0."""
    z = classical_credibility(5000, default_config)
    assert z == pytest.approx(1.0, abs=1e-12)
    assert z <= 1.0  # never exceeds 1


def test_classical_credibility_cap_very_large(default_config):
    """Any large death count is capped."""
    for d in (2000, 10_000, 1_000_000):
        assert classical_credibility(d, default_config) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Boundary: blending invariant
# ---------------------------------------------------------------------------

def test_blend_invariant_equal_rates(default_config):
    """When experience == standard, blend == standard regardless of Z."""
    rate = 0.00427
    for deaths in (0, 100, 271, 1082, 5000):
        z = classical_credibility(deaths, default_config)
        blended = blend_rates(rate, rate, z)
        assert blended == pytest.approx(rate, rel=1e-9), (
            f"Blending invariant failed at deaths={deaths}, Z={z:.4f}"
        )


def test_blend_nan_experience_falls_back_to_standard():
    """NaN experience rate (no exposure) → blend returns standard rate."""
    blended = blend_rates(float("nan"), 0.00179, 0.5)
    assert blended == pytest.approx(0.00179, rel=1e-9)


# ---------------------------------------------------------------------------
# apply_credibility: DataFrame-level function
# ---------------------------------------------------------------------------

def test_apply_credibility_adds_required_columns(a1967_70, default_config):
    """apply_credibility adds credibility_z, blended_rate, is_fully_credible."""
    seg = pd.DataFrame({
        "age": [40, 40, 50],
        "deaths": [0.0, 1082.0, 271.0],
        "crude_rate": [0.0, 0.00280, 0.00550],
        "central_etr": [500.0, 386_000.0, 63_400.0],
    })
    result = apply_credibility(seg, a1967_70, default_config)

    assert "credibility_z" in result.columns
    assert "blended_rate" in result.columns
    assert "is_fully_credible" in result.columns

    # Row 0: deaths=0 → Z=0, blended=standard
    assert result.loc[0, "credibility_z"] == pytest.approx(0.0)
    assert result.loc[0, "blended_rate"] == pytest.approx(a1967_70.qx(40), rel=1e-9)
    assert not result.loc[0, "is_fully_credible"]

    # Row 1: deaths=1082 → Z=1.0, blended=experience
    assert result.loc[1, "credibility_z"] == pytest.approx(1.0)
    assert result.loc[1, "blended_rate"] == pytest.approx(0.00280, rel=1e-9)
    assert result.loc[1, "is_fully_credible"]

    # Row 2: deaths=271 → Z≈0.5
    assert result.loc[2, "credibility_z"] == pytest.approx(math.sqrt(271 / 1082), rel=1e-9)
    assert not result.loc[2, "is_fully_credible"]


def test_apply_credibility_missing_columns_raises(a1967_70, default_config):
    """apply_credibility raises ValueError if required columns are absent."""
    bad_df = pd.DataFrame({"age": [40], "deaths": [100]})  # missing crude_rate
    with pytest.raises(ValueError, match="crude_rate"):
        apply_credibility(bad_df, a1967_70, default_config)


# ---------------------------------------------------------------------------
# CredibilityConfig: custom threshold
# ---------------------------------------------------------------------------

def test_custom_threshold():
    """A custom threshold changes the full-credibility boundary."""
    config = CredibilityConfig(full_credibility_threshold=400)
    assert classical_credibility(400, config) == pytest.approx(1.0, abs=1e-12)
    assert classical_credibility(100, config) == pytest.approx(0.5, abs=1e-3)
    assert classical_credibility(800, config) == pytest.approx(1.0, abs=1e-12)  # capped


# ---------------------------------------------------------------------------
# segment: merge + crude rate
# ---------------------------------------------------------------------------

def test_segment_merge_and_crude_rate():
    """segment merges ETR and deaths, computes crude_rate, and adds is_credible flag."""
    etr_df = pd.DataFrame({
        "age": [40, 40, 41],
        "policy_year": [1, 2, 1],
        "central_etr": [100.0, 50.0, 80.0],
        "policy_count": [10, 5, 8],
    })
    deaths_df = pd.DataFrame({
        "age": [40, 40, 41],
        "policy_year": [1, 2, 1],
        "deaths": [2.0, 1.0, 0.0],
        "initial_etr": [1.0, 1.0, 0.0],  # extra column should be ignored
    })
    result = segment(etr_df, deaths_df, group_by=[])

    assert len(result) == 2  # age 40 and age 41 after aggregation

    row40 = result[result["age"] == 40].iloc[0]
    row41 = result[result["age"] == 41].iloc[0]

    # Age 40: ETR = 150, deaths = 3
    assert row40["central_etr"] == pytest.approx(150.0)
    assert row40["deaths"] == pytest.approx(3.0)
    assert row40["crude_rate"] == pytest.approx(3.0 / 150.0, rel=1e-9)
    assert not row40["is_credible"]  # 3 << 1082

    # Age 41: deaths = 0, crude_rate = 0.0 (not NaN — there is exposure)
    assert row41["deaths"] == pytest.approx(0.0)
    assert row41["crude_rate"] == pytest.approx(0.0, abs=1e-12)


def test_segment_group_by():
    """segment respects group_by and keeps separate rows per group."""
    etr_df = pd.DataFrame({
        "gender": ["M", "F"],
        "age": [40, 40],
        "policy_year": [1, 1],
        "central_etr": [200.0, 150.0],
        "policy_count": [20, 15],
    })
    deaths_df = pd.DataFrame({
        "gender": ["M", "F"],
        "age": [40, 40],
        "policy_year": [1, 1],
        "deaths": [3.0, 1.0],
    })
    result = segment(etr_df, deaths_df, group_by=["gender"])

    assert len(result) == 2
    m = result[result["gender"] == "M"].iloc[0]
    f = result[result["gender"] == "F"].iloc[0]

    assert m["crude_rate"] == pytest.approx(3.0 / 200.0, rel=1e-9)
    assert f["crude_rate"] == pytest.approx(1.0 / 150.0, rel=1e-9)


def test_segment_zero_etr_gives_nan_crude_rate():
    """When central_etr = 0, crude_rate is NaN (no meaningful rate)."""
    etr_df = pd.DataFrame({
        "age": [40],
        "policy_year": [1],
        "central_etr": [0.0],
        "policy_count": [0],
    })
    deaths_df = pd.DataFrame({
        "age": [40],
        "policy_year": [1],
        "deaths": [0.0],
    })
    result = segment(etr_df, deaths_df, group_by=[])
    assert pd.isna(result.iloc[0]["crude_rate"])
