"""Tests for lifexp.mortality.study: MortalityStudy and MortalityResults."""

from __future__ import annotations

from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import PolicyDataset
from lifexp.core.study_period import AgeBasis, StudyPeriod
from lifexp.core.tables import TableRegistry
from lifexp.mortality.study import MortalityStudy, MortalityResults


# ---------------------------------------------------------------------------
# Study-period constants used across all fixtures
#
# 2023-07-01 → 2024-06-30  (366 days inclusive, spans a leap Feb)
# DOB  = date(YEAR - age, 7, 1)  → ALB == age at study start; no birthday
#         falls strictly inside (obs_start, obs_end] since 2024-07-01 > obs_end
# ISSUE= date(YEAR - 3, 7, 1)    → no policy-anniversary inside the window
#         (2023-07-01 == obs_start is NOT strictly >, 2024-07-01 > obs_end)
# ---------------------------------------------------------------------------

_STUDY_START = date(2023, 7, 1)
_STUDY_END = date(2024, 6, 30)
_STUDY = StudyPeriod(_STUDY_START, _STUDY_END)
_STUDY_DAYS = (_STUDY_END - _STUDY_START).days + 1  # 366
_ETR_PER_POLICY = _STUDY_DAYS / 365.25              # ≈ 1.002055


# ---------------------------------------------------------------------------
# Dataset-building helpers
# ---------------------------------------------------------------------------

def _n_for_ae(qx: float, n_deaths: int, target_ae: float) -> int:
    """Total policy count to achieve AE ≈ target_ae for a given age."""
    exact = n_deaths / (qx * target_ae * _ETR_PER_POLICY)
    return max(n_deaths, round(exact))


def _policy_rows(
    age: int,
    n_deaths: int,
    n_total: int,
    gender: str = "M",
    pid_offset: int = 0,
) -> List[dict]:
    """Generate policy records for one age cell.

    * Death policies die at study_end (full-study central ETR).
    * Alive (IF) policies are in-force through study_end.
    * DOB and issue_date are chosen so no age or policy-year boundary
      falls strictly inside the study window — each policy occupies
      exactly one (age, policy_year) segment.
    """
    dob = date(_STUDY_START.year - age, _STUDY_START.month, _STUDY_START.day)
    issue = date(_STUDY_START.year - 3, _STUDY_START.month, _STUDY_START.day)
    rows = []
    for i in range(n_total):
        is_death = i < n_deaths
        rows.append({
            "policy_id":       f"P{pid_offset + i + 1:07d}",
            "date_of_birth":   dob,
            "issue_date":      issue,
            "gender":          gender,
            "smoker_status":   "NS",
            "sum_assured":     100_000.0,
            "annual_premium":  100.0,
            "product_code":    "TERM",
            "channel":         "DIRECT",
            "status":          "DEATH" if is_death else "IF",
            "exit_date":       _STUDY_END if is_death else None,
            "exit_reason":     "DEATH" if is_death else None,
        })
    return rows


def _build_dataset(age_specs: List[Tuple], gender: str = "M") -> PolicyDataset:
    """Build a PolicyDataset from a list of (age, n_deaths, n_total) tuples."""
    all_rows: List[dict] = []
    for age, n_deaths, n_total in age_specs:
        offset = len(all_rows)
        all_rows.extend(_policy_rows(age, n_deaths, n_total, gender, offset))
    return PolicyDataset.from_dataframe(pd.DataFrame(all_rows))


# ---------------------------------------------------------------------------
# Gold-standard fixture: A/E ≈ 1.0
#
# Ages 40, 50, 60 — one death each.  n_total chosen so that
# deaths / (qx × n_total × ETR_per_policy) ≈ 1.0 (within ±0.01).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ae_one_study() -> MortalityStudy:
    """MortalityStudy whose run() produces overall_ae ≈ 1.0."""
    table = TableRegistry().get("A_1967_70")
    specs = [
        (age, 1, _n_for_ae(table.qx(age), 1, 1.0))
        for age in (40, 50, 60)
    ]
    dataset = _build_dataset(specs)
    return MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY)


@pytest.fixture(scope="module")
def ae_one_result(ae_one_study) -> MortalityResults:
    return ae_one_study.run()


# ---------------------------------------------------------------------------
# 1. Normal: A/E = 1.0 when experience matches table
# ---------------------------------------------------------------------------

def test_ae_one_overall(ae_one_result):
    """Gold-standard: overall A/E ≈ 1.0 (within ±0.01)."""
    assert ae_one_result.overall_ae == pytest.approx(1.0, abs=0.01)


def test_ae_one_total_deaths(ae_one_result):
    """Three ages × 1 death = 3 total deaths."""
    assert ae_one_result.total_deaths == pytest.approx(3.0)


def test_ae_one_summary_columns(ae_one_result):
    """summary_df contains all required columns."""
    required = {
        "age", "central_etr", "initial_etr", "deaths",
        "crude_central_rate", "crude_initial_rate",
        "standard_rate", "expected_deaths",
        "ae_ratio", "ae_central", "ae_initial",
    }
    assert required.issubset(set(ae_one_result.summary_df.columns))


def test_ae_one_ae_by_age_consistent(ae_one_result):
    """ae_by_age() totals agree with overall_ae."""
    age_df = ae_one_result.ae_by_age()
    recomputed_ae = age_df["deaths"].sum() / age_df["expected_deaths"].sum()
    assert recomputed_ae == pytest.approx(ae_one_result.overall_ae, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Normal: A/E = 0.5 when deaths = half expected
# ---------------------------------------------------------------------------

def test_ae_half():
    """A/E ≈ 0.5 when ETR is doubled relative to expected deaths."""
    table = TableRegistry().get("A_1967_70")
    age = 50
    qx = table.qx(age)
    # 1 death; double the n_total → expected ≈ 2 → A/E ≈ 0.5
    n_total = _n_for_ae(qx, 1, 0.5)
    dataset = _build_dataset([(age, 1, n_total)])
    result = MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY).run()
    assert result.overall_ae == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Normal: Male/female segment A/E
# ---------------------------------------------------------------------------

def test_gender_segment_ae():
    """group_by=['gender'] → separate M and F A/E; aggregate = total A/E."""
    table = TableRegistry().get("A_1967_70")
    age = 50
    qx = table.qx(age)

    # Males: 1 death, n_total for AE ≈ 1.0
    n_m = _n_for_ae(qx, 1, 1.0)
    # Females: 1 death, n_total for AE ≈ 0.5 (double ETR)
    n_f = _n_for_ae(qx, 1, 0.5)

    rows_m = _policy_rows(age, 1, n_m, gender="M", pid_offset=0)
    rows_f = _policy_rows(age, 1, n_f, gender="F", pid_offset=n_m)
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows_m + rows_f))

    result = MortalityStudy(
        dataset, _STUDY, AgeBasis.LAST_BIRTHDAY, group_by=["gender"]
    ).run()

    summary = result.summary_df
    assert set(summary["gender"].unique()) == {"M", "F"}

    m_ae = summary.loc[summary["gender"] == "M", "ae_ratio"].iloc[0]
    f_ae = summary.loc[summary["gender"] == "F", "ae_ratio"].iloc[0]

    assert m_ae == pytest.approx(1.0, abs=0.02)
    assert f_ae == pytest.approx(0.5, abs=0.02)

    # Weighted average by expected deaths must equal overall A/E
    total_d = summary["deaths"].sum()
    total_e = summary["expected_deaths"].sum()
    weighted_ae = total_d / total_e
    assert weighted_ae == pytest.approx(result.overall_ae, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Normal: Confidence intervals widen with fewer deaths
# ---------------------------------------------------------------------------

def test_ci_wider_for_fewer_deaths():
    """CI width at a low-death age > CI width at a high-death age."""
    table = TableRegistry().get("A_1967_70")

    # Age 40 (qx=0.00179): 1 death → expected ≈ 1.0 → wide CI
    age_lo, deaths_lo = 40, 1
    n_lo = _n_for_ae(table.qx(age_lo), deaths_lo, 1.0)

    # Age 70 (qx=0.02744): 5 deaths → expected ≈ 5.0 → narrow CI
    age_hi, deaths_hi = 70, 5
    n_hi = _n_for_ae(table.qx(age_hi), deaths_hi, 1.0)

    dataset = _build_dataset([
        (age_lo, deaths_lo, n_lo),
        (age_hi, deaths_hi, n_hi),
    ])
    result = MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY).run()
    ci_df = result.confidence_interval(level=0.95)

    width_lo = float(ci_df.loc[ci_df["age"] == age_lo, "ci_width"].iloc[0])
    width_hi = float(ci_df.loc[ci_df["age"] == age_hi, "ci_width"].iloc[0])

    assert width_lo > width_hi, (
        f"CI width at age {age_lo} ({width_lo:.3f}) should exceed "
        f"width at age {age_hi} ({width_hi:.3f})"
    )


def test_ci_required_keys():
    """confidence_interval() returns ae_ci_lower, ae_ci_upper, ci_width columns."""
    table = TableRegistry().get("A_1967_70")
    n = _n_for_ae(table.qx(50), 1, 1.0)
    dataset = _build_dataset([(50, 1, n)])
    ci_df = MortalityStudy(dataset, _STUDY).run().confidence_interval()
    assert {"ae_ci_lower", "ae_ci_upper", "ci_width"}.issubset(ci_df.columns)


# ---------------------------------------------------------------------------
# 5. Edge: zero deaths in an age cell
# ---------------------------------------------------------------------------

def test_zero_deaths_age_cell():
    """Age cell with 0 deaths: ae_ratio=0.0, CI lower=0.0, no division error."""
    table = TableRegistry().get("A_1967_70")

    # Age 50 with 1 death (reference), age 40 with 0 deaths
    n_50 = _n_for_ae(table.qx(50), 1, 1.0)
    n_40 = 100  # alive only, no deaths

    dataset = _build_dataset([(50, 1, n_50), (40, 0, n_40)])
    result = MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY).run()

    row40 = result.summary_df.loc[result.summary_df["age"] == 40].iloc[0]
    assert row40["deaths"] == pytest.approx(0.0)
    assert row40["ae_ratio"] == pytest.approx(0.0, abs=1e-12)

    # CI lower for zero deaths must be 0.0
    ci_df = result.confidence_interval()
    ci40 = ci_df.loc[ci_df["age"] == 40].iloc[0]
    assert ci40["ae_ci_lower"] == pytest.approx(0.0, abs=1e-12)
    assert ci40["ae_ci_upper"] > 0.0  # upper bound is finite and positive


# ---------------------------------------------------------------------------
# 6. Edge: all policies die
# ---------------------------------------------------------------------------

def test_all_policies_die():
    """Every policy has status='DEATH': ae_ratio computable; initial_etr = death count."""
    # 10 death policies at age 50, dying at different times during study
    age = 50
    dob = date(_STUDY_START.year - age, _STUDY_START.month, _STUDY_START.day)
    issue = date(_STUDY_START.year - 3, _STUDY_START.month, _STUDY_START.day)

    # Deaths at 10 equally-spaced dates from study start to study end
    from datetime import timedelta as _td
    n = 10
    death_dates = [
        _STUDY_START + _td(days=int(i * _STUDY_DAYS / n))
        for i in range(1, n + 1)
    ]
    death_dates[-1] = _STUDY_END  # ensure last date is exactly study_end

    rows = [
        {
            "policy_id":      f"D{i+1:04d}",
            "date_of_birth":  dob,
            "issue_date":     issue,
            "gender":         "M",
            "smoker_status":  "NS",
            "sum_assured":    100_000.0,
            "annual_premium": 100.0,
            "product_code":   "TERM",
            "channel":        "DIRECT",
            "status":         "DEATH",
            "exit_date":      death_dates[i],
            "exit_reason":    "DEATH",
        }
        for i in range(n)
    ]
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows))
    result = MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY).run()

    assert result.total_deaths == pytest.approx(float(n), abs=1e-6)

    # initial_etr = sum of 1.0 per death = n (each death contributes exactly 1.0)
    row = result.summary_df.loc[result.summary_df["age"] == age].iloc[0]
    assert row["initial_etr"] == pytest.approx(float(n), abs=1e-6)

    # A/E is computable (not NaN, not inf)
    assert np.isfinite(result.overall_ae)
    assert result.overall_ae > 0.0


# ---------------------------------------------------------------------------
# 7. Boundary: single policy, single death
# ---------------------------------------------------------------------------

def test_single_policy_single_death():
    """1 death policy → valid result; ae_ratio = 1 / expected_deaths."""
    age = 50
    table = TableRegistry().get("A_1967_70")
    qx = table.qx(age)

    dataset = _build_dataset([(age, 1, 1)])  # 1 policy, 1 death, 0 alive
    result = MortalityStudy(dataset, _STUDY, AgeBasis.LAST_BIRTHDAY).run()

    assert result.total_deaths == pytest.approx(1.0)

    expected = qx * _ETR_PER_POLICY  # qx × (1 policy × 366/365.25)
    assert result.overall_ae == pytest.approx(1.0 / expected, rel=1e-6)

    # Summary and CI must be finite
    assert np.isfinite(result.overall_ae)
    ci_df = result.confidence_interval()
    assert len(ci_df) == 1
    assert np.isfinite(ci_df.iloc[0]["ae_ci_upper"])


# ---------------------------------------------------------------------------
# Extra: graduate() method returns valid DataFrame with graduated_rate column
# ---------------------------------------------------------------------------

def test_graduate_returns_graduated_rate(ae_one_result):
    """graduate() with Whittaker produces graduated_rate column."""
    grad_df = ae_one_result.graduate(method="whittaker", lam=100.0)
    assert "graduated_rate" in grad_df.columns
    assert "crude_central_rate" in grad_df.columns
    assert not grad_df["graduated_rate"].isna().all()
    assert len(grad_df) == len(ae_one_result.ae_by_age())
