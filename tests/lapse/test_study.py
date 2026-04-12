"""Tests for lifexp.lapse.study: LapseStudy and LapseResults."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import PolicyDataset
from lifexp.core.study_period import StudyPeriod
from lifexp.lapse.study import LapseStudy, LapseResults


# ---------------------------------------------------------------------------
# Study-period constants
#
# 2020-01-01 → 2024-12-31  (5 full policy years for policies issued 2020-01-01)
# All policies share the same DOB and issue_date so no age or PY boundary falls
# mid-segment — each policy maps cleanly to one or more complete PY cells.
# ---------------------------------------------------------------------------

_STUDY_START = date(2020, 1, 1)
_STUDY_END = date(2024, 12, 31)
_STUDY = StudyPeriod(_STUDY_START, _STUDY_END)

_ISSUE_DATE = date(2020, 1, 1)
_DOB = date(1980, 1, 1)   # age doesn't matter for lapse study


# ---------------------------------------------------------------------------
# Dataset-building helpers
# ---------------------------------------------------------------------------

def _lapse_rows(
    n_lapses_per_py: List[int],
    n_alive: int,
    gender: str = "M",
    pid_offset: int = 0,
) -> List[dict]:
    """Build policy rows for a lapse study.

    Policies lapse mid-year of their designated policy year.
    ``n_lapses_per_py[k]`` = number of lapses in PY k+1 (1-indexed).
    ``n_alive`` = policies that are still in-force at study end.

    Exit date for PY-k lapses = date(2020 + k - 1, 6, 15)
    (June 15 of the calendar year, which is always in PY k for issue=Jan 1).
    """
    rows: List[dict] = []
    pid = pid_offset

    for py_idx, n in enumerate(n_lapses_per_py, start=1):
        exit_date = date(2020 + py_idx - 1, 6, 15)
        for _ in range(n):
            rows.append({
                "policy_id":      f"P{pid + 1:07d}",
                "date_of_birth":  _DOB,
                "issue_date":     _ISSUE_DATE,
                "gender":         gender,
                "smoker_status":  "NS",
                "sum_assured":    100_000.0,
                "annual_premium": 100.0,
                "product_code":   "TERM",
                "channel":        "DIRECT",
                "status":         "LAPSED",
                "exit_date":      exit_date,
                "exit_reason":    "LAPSE",
            })
            pid += 1

    for _ in range(n_alive):
        rows.append({
            "policy_id":      f"P{pid + 1:07d}",
            "date_of_birth":  _DOB,
            "issue_date":     _ISSUE_DATE,
            "gender":         gender,
            "smoker_status":  "NS",
            "sum_assured":    100_000.0,
            "annual_premium": 100.0,
            "product_code":   "TERM",
            "channel":        "DIRECT",
            "status":         "IF",
            "exit_date":      None,
            "exit_reason":    None,
        })
        pid += 1

    return rows


# ---------------------------------------------------------------------------
# Gold-standard fixture: 100 policies, ~10% lapse each PY for 5 years
#
# PY1: 10 lapses out of 100  → rate = 10/100 = 0.100
# PY2:  9 lapses out of  90  → rate =  9/90  = 0.100
# PY3:  8 lapses out of  81  → rate =  8/81  ≈ 0.099
# PY4:  7 lapses out of  73  → rate =  7/73  ≈ 0.096
# PY5:  7 lapses out of  66  → rate =  7/66  ≈ 0.106
# 59 IF at end
#
# Persistency[5] = 59/100 = 0.59  vs  0.9^5 = 0.59049  → |diff| < 0.005
# ---------------------------------------------------------------------------

_LAPSES_PER_PY = [10, 9, 8, 7, 7]
_N_ALIVE = 59   # 100 - 10 - 9 - 8 - 7 - 7


@pytest.fixture(scope="module")
def lapse_study() -> LapseStudy:
    rows = _lapse_rows(_LAPSES_PER_PY, _N_ALIVE)
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows))
    return LapseStudy(dataset, _STUDY)


@pytest.fixture(scope="module")
def lapse_result(lapse_study) -> LapseResults:
    return lapse_study.run()


# ---------------------------------------------------------------------------
# 1. Normal: required columns present
# ---------------------------------------------------------------------------

def test_by_policy_year_columns(lapse_result):
    """by_policy_year() contains all required columns."""
    required = {
        "policy_year", "policies_exposed", "lapses", "surrenders",
        "deaths", "gross_lapse_rate", "net_lapse_rate",
    }
    assert required.issubset(set(lapse_result.by_policy_year().columns))


def test_five_policy_years_present(lapse_result):
    """Study spans PY1–PY5: five rows in by_policy_year()."""
    df = lapse_result.by_policy_year()
    assert set(df["policy_year"]) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# 2. Normal: policies_exposed decreases as policies lapse
# ---------------------------------------------------------------------------

def test_policies_exposed_decreasing(lapse_result):
    """Exposed count decreases strictly from PY1 to PY5."""
    df = lapse_result.by_policy_year().sort_values("policy_year")
    exposed = df["policies_exposed"].tolist()
    for i in range(len(exposed) - 1):
        assert exposed[i] > exposed[i + 1], (
            f"Expected policies_exposed[{i}] > policies_exposed[{i+1}]; "
            f"got {exposed[i]} vs {exposed[i+1]}"
        )


def test_policies_exposed_values(lapse_result):
    """Exact headcounts: 100, 90, 81, 73, 66."""
    df = lapse_result.by_policy_year().sort_values("policy_year").reset_index(drop=True)
    expected = [100, 90, 81, 73, 66]
    for py_idx, exp in enumerate(expected):
        actual = int(df.loc[py_idx, "policies_exposed"])
        assert actual == exp, f"PY{py_idx+1}: exposed={actual}, expected={exp}"


# ---------------------------------------------------------------------------
# 3. Normal: gross lapse rate ≈ 10% for PY1 and PY2 (exact)
# ---------------------------------------------------------------------------

def test_lapse_rate_approx_10pct(lapse_result):
    """PY1 and PY2 gross_lapse_rate = 10/100 and 9/90 = 0.10 exactly."""
    df = lapse_result.by_policy_year().sort_values("policy_year").reset_index(drop=True)
    assert df.loc[0, "gross_lapse_rate"] == pytest.approx(10 / 100, rel=1e-9)
    assert df.loc[1, "gross_lapse_rate"] == pytest.approx(9 / 90, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Normal: persistency[5] ≈ 0.9^5 = 0.59049
# ---------------------------------------------------------------------------

def test_persistency_compound(lapse_result):
    """Persistency at PY5 ≈ 0.9^5 = 0.59049 (within ±0.005)."""
    pt = lapse_result.persistency_table().sort_values("policy_year")
    p5 = float(pt.loc[pt["policy_year"] == 5, "persistency"].iloc[0])
    assert p5 == pytest.approx(0.9 ** 5, abs=0.005), (
        f"Persistency[5] = {p5:.5f}, target = {0.9**5:.5f}"
    )


def test_persistency_exact_values(lapse_result):
    """Cumulative persistency matches exact surviving fractions."""
    pt = lapse_result.persistency_table().sort_values("policy_year").reset_index(drop=True)
    # Surviving counts: 90, 81, 73, 66, 59 out of 100 initial
    expected = [90 / 100, 81 / 100, 73 / 100, 66 / 100, 59 / 100]
    for i, exp in enumerate(expected):
        actual = float(pt.loc[i, "persistency"])
        assert actual == pytest.approx(exp, rel=1e-9), (
            f"PY{i+1}: persistency={actual:.6f}, expected={exp:.6f}"
        )


def test_persistency_monotone_decreasing(lapse_result):
    """Persistency is strictly decreasing when there are lapses in every PY."""
    pt = lapse_result.persistency_table().sort_values("policy_year")
    vals = pt["persistency"].tolist()
    for i in range(len(vals) - 1):
        assert vals[i] > vals[i + 1], (
            f"Persistency should decrease: p[{i}]={vals[i]:.4f} >= p[{i+1}]={vals[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Edge: zero lapses → persistency stays at 1.0
# ---------------------------------------------------------------------------

def test_zero_lapse_persistency_stays_one():
    """All-IF dataset: gross_lapse_rate = 0, persistency = 1.0 for every PY."""
    rows = _lapse_rows([], n_alive=50)  # no lapses
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows))
    result = LapseStudy(dataset, _STUDY).run()

    df = result.by_policy_year()
    assert (df["lapses"] == 0).all(), "Expected zero lapses"
    assert (df["gross_lapse_rate"] == 0.0).all()

    pt = result.persistency_table()
    # cumprod of (1-0)^n = 1.0 exactly in floating point
    assert (pt["persistency"] == 1.0).all(), (
        "Persistency should be 1.0 when no lapses"
    )


# ---------------------------------------------------------------------------
# 6. Normal: net lapse rate adjusts upward in the presence of deaths
# ---------------------------------------------------------------------------

def test_net_lapse_rate_adjusts_for_deaths():
    """With deaths competing, net_lapse_rate > gross_lapse_rate in that PY."""
    # 5 lapses (exit PY1 mid-year) + 5 deaths (exit PY1 mid-year) + 90 IF
    rows: List[dict] = []
    for i in range(5):   # lapses
        rows.append({
            "policy_id": f"L{i+1:04d}",
            "date_of_birth": _DOB,
            "issue_date": _ISSUE_DATE,
            "gender": "M",
            "smoker_status": "NS",
            "sum_assured": 100_000.0,
            "annual_premium": 100.0,
            "product_code": "TERM",
            "channel": "DIRECT",
            "status": "LAPSED",
            "exit_date": date(2020, 6, 15),
            "exit_reason": "LAPSE",
        })
    for i in range(5):   # deaths
        rows.append({
            "policy_id": f"D{i+1:04d}",
            "date_of_birth": _DOB,
            "issue_date": _ISSUE_DATE,
            "gender": "M",
            "smoker_status": "NS",
            "sum_assured": 100_000.0,
            "annual_premium": 100.0,
            "product_code": "TERM",
            "channel": "DIRECT",
            "status": "DEATH",
            "exit_date": date(2020, 6, 15),
            "exit_reason": "DEATH",
        })
    for i in range(90):  # in-force
        rows.append({
            "policy_id": f"IF{i+1:05d}",
            "date_of_birth": _DOB,
            "issue_date": _ISSUE_DATE,
            "gender": "M",
            "smoker_status": "NS",
            "sum_assured": 100_000.0,
            "annual_premium": 100.0,
            "product_code": "TERM",
            "channel": "DIRECT",
            "status": "IF",
            "exit_date": None,
            "exit_reason": None,
        })

    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows))
    result = LapseStudy(dataset, _STUDY).run()

    df = result.by_policy_year()
    py1 = df.loc[df["policy_year"] == 1].iloc[0]

    # gross = 5/100 = 0.05; net = 5/(100 - 5/2) = 5/97.5 ≈ 0.0513
    assert py1["gross_lapse_rate"] == pytest.approx(5 / 100, rel=1e-9)
    assert py1["net_lapse_rate"] == pytest.approx(5 / 97.5, rel=1e-9)
    assert py1["net_lapse_rate"] > py1["gross_lapse_rate"]


# ---------------------------------------------------------------------------
# 7. Normal: group_by splits results correctly
# ---------------------------------------------------------------------------

def test_group_by_gender():
    """group_by=['gender'] → separate M and F rows; aggregate is consistent."""
    # 100 M policies: 10% lapse each PY (same as gold-standard fixture)
    # 100 F policies: all IF (0% lapse)
    rows_m = _lapse_rows(_LAPSES_PER_PY, _N_ALIVE, gender="M", pid_offset=0)
    rows_f = _lapse_rows([], n_alive=100, gender="F", pid_offset=200)
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows_m + rows_f))

    result = LapseStudy(dataset, _STUDY, group_by=["gender"]).run()
    df = result.by_policy_year()

    assert set(df["gender"].unique()) == {"M", "F"}

    # Males: PY1 rate = 10/100 = 0.10
    m_py1 = df.loc[(df["gender"] == "M") & (df["policy_year"] == 1)].iloc[0]
    assert m_py1["gross_lapse_rate"] == pytest.approx(0.10, rel=1e-9)

    # Females: all PYs have 0 lapses
    f_rows = df.loc[df["gender"] == "F"]
    assert (f_rows["lapses"] == 0).all()
    assert (f_rows["gross_lapse_rate"] == 0.0).all()

    # Combined persistency[5] for Males ≈ 0.9^5
    pt = result.persistency_table()
    m_p5 = float(
        pt.loc[(pt["gender"] == "M") & (pt["policy_year"] == 5), "persistency"].iloc[0]
    )
    assert m_p5 == pytest.approx(0.9 ** 5, abs=0.005)


# ---------------------------------------------------------------------------
# 8. Normal: A/E vs assumption
# ---------------------------------------------------------------------------

def test_ae_vs_assumption_at_one(lapse_result):
    """A/E = 1.0 when assumed rate equals observed gross_lapse_rate."""
    df = lapse_result.by_policy_year()
    # Use the actual observed rates as the assumed rates
    assumption = df[["policy_year", "gross_lapse_rate"]].rename(
        columns={"gross_lapse_rate": "assumed_lapse_rate"}
    )

    ae_df = lapse_result.ae_vs_assumption(assumption)
    assert "ae_ratio" in ae_df.columns
    # A/E should be ≈ 1.0 for every PY
    for _, row in ae_df.iterrows():
        assert row["ae_ratio"] == pytest.approx(1.0, rel=1e-9), (
            f"PY{int(row['policy_year'])}: A/E = {row['ae_ratio']:.4f}"
        )


def test_ae_vs_assumption_half_rate():
    """A/E ≈ 2.0 when assumed rate is half the observed rate."""
    rows = _lapse_rows([10], n_alive=90)   # 10 lapses in PY1, 90 IF after
    dataset = PolicyDataset.from_dataframe(pd.DataFrame(rows))
    result = LapseStudy(dataset, _STUDY).run()

    # Assume half the true rate → A/E should be ≈ 2.0
    assumption = pd.DataFrame({"policy_year": [1], "assumed_lapse_rate": [0.05]})
    ae_df = result.ae_vs_assumption(assumption)

    py1_ae = float(ae_df.loc[ae_df["policy_year"] == 1, "ae_ratio"].iloc[0])
    assert py1_ae == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 9. Normal: Kaplan-Meier survival is non-increasing
# ---------------------------------------------------------------------------

def test_survival_curve_monotone(lapse_result):
    """KM survival estimate is non-increasing over policy years."""
    sc = lapse_result.survival_curve().sort_values("policy_year")
    vals = sc["km_survival"].tolist()
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1], (
            f"KM survival should not increase: "
            f"S({i+1})={vals[i]:.4f} < S({i})={vals[i-1]:.4f}"
        )


def test_survival_curve_no_deaths_equals_persistency(lapse_result):
    """Without deaths, KM survival equals the persistency table."""
    sc = lapse_result.survival_curve().sort_values("policy_year").reset_index(drop=True)
    pt = lapse_result.persistency_table().sort_values("policy_year").reset_index(drop=True)

    np.testing.assert_allclose(
        sc["km_survival"].to_numpy(),
        pt["persistency"].to_numpy(),
        rtol=1e-9,
        err_msg="KM survival should equal persistency when deaths = 0",
    )
