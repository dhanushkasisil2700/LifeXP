"""Tests for dual_etr in lifexp.core.exposure: morbidity ETR split."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import ClaimDataset, ClaimRecord, PolicyDataset
from lifexp.core.exposure import DualETRResult, dual_etr
from lifexp.core.study_period import AgeBasis, StudyPeriod


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_STUDY_START = date(2020, 1, 1)
_STUDY_END = date(2022, 12, 31)
_STUDY = StudyPeriod(_STUDY_START, _STUDY_END)

# Total calendar days in the study (2020 is leap): 366 + 365 + 365 = 1096
_STUDY_DAYS = (
    (_STUDY_END - _STUDY_START).days + 1
)  # 1096
_TOTAL_ETR = _STUDY_DAYS / 365.25

_DOB = date(1970, 1, 1)
_ISSUE = date(2019, 1, 1)


# ---------------------------------------------------------------------------
# Dataset-building helpers
# ---------------------------------------------------------------------------

def _pol(pid: str, status: str = "IF", exit_date=None) -> dict:
    return {
        "policy_id": pid,
        "date_of_birth": _DOB,
        "issue_date": _ISSUE,
        "gender": "M",
        "smoker_status": "NS",
        "sum_assured": 100_000.0,
        "annual_premium": 100.0,
        "product_code": "IP",
        "channel": "DIRECT",
        "status": status,
        "exit_date": exit_date,
        "exit_reason": status if status != "IF" else None,
    }


def _clm(
    cid: str,
    pid: str,
    start: date,
    end: date | None = None,
    status: str | None = None,
) -> dict:
    if status is None:
        status = "OPEN" if end is None else "CLOSED_RECOVERY"
    return {
        "claim_id": cid,
        "policy_id": pid,
        "claim_start_date": start,
        "claim_end_date": end,
        "claim_status": status,
        "benefit_type": "PERIODIC",
        "claim_amount": 1_000.0,
        "benefit_period_days": None,
    }


def _days(start: date, end: date) -> int:
    """Inclusive day count."""
    return (end - start).days + 1


def _etr(start: date, end: date) -> float:
    return _days(start, end) / 365.25


# ---------------------------------------------------------------------------
# 1. Checkpoint S13 — Conservation invariant on a 200-policy dataset
#
# For every policy: healthy_etr + sick_etr == total_etr (within 1e-9 rel).
#
# Six groups with different claim patterns:
#   G1 (80):  never ill                   → all healthy
#   G2 (50):  one short claim             → healthy + sick = total
#   G3 (30):  one long claim              → spans a calendar year boundary
#   G4 (20):  two separate claims         → gap between claims stays healthy
#   G5 (10):  open claim at study end     → sick_etr counted to study_end
#   G6 (10):  claim started before study  → sick from obs_start to claim_end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def large_dataset():
    """200-policy dataset with diverse claim patterns for conservation test."""
    pol_rows: List[dict] = []
    clm_rows: List[dict] = []
    pid_counter = 0
    cid_counter = 0

    def _next_pid():
        nonlocal pid_counter
        pid_counter += 1
        return f"P{pid_counter:05d}"

    def _next_cid():
        nonlocal cid_counter
        cid_counter += 1
        return f"C{cid_counter:05d}"

    # G1: 80 never-ill policies
    for _ in range(80):
        pid = _next_pid()
        pol_rows.append(_pol(pid))

    # G2: 50 policies, one claim 2020-06-01 → 2020-08-31 (92 days)
    for _ in range(50):
        pid = _next_pid()
        pol_rows.append(_pol(pid))
        clm_rows.append(_clm(_next_cid(), pid, date(2020, 6, 1), date(2020, 8, 31)))

    # G3: 30 policies, one long claim 2020-03-01 → 2021-12-31 (spans year boundary)
    for _ in range(30):
        pid = _next_pid()
        pol_rows.append(_pol(pid))
        clm_rows.append(_clm(_next_cid(), pid, date(2020, 3, 1), date(2021, 12, 31)))

    # G4: 20 policies, two claims with a gap
    for _ in range(20):
        pid = _next_pid()
        pol_rows.append(_pol(pid))
        clm_rows.append(_clm(_next_cid(), pid, date(2020, 3, 1), date(2020, 4, 30)))
        clm_rows.append(_clm(_next_cid(), pid, date(2021, 3, 1), date(2021, 4, 30)))

    # G5: 10 policies, open claim starting 2021-06-01 (no end date)
    for _ in range(10):
        pid = _next_pid()
        pol_rows.append(_pol(pid))
        clm_rows.append(_clm(_next_cid(), pid, date(2021, 6, 1)))  # OPEN

    # G6: 10 policies, claim started BEFORE study (2019-06-01 → 2020-04-30)
    for _ in range(10):
        pid = _next_pid()
        pol_rows.append(_pol(pid))
        clm_rows.append(_clm(_next_cid(), pid, date(2019, 6, 1), date(2020, 4, 30)))

    policy_ds = PolicyDataset.from_dataframe(pd.DataFrame(pol_rows))
    claim_ds = ClaimDataset.from_dataframe(pd.DataFrame(clm_rows))
    return policy_ds, claim_ds


def test_conservation_invariant(large_dataset):
    """healthy_etr + sick_etr == total_etr for every policy in the 200-policy dataset."""
    policy_ds, claim_ds = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=0)

    pp = result.per_policy
    assert len(pp) == 200, f"Expected 200 rows in per_policy, got {len(pp)}"

    for _, row in pp.iterrows():
        computed = row["healthy_etr"] + row["sick_etr"]
        expected = row["total_etr"]
        assert computed == pytest.approx(expected, rel=1e-9), (
            f"policy_id={row['policy_id']}: "
            f"healthy({row['healthy_etr']:.8f}) + "
            f"sick({row['sick_etr']:.8f}) = {computed:.8f} "
            f"≠ total({expected:.8f})"
        )


def test_conservation_invariant_total_count(large_dataset):
    """Sum of healthy_etr + sum of sick_etr across all policies = 200 × total_etr."""
    policy_ds, claim_ds = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=0)

    pp = result.per_policy
    combined = pp["healthy_etr"].sum() + pp["sick_etr"].sum()
    expected = pp["total_etr"].sum()
    assert combined == pytest.approx(expected, rel=1e-9)


def test_conservation_invariant_with_nonzero_deferred(large_dataset):
    """Conservation holds even when deferred_days=14 (deferred period shifts healthy/sick boundary)."""
    policy_ds, claim_ds = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=14)

    for _, row in result.per_policy.iterrows():
        computed = row["healthy_etr"] + row["sick_etr"]
        assert computed == pytest.approx(row["total_etr"], rel=1e-9), (
            f"policy_id={row['policy_id']}: conservation fails with deferred_days=14"
        )


# ---------------------------------------------------------------------------
# 2. Normal: deferred period exclusion
#
# Claim starts 2020-01-10, ends 2020-01-20 (11 days), deferred=30.
# incidence_date = 2020-01-10 + 30 = 2020-02-09
# claim_end = 2020-01-20 < 2020-02-09 → claim does NOT survive deferred period
# → new_claims = 0, sick_etr = 0 (entire period is healthy)
# ---------------------------------------------------------------------------

def test_deferred_excludes_short_claim():
    """Claim ending before the deferred period elapses: not counted as incidence."""
    pid = "D001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))

    # Claim lasts 11 days; deferred period is 30 → claim terminates before deferral ends
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, date(2020, 1, 10), date(2020, 1, 20)),
    ]))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=30)

    assert result.new_claims.empty or result.new_claims["new_claims"].sum() == 0, (
        "Claim that terminates within deferred period should not be counted"
    )
    assert result.sick_etr.empty or result.sick_etr["sick_etr"].sum() == pytest.approx(
        0.0, abs=1e-12
    ), "No sick ETR when claim ends before deferred period"


def test_deferred_excludes_claim_incidence_date_outside_study():
    """incidence_date = claim_start + deferred falls after study end → not counted."""
    pid = "D002"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))

    # Claim starts 2022-12-15, deferred=30 → incidence 2023-01-14 > study_end 2022-12-31
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C002", pid, date(2022, 12, 15), None),  # open
    ]))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=30)

    assert result.new_claims.empty or result.new_claims["new_claims"].sum() == 0, (
        "Incidence date after study end should not be counted"
    )


# ---------------------------------------------------------------------------
# 3. Normal: never-ill policy
#
# Policy with no claims → healthy_etr = full observation, sick_etr = 0.
# ---------------------------------------------------------------------------

def test_never_ill_policy():
    """Policy with no claims: healthy_etr = total_etr, sick_etr = 0."""
    pid = "H001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame(
        columns=["claim_id", "policy_id", "claim_start_date", "claim_end_date",
                 "claim_status", "benefit_type", "claim_amount", "benefit_period_days"]
    ))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    pp = result.per_policy.iloc[0]
    assert pp["healthy_etr"] == pytest.approx(pp["total_etr"], rel=1e-9)
    assert pp["sick_etr"] == pytest.approx(0.0, abs=1e-12)
    assert result.sick_etr.empty or result.sick_etr["sick_etr"].sum() == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.new_claims.empty or result.new_claims["new_claims"].sum() == 0


def test_never_ill_healthy_etr_exact_value():
    """Never-ill policy: healthy_etr = study_days / 365.25 exactly."""
    pid = "H002"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    clm_ds = ClaimDataset(records=[])

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    pp = result.per_policy.iloc[0]
    assert pp["healthy_etr"] == pytest.approx(_TOTAL_ETR, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Edge: multiple claims same policy
#
# Two separate claims with a healthy gap between them:
#   Claim 1: 2020-03-01 → 2020-04-30  (61 days)
#   Claim 2: 2021-03-01 → 2021-04-30  (61 days)
#   Gap:     2020-05-01 → 2021-02-28  (healthy)
#
# → sick_etr = 122 / 365.25
# → new_claims = 2
# → claim_terminations = 2
# → healthy_etr = (total - 122) / 365.25
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_claim_result():
    pid = "M001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, date(2020, 3, 1), date(2020, 4, 30)),
        _clm("C002", pid, date(2021, 3, 1), date(2021, 4, 30)),
    ]))
    return dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)


def test_multiple_claims_new_claims_count(two_claim_result):
    """Two separate claims → new_claims = 2."""
    total = two_claim_result.new_claims["new_claims"].sum()
    assert total == 2, f"Expected 2 new claims, got {total}"


def test_multiple_claims_terminations_count(two_claim_result):
    """Two closed claims → terminations = 2."""
    total = two_claim_result.claim_terminations["terminations"].sum()
    assert total == 2, f"Expected 2 terminations, got {total}"


def test_multiple_claims_sick_etr_exact(two_claim_result):
    """Sick ETR equals sum of both claim durations / 365.25."""
    # Claim 1: 2020-03-01 to 2020-04-30 = 61 days
    # Claim 2: 2021-03-01 to 2021-04-30 = 61 days
    c1_days = _days(date(2020, 3, 1), date(2020, 4, 30))
    c2_days = _days(date(2021, 3, 1), date(2021, 4, 30))
    expected_sick = (c1_days + c2_days) / 365.25

    actual_sick = two_claim_result.sick_etr["sick_etr"].sum()
    assert actual_sick == pytest.approx(expected_sick, rel=1e-9)


def test_multiple_claims_healthy_etr_is_complement(two_claim_result):
    """Healthy ETR = total ETR minus sick ETR (conservation)."""
    pp = two_claim_result.per_policy.iloc[0]
    assert pp["healthy_etr"] + pp["sick_etr"] == pytest.approx(pp["total_etr"], rel=1e-9)


def test_multiple_claims_healthy_captures_gap(two_claim_result):
    """Healthy ETR is strictly positive (gap between claims is captured)."""
    pp = two_claim_result.per_policy.iloc[0]
    assert pp["healthy_etr"] > 0.0, "Gap between claims should contribute healthy ETR"


# ---------------------------------------------------------------------------
# 5. Edge: open claim spans study end
#
# Claim starts 2022-06-01, no end date (OPEN).
# sick_etr should be counted from 2022-06-01 to study_end 2022-12-31.
# No termination should be counted.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_claim_result():
    pid = "O001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, date(2022, 6, 1)),   # OPEN — no end date
    ]))
    return dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)


def test_open_claim_sick_etr_to_study_end(open_claim_result):
    """Open claim: sick_etr = (study_end - claim_start + 1) / 365.25."""
    claim_start = date(2022, 6, 1)
    expected_sick = _etr(claim_start, _STUDY_END)
    actual_sick = open_claim_result.sick_etr["sick_etr"].sum()
    assert actual_sick == pytest.approx(expected_sick, rel=1e-9)


def test_open_claim_no_termination(open_claim_result):
    """Open claim: no termination events."""
    assert (
        open_claim_result.claim_terminations.empty
        or open_claim_result.claim_terminations["terminations"].sum() == 0
    ), "OPEN claim should produce no termination"


def test_open_claim_conservation(open_claim_result):
    """Conservation holds for a policy with an open claim."""
    pp = open_claim_result.per_policy.iloc[0]
    assert pp["healthy_etr"] + pp["sick_etr"] == pytest.approx(pp["total_etr"], rel=1e-9)


def test_open_claim_healthy_etr_is_preamble(open_claim_result):
    """Healthy ETR = period before claim start / 365.25."""
    expected_healthy = _etr(_STUDY_START, date(2022, 5, 31))
    actual_healthy = open_claim_result.per_policy.iloc[0]["healthy_etr"]
    assert actual_healthy == pytest.approx(expected_healthy, rel=1e-9)


# ---------------------------------------------------------------------------
# 6. Boundary: deferred_days = 0 → all claims counted immediately
# ---------------------------------------------------------------------------

def test_deferred_zero_counts_all_claims():
    """deferred_days=0: every claim within the study is counted as an incidence."""
    n_claims = 5
    pid = "Z001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))

    # 5 non-overlapping claims, one per year-quarter
    claim_dates = [
        (date(2020, 2, 1), date(2020, 3, 31)),
        (date(2020, 7, 1), date(2020, 8, 31)),
        (date(2021, 2, 1), date(2021, 3, 31)),
        (date(2021, 7, 1), date(2021, 8, 31)),
        (date(2022, 2, 1), date(2022, 3, 31)),
    ]
    clm_rows = [
        _clm(f"C{i+1:03d}", pid, s, e)
        for i, (s, e) in enumerate(claim_dates)
    ]
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame(clm_rows))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    assert result.new_claims["new_claims"].sum() == n_claims, (
        f"Expected {n_claims} new claims with deferred_days=0"
    )
    assert result.claim_terminations["terminations"].sum() == n_claims, (
        f"Expected {n_claims} terminations with deferred_days=0"
    )


def test_deferred_zero_sick_etr_matches_claim_durations():
    """deferred_days=0: sick_etr = sum of all claim durations / 365.25."""
    pid = "Z002"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))

    claim1_start, claim1_end = date(2020, 4, 1), date(2020, 6, 30)
    claim2_start, claim2_end = date(2021, 4, 1), date(2021, 6, 30)
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, claim1_start, claim1_end),
        _clm("C002", pid, claim2_start, claim2_end),
    ]))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    expected = (_days(claim1_start, claim1_end) + _days(claim2_start, claim2_end)) / 365.25
    assert result.sick_etr["sick_etr"].sum() == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------

def test_result_dataframe_columns(large_dataset):
    """DualETRResult DataFrames contain all required columns."""
    _, claim_ds = large_dataset
    policy_ds, _ = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=0)

    assert "healthy_etr" in result.healthy_etr.columns
    assert "sick_etr" in result.sick_etr.columns
    assert "new_claims" in result.new_claims.columns
    assert "terminations" in result.claim_terminations.columns
    assert set(result.per_policy.columns) >= {"policy_id", "healthy_etr", "sick_etr", "total_etr"}


def test_healthy_etr_nonneg(large_dataset):
    """healthy_etr is non-negative for all cells."""
    policy_ds, claim_ds = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=0)
    assert (result.healthy_etr["healthy_etr"] >= 0).all()


def test_sick_etr_nonneg(large_dataset):
    """sick_etr is non-negative for all cells."""
    policy_ds, claim_ds = large_dataset
    result = dual_etr(policy_ds, claim_ds, _STUDY, deferred_days=0)
    assert (result.sick_etr["sick_etr"] >= 0).all()


def test_claim_before_study_no_new_incidence():
    """Claim that starts AND ends before study_start → not counted as incidence."""
    pid = "B001"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    # Claim ends 2019-12-31, before study starts 2020-01-01
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, date(2019, 6, 1), date(2019, 12, 31)),
    ]))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    assert result.new_claims.empty or result.new_claims["new_claims"].sum() == 0
    # Policy was well before study, so no sick ETR carried over
    assert result.sick_etr.empty or result.sick_etr["sick_etr"].sum() == pytest.approx(
        0.0, abs=1e-12
    )


def test_pre_study_claim_carried_over_contributes_sick_etr():
    """Claim that starts before study but ends during study → sick ETR from study_start."""
    pid = "B002"
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame([_pol(pid)]))
    # Claim starts 2019-06-01, ends 2020-04-30 → sick from 2020-01-01 to 2020-04-30
    claim_end = date(2020, 4, 30)
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame([
        _clm("C001", pid, date(2019, 6, 1), claim_end),
    ]))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0)

    expected_sick = _etr(_STUDY_START, claim_end)
    actual_sick = result.sick_etr["sick_etr"].sum()
    assert actual_sick == pytest.approx(expected_sick, rel=1e-9)

    # No new incidence: claim_start + 0 = 2019-06-01 < study_start
    assert result.new_claims.empty or result.new_claims["new_claims"].sum() == 0


def test_group_by_gender_segments_correctly():
    """group_by=['gender'] produces separate M/F rows."""
    pol_rows = [
        _pol("M001"),
        {**_pol("F001"), "gender": "F"},
    ]
    pol_ds = PolicyDataset.from_dataframe(pd.DataFrame(pol_rows))

    # One claim for each policy
    clm_rows = [
        _clm("C001", "M001", date(2020, 6, 1), date(2020, 6, 30)),
        _clm("C002", "F001", date(2021, 6, 1), date(2021, 6, 30)),
    ]
    clm_ds = ClaimDataset.from_dataframe(pd.DataFrame(clm_rows))

    result = dual_etr(pol_ds, clm_ds, _STUDY, deferred_days=0, group_by=["gender"])

    h_df = result.healthy_etr
    s_df = result.sick_etr

    assert "gender" in h_df.columns, "group_by column 'gender' missing from healthy_etr"
    assert "gender" in s_df.columns, "group_by column 'gender' missing from sick_etr"
    assert set(h_df["gender"].unique()) == {"M", "F"}
