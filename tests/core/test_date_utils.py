"""Tests for lifexp.core.date_utils and lifexp.core.study_period."""

from datetime import date, timedelta

import pytest

from lifexp.core.data_model import PolicyRecord
from lifexp.core.date_utils import OutOfStudyError, age_at, days_in_study, policy_year_at
from lifexp.core.study_period import AgeBasis, StudyPeriod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STUDY = StudyPeriod(start_date=date(2020, 1, 1), end_date=date(2022, 12, 31))


def make_record(**overrides) -> PolicyRecord:
    defaults = dict(
        policy_id="P001",
        date_of_birth=date(1980, 6, 15),
        issue_date=date(2018, 1, 1),
        gender="M",
        smoker_status="NS",
        sum_assured=100_000.0,
        annual_premium=1_200.0,
        product_code="TERM10",
        channel="AGENT",
        status="IF",
        exit_date=None,
        exit_reason=None,
    )
    defaults.update(overrides)
    return PolicyRecord(**defaults)


# ---------------------------------------------------------------------------
# Normal: age nearest birthday — boundary flip at the birthday
# ---------------------------------------------------------------------------

def test_age_nearest_birthday_boundary():
    dob = date(1980, 7, 15)

    # One day BEFORE the 43rd birthday → still 42
    ref_before = date(2023, 7, 14)
    assert age_at(dob, ref_before, AgeBasis.NEAREST_BIRTHDAY) == 42

    # One day AFTER the 43rd birthday → 43
    ref_after = date(2023, 7, 16)
    assert age_at(dob, ref_after, AgeBasis.NEAREST_BIRTHDAY) == 43


# ---------------------------------------------------------------------------
# Normal: policy year — 1-indexed with exact anniversary boundary
# ---------------------------------------------------------------------------

def test_policy_year_boundary():
    issue = date(2020, 3, 1)

    # Day before first anniversary → still policy year 1
    assert policy_year_at(issue, date(2021, 2, 28)) == 1

    # Exactly on first anniversary → policy year 2
    assert policy_year_at(issue, date(2021, 3, 1)) == 2


# ---------------------------------------------------------------------------
# Normal: in-force policy spanning the whole study period
# ---------------------------------------------------------------------------

def test_inforce_policy_full_study():
    # Issued before study, never exits — clipped to study window
    record = make_record(issue_date=date(2015, 6, 1), status="IF", exit_date=None)
    obs_start, obs_end = days_in_study(record, STUDY)
    assert obs_start == STUDY.start_date
    assert obs_end == STUDY.end_date


# ---------------------------------------------------------------------------
# Edge: policy issued after study ends
# ---------------------------------------------------------------------------

def test_policy_issued_after_study_end():
    record = make_record(
        issue_date=date(2023, 1, 1),
        status="IF",
        exit_date=None,
    )
    with pytest.raises(OutOfStudyError):
        days_in_study(record, STUDY)


# ---------------------------------------------------------------------------
# Edge: policy exited before study starts
# ---------------------------------------------------------------------------

def test_policy_exited_before_study_start():
    record = make_record(
        issue_date=date(2010, 1, 1),
        status="DEATH",
        exit_date=date(2019, 12, 31),
    )
    with pytest.raises(OutOfStudyError):
        days_in_study(record, STUDY)


# ---------------------------------------------------------------------------
# Boundary: Feb-29 DOB in a non-leap year
# ---------------------------------------------------------------------------

def test_feb29_dob_non_leap_year():
    dob = date(1980, 2, 29)  # leap year DOB

    # 2023 is not a leap year — must not crash
    ref = date(2023, 2, 28)
    result = age_at(dob, ref, AgeBasis.LAST_BIRTHDAY)
    # On Feb-28 2023, anniversary is treated as Feb-28 → birthday has just arrived
    # 2023 - 1980 = 43
    assert result == 43

    # Also works for NEAREST and NEXT
    assert age_at(dob, ref, AgeBasis.NEAREST_BIRTHDAY) == 43
    assert age_at(dob, ref, AgeBasis.NEXT_BIRTHDAY) == 43


# ---------------------------------------------------------------------------
# Boundary: policy exits exactly on study end date (inclusive)
# ---------------------------------------------------------------------------

def test_policy_exits_on_study_end():
    record = make_record(
        issue_date=date(2015, 1, 1),
        status="DEATH",
        exit_date=STUDY.end_date,
    )
    obs_start, obs_end = days_in_study(record, STUDY)
    assert obs_end == STUDY.end_date


# ---------------------------------------------------------------------------
# Additional: StudyPeriod rejects invalid dates
# ---------------------------------------------------------------------------

def test_study_period_invalid():
    with pytest.raises(ValueError):
        StudyPeriod(start_date=date(2022, 1, 1), end_date=date(2021, 1, 1))

    with pytest.raises(ValueError):
        StudyPeriod(start_date=date(2022, 1, 1), end_date=date(2022, 1, 1))


# ---------------------------------------------------------------------------
# Additional: policy year on issue date itself is PY=1
# ---------------------------------------------------------------------------

def test_policy_year_on_issue():
    issue = date(2020, 3, 1)
    assert policy_year_at(issue, issue) == 1


# ---------------------------------------------------------------------------
# Additional: LAST vs NEXT birthday basis
# ---------------------------------------------------------------------------

def test_age_last_and_next_birthday():
    dob = date(1980, 7, 15)
    ref = date(2023, 7, 14)  # one day before 43rd birthday

    assert age_at(dob, ref, AgeBasis.LAST_BIRTHDAY) == 42
    assert age_at(dob, ref, AgeBasis.NEXT_BIRTHDAY) == 43

    # Exactly on birthday
    ref_on = date(2023, 7, 15)
    assert age_at(dob, ref_on, AgeBasis.LAST_BIRTHDAY) == 43
    assert age_at(dob, ref_on, AgeBasis.NEXT_BIRTHDAY) == 43  # on birthday: no increment
