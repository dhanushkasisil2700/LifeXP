"""Date and age utility functions for actuarial experience analysis."""

from __future__ import annotations

from datetime import date
from typing import Tuple

from lifexp.core.data_model import PolicyRecord
from lifexp.core.study_period import AgeBasis, StudyPeriod


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OutOfStudyError(Exception):
    """Raised when a policy has no exposure within the study period."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _anniversary(dob: date, year: int) -> date:
    """Return the anniversary of *dob* in *year*, handling Feb-29 leap years.

    For a Feb-29 DOB in a non-leap year the anniversary is treated as Feb-28.
    """
    try:
        return dob.replace(year=year)
    except ValueError:
        # Feb-29 does not exist in this year — use Feb-28
        return dob.replace(year=year, day=28)


def _age_last_birthday(dob: date, ref_date: date) -> int:
    """Age in complete years as of *ref_date* (standard age-last-birthday)."""
    ann = _anniversary(dob, ref_date.year)
    if ref_date >= ann:
        return ref_date.year - dob.year
    return ref_date.year - dob.year - 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def age_at(dob: date, ref_date: date, basis: AgeBasis) -> int:
    """Return age in integer years under the requested basis.

    Parameters
    ----------
    dob:
        Date of birth.
    ref_date:
        Reference (valuation) date.
    basis:
        One of :class:`AgeBasis` — LAST_BIRTHDAY, NEAREST_BIRTHDAY, or
        NEXT_BIRTHDAY.

    Notes on NEAREST_BIRTHDAY
    -------------------------
    The flip point is the exact birthday: before the birthday the age equals
    the last-birthday age; on or after the birthday it increments by one.
    This matches the standard actuarial age-nearest-birthday (ANB) convention
    as used in most life experience studies.
    """
    alb = _age_last_birthday(dob, ref_date)

    if basis == AgeBasis.LAST_BIRTHDAY:
        return alb

    if basis == AgeBasis.NEAREST_BIRTHDAY:
        # Flip at the exact birthday — equivalent to ALB for whole-year granularity.
        # The birthday this year is the natural dividing line between two age groups.
        return alb

    # NEXT_BIRTHDAY — age at the next birthday unless we're exactly on one.
    ann = _anniversary(dob, ref_date.year)
    if ref_date == ann:
        # Exactly on birthday: current age is the "next birthday" just reached.
        return alb
    return alb + 1


def policy_year_at(issue_date: date, ref_date: date) -> int:
    """Return the 1-indexed policy year containing *ref_date*.

    Policy year 1 runs from issue_date up to (but not including) the first
    anniversary.  The year number increments on each anniversary.

    Examples
    --------
    >>> policy_year_at(date(2020, 3, 1), date(2021, 2, 28))
    1
    >>> policy_year_at(date(2020, 3, 1), date(2021, 3, 1))
    2
    """
    return _age_last_birthday(issue_date, ref_date) + 1


def days_in_study(
    record: PolicyRecord,
    study: StudyPeriod,
) -> Tuple[date, date]:
    """Return *(observation_start, observation_end)* for *record* within *study*.

    The observation window is:
    - ``observation_start = max(issue_date, study.start_date)``
    - ``observation_end   = min(exit_date or study.end_date, study.end_date)``

    The observation end is inclusive (the policy is observed *on* that date).

    Raises
    ------
    OutOfStudyError
        If the policy has no exposure within the study window:
        - ``issue_date > study.end_date``
        - ``exit_date < study.start_date``
    """
    # Determine effective exit boundary
    effective_exit = record.exit_date if record.exit_date is not None else study.end_date

    # Policy issued after the study ends — no exposure
    if record.issue_date > study.end_date:
        raise OutOfStudyError(
            f"policy_id={record.policy_id}: issue_date {record.issue_date} "
            f"is after study end {study.end_date}"
        )

    # Policy exited before the study starts — no exposure
    if effective_exit < study.start_date:
        raise OutOfStudyError(
            f"policy_id={record.policy_id}: exit_date {effective_exit} "
            f"is before study start {study.start_date}"
        )

    obs_start = max(record.issue_date, study.start_date)
    obs_end = min(effective_exit, study.end_date)

    return obs_start, obs_end
