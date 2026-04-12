"""Central and initial exposed-to-risk (ETR) calculation for life experience studies."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from lifexp.core.data_model import ClaimDataset, PolicyDataset
from lifexp.core.date_utils import OutOfStudyError, age_at, days_in_study, policy_year_at
from lifexp.core.study_period import AgeBasis, StudyPeriod


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _anniversary(dob: date, year: int) -> date:
    """Anniversary of *dob* in *year*; Feb-29 in non-leap year → Feb-28."""
    try:
        return dob.replace(year=year)
    except ValueError:
        return dob.replace(year=year, day=28)


def _boundary_dates(anchor: date, obs_start: date, obs_end: date) -> List[date]:
    """Anniversary dates of *anchor* that fall strictly within (obs_start, obs_end].

    These are the dates on which the age or policy-year label changes inside the
    observation window and therefore define new time segments.
    """
    result: List[date] = []
    for year in range(obs_start.year - 1, obs_end.year + 2):
        ann = _anniversary(anchor, year)
        if obs_start < ann <= obs_end:
            result.append(ann)
    return result


def _iter_segments(
    dob: date,
    issue_date: date,
    obs_start: date,
    obs_end: date,
    age_basis: AgeBasis,
):
    """Yield ``(age, policy_year, days, is_last)`` for every time segment.

    The observation window [obs_start, obs_end] (inclusive both ends) is split
    at every birthday and policy-anniversary date that falls strictly inside it.
    Each resulting sub-interval has a constant age and policy year.

    *days* counts calendar days inclusive of both endpoints, so a single-day
    segment contributes days=1 and a 365-day year contributes days=365.

    *is_last* is True only for the final segment — the one that ends at obs_end.
    It is used by :func:`initial_etr` to apply the death correction to the
    segment that actually contains the exit event.
    """
    starts = sorted(
        {obs_start}
        | set(_boundary_dates(dob, obs_start, obs_end))
        | set(_boundary_dates(issue_date, obs_start, obs_end))
    )
    n = len(starts)

    for i, seg_start in enumerate(starts):
        is_last = i == n - 1
        seg_end = starts[i + 1] - timedelta(days=1) if not is_last else obs_end
        age = age_at(dob, seg_start, age_basis)
        py = policy_year_at(issue_date, seg_start)
        days = (seg_end - seg_start).days + 1  # inclusive count
        yield age, py, days, is_last


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def central_etr(
    dataset: PolicyDataset,
    study: StudyPeriod,
    age_basis: AgeBasis = AgeBasis.LAST_BIRTHDAY,
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute central exposed-to-risk segmented by age, policy year, and optional fields.

    Parameters
    ----------
    dataset:
        Collection of policy records.
    study:
        Observation window.
    age_basis:
        How to assign an integer age to each time segment.
    group_by:
        PolicyRecord field names to use as additional segmentation dimensions
        (e.g. ``['gender', 'smoker_status']``).  Pass ``[]`` for aggregate only.

    Returns
    -------
    pd.DataFrame
        Columns: ``[*group_by, age, policy_year, central_etr, policy_count]``.
        Policies with no exposure in the study window are silently skipped.
        All exit types (death, lapse, surrender …) are treated as right-censoring:
        they contribute fractional ETR up to their exit date, the same as in-force
        policies.
    """
    if group_by is None:
        group_by = []

    rows: List[dict] = []

    for record in dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        for age, py, days, _ in _iter_segments(
            record.date_of_birth,
            record.issue_date,
            obs_start,
            obs_end,
            age_basis,
        ):
            row: dict = {f: getattr(record, f) for f in group_by}
            row["_policy_id"] = record.policy_id
            row["age"] = age
            row["policy_year"] = py
            row["_etr"] = days / 365.25
            rows.append(row)

    group_cols = group_by + ["age", "policy_year"]

    if not rows:
        return pd.DataFrame(columns=group_cols + ["central_etr", "policy_count"])

    df = pd.DataFrame(rows)
    result = (
        df.groupby(group_cols, as_index=False)
        .agg(
            central_etr=pd.NamedAgg(column="_etr", aggfunc="sum"),
            policy_count=pd.NamedAgg(column="_policy_id", aggfunc="nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return result


def initial_etr(
    dataset: PolicyDataset,
    study: StudyPeriod,
    age_basis: AgeBasis = AgeBasis.LAST_BIRTHDAY,
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute initial exposed-to-risk with the actuarial death correction.

    The initial ETR adjusts for the fact that deaths terminate exposure before
    the end of the age cell.  Each death contributes exactly **1.0** to the age
    cell it occurs in, regardless of how many days elapsed.  Non-death exits
    (lapse, surrender, maturity) are treated identically to central ETR —
    fractional exposure only.

    The relationship to central ETR is::

        initial_etr = central_etr + Σ (1.0 − days/365.25) for each death segment

    Parameters
    ----------
    dataset, study, age_basis, group_by:
        Same as :func:`central_etr`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[*group_by, age, policy_year, initial_etr, deaths, policy_count]``.
        *deaths* is the count of death exits whose last segment falls in that cell.
    """
    if group_by is None:
        group_by = []

    rows: List[dict] = []

    for record in dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        is_death_record = record.status == "DEATH"

        for age, py, days, is_last in _iter_segments(
            record.date_of_birth,
            record.issue_date,
            obs_start,
            obs_end,
            age_basis,
        ):
            # The death correction applies only to the final segment of a death record.
            # All earlier segments (crossing age or policy-year boundaries) contribute
            # fractional ETR exactly like central ETR.
            is_death_seg = is_last and is_death_record
            etr_contrib = days / 365.25
            initial_contrib = 1.0 if is_death_seg else etr_contrib

            row: dict = {f: getattr(record, f) for f in group_by}
            row["_policy_id"] = record.policy_id
            row["age"] = age
            row["policy_year"] = py
            row["_initial_etr"] = initial_contrib
            row["_is_death"] = int(is_death_seg)
            rows.append(row)

    group_cols = group_by + ["age", "policy_year"]

    if not rows:
        return pd.DataFrame(
            columns=group_cols + ["initial_etr", "deaths", "policy_count"]
        )

    df = pd.DataFrame(rows)
    result = (
        df.groupby(group_cols, as_index=False)
        .agg(
            initial_etr=pd.NamedAgg(column="_initial_etr", aggfunc="sum"),
            deaths=pd.NamedAgg(column="_is_death", aggfunc="sum"),
            policy_count=pd.NamedAgg(column="_policy_id", aggfunc="nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return result


def etr_comparison(
    central_df: pd.DataFrame,
    initial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side comparison of central and initial ETR collapsed to age.

    Parameters
    ----------
    central_df:
        Output of :func:`central_etr`.
    initial_df:
        Output of :func:`initial_etr`.

    Returns
    -------
    pd.DataFrame
        Columns: ``age, deaths, central_etr, initial_etr, difference``
        where *difference* = initial_etr − central_etr (always ≥ 0).
    """
    c = (
        central_df.groupby("age", as_index=False)
        .agg(central_etr=("central_etr", "sum"))
    )
    i = (
        initial_df.groupby("age", as_index=False)
        .agg(initial_etr=("initial_etr", "sum"), deaths=("deaths", "sum"))
    )
    merged = (
        c.merge(i, on="age", how="outer")
        .fillna(0)
        .sort_values("age")
        .reset_index(drop=True)
    )
    merged["difference"] = merged["initial_etr"] - merged["central_etr"]
    return merged[["age", "deaths", "central_etr", "initial_etr", "difference"]]


def etr_summary(etr_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse all dimensions except age; return total ETR and policy count by age.

    Parameters
    ----------
    etr_df:
        Output of :func:`central_etr`.
    """
    return (
        etr_df.groupby("age", as_index=False)
        .agg(
            central_etr=("central_etr", "sum"),
            policy_count=("policy_count", "sum"),
        )
        .sort_values("age")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Dual ETR for morbidity experience studies
# ---------------------------------------------------------------------------

class DualETRResult:
    """Result of :func:`dual_etr` for a morbidity experience study.

    Attributes
    ----------
    healthy_etr : pd.DataFrame
        Central exposure for the **healthy** state — the denominator for
        new-claim incidence rates.
        Columns: ``[*group_by, age, policy_year, healthy_etr, policy_count]``.
    sick_etr : pd.DataFrame
        Central exposure for the **sick** state — the denominator for claim
        termination rates.
        Columns: ``[*group_by, age, policy_year, sick_etr, policy_count]``.
    new_claims : pd.DataFrame
        Count of new claim incidences (i.e. claims that survived the deferred
        period and whose incidence date falls within the study window).
        Columns: ``[*group_by, age, policy_year, new_claims]``.
    claim_terminations : pd.DataFrame
        Count of claim terminations (recoveries + deaths from the sick state)
        occurring within the study window.
        Columns: ``[*group_by, age, policy_year, terminations]``.
    per_policy : pd.DataFrame
        Per-policy summary for diagnostic use.  Verifying
        ``healthy_etr + sick_etr ≈ total_etr`` for every row confirms the
        conservation invariant.
        Columns: ``[policy_id, healthy_etr, sick_etr, total_etr]``.
    """

    def __init__(
        self,
        healthy_etr: pd.DataFrame,
        sick_etr: pd.DataFrame,
        new_claims: pd.DataFrame,
        claim_terminations: pd.DataFrame,
        per_policy: pd.DataFrame,
    ) -> None:
        self.healthy_etr = healthy_etr
        self.sick_etr = sick_etr
        self.new_claims = new_claims
        self.claim_terminations = claim_terminations
        self.per_policy = per_policy


def _sick_intervals_for_policy(
    claims: list,
    obs_start: date,
    obs_end: date,
    deferred_days: int,
) -> List[Tuple[date, date]]:
    """Sorted, non-overlapping sick intervals for one policy.

    A sick interval begins at ``claim_start + deferred_days`` (clamped to
    ``obs_start``) and ends at ``claim_end`` (or ``obs_end`` for open claims).
    Overlapping or adjacent intervals are merged so the result is a minimal
    cover of the sick time within ``[obs_start, obs_end]``.
    """
    raw: List[Tuple[date, date]] = []

    for claim in claims:
        eff_start = claim.claim_start_date + timedelta(days=deferred_days)
        eff_end = (
            claim.claim_end_date
            if claim.claim_end_date is not None
            else obs_end
        )
        sick_start = max(obs_start, eff_start)
        sick_end = min(obs_end, eff_end)
        if sick_start <= sick_end:
            raw.append((sick_start, sick_end))

    if not raw:
        return []

    raw.sort()
    merged: List[Tuple[date, date]] = [raw[0]]
    for iv_s, iv_e in raw[1:]:
        prev_s, prev_e = merged[-1]
        if iv_s <= prev_e + timedelta(days=1):
            merged[-1] = (prev_s, max(prev_e, iv_e))
        else:
            merged.append((iv_s, iv_e))
    return merged


def _healthy_intervals(
    sick_ivs: List[Tuple[date, date]],
    obs_start: date,
    obs_end: date,
) -> List[Tuple[date, date]]:
    """Complement of *sick_ivs* within ``[obs_start, obs_end]``."""
    result: List[Tuple[date, date]] = []
    current = obs_start
    for iv_s, iv_e in sick_ivs:
        if current < iv_s:
            result.append((current, iv_s - timedelta(days=1)))
        current = iv_e + timedelta(days=1)
    if current <= obs_end:
        result.append((current, obs_end))
    return result


def dual_etr(
    policy_dataset: PolicyDataset,
    claim_dataset: ClaimDataset,
    study: StudyPeriod,
    deferred_days: int = 0,
    age_basis: AgeBasis = AgeBasis.LAST_BIRTHDAY,
    group_by: Optional[List[str]] = None,
) -> DualETRResult:
    """Compute dual exposed-to-risk for a morbidity experience study.

    Splits each policy's observation window into:

    * **Healthy** periods — while the policy is NOT on claim.  These are the
      denominator for new-claim incidence rates.
    * **Sick** periods — while the policy IS on claim and the deferred period
      has elapsed.  These are the denominator for claim termination rates.

    **Conservation invariant**: for every policy, ``healthy_etr + sick_etr``
    equals the policy's total central exposure for the study window, regardless
    of *deferred_days*.

    Parameters
    ----------
    policy_dataset:
        Collection of PolicyRecord objects.
    claim_dataset:
        Collection of ClaimRecord objects keyed by policy_id.
    study:
        Observation window.
    deferred_days:
        Waiting period length in calendar days.  A new claim is counted as
        an incidence only if the policy remains on claim for at least this
        many days.  ``0`` means all claims are counted immediately.
    age_basis:
        Age convention used when segmenting exposure at birthday boundaries.
    group_by:
        Additional PolicyRecord field names to segment results by
        (e.g. ``['gender', 'smoker_status']``).

    Returns
    -------
    DualETRResult
    """
    if group_by is None:
        group_by = []

    group_cols = group_by + ["age", "policy_year"]

    healthy_rows: List[dict] = []
    sick_rows: List[dict] = []
    new_claim_rows: List[dict] = []
    term_rows: List[dict] = []
    per_policy_rows: List[dict] = []

    for record in policy_dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        group_vals = {f: getattr(record, f) for f in group_by}
        claims = claim_dataset.claims_for(record.policy_id)

        # Partition [obs_start, obs_end] into sick and healthy intervals.
        # The deferred period is counted as healthy time, so the partition
        # is exhaustive and the conservation invariant holds by construction.
        sick_ivs = _sick_intervals_for_policy(claims, obs_start, obs_end, deferred_days)
        healthy_ivs = _healthy_intervals(sick_ivs, obs_start, obs_end)

        policy_healthy_etr = 0.0
        policy_sick_etr = 0.0

        for h_start, h_end in healthy_ivs:
            for age, py, days, _ in _iter_segments(
                record.date_of_birth, record.issue_date,
                h_start, h_end, age_basis,
            ):
                etr = days / 365.25
                policy_healthy_etr += etr
                healthy_rows.append({
                    **group_vals,
                    "_policy_id": record.policy_id,
                    "age": age,
                    "policy_year": py,
                    "_etr": etr,
                })

        for s_start, s_end in sick_ivs:
            for age, py, days, _ in _iter_segments(
                record.date_of_birth, record.issue_date,
                s_start, s_end, age_basis,
            ):
                etr = days / 365.25
                policy_sick_etr += etr
                sick_rows.append({
                    **group_vals,
                    "_policy_id": record.policy_id,
                    "age": age,
                    "policy_year": py,
                    "_etr": etr,
                })

        total_days = (obs_end - obs_start).days + 1
        per_policy_rows.append({
            "policy_id": record.policy_id,
            "healthy_etr": policy_healthy_etr,
            "sick_etr": policy_sick_etr,
            "total_etr": total_days / 365.25,
        })

        # ---------------------------------------------------------------
        # New claim incidences
        # ---------------------------------------------------------------
        for claim in claims:
            incidence_date = claim.claim_start_date + timedelta(days=deferred_days)
            # Claim survives deferred period if it hasn't ended before the
            # deferred period completes.
            survives = (
                claim.claim_end_date is None
                or claim.claim_end_date >= incidence_date
            )
            if obs_start <= incidence_date <= obs_end and survives:
                age = age_at(record.date_of_birth, incidence_date, age_basis)
                py = policy_year_at(record.issue_date, incidence_date)
                new_claim_rows.append({
                    **group_vals,
                    "age": age,
                    "policy_year": py,
                    "_claim_id": claim.claim_id,
                })

        # ---------------------------------------------------------------
        # Claim terminations (recoveries and deaths from sick state)
        # ---------------------------------------------------------------
        for claim in claims:
            if claim.claim_end_date is None:
                continue
            eff_sick_start = claim.claim_start_date + timedelta(days=deferred_days)
            term_date = claim.claim_end_date
            # Count only terminations that occur at or after the sick period
            # starts (i.e., after the deferred period ends).
            if (obs_start <= term_date <= obs_end
                    and term_date >= eff_sick_start):
                age = age_at(record.date_of_birth, term_date, age_basis)
                py = policy_year_at(record.issue_date, term_date)
                term_rows.append({
                    **group_vals,
                    "age": age,
                    "policy_year": py,
                    "_claim_id": claim.claim_id,
                })

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------

    def _agg_etr(rows: List[dict], col: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=group_cols + [col, "policy_count"])
        df = pd.DataFrame(rows)
        return (
            df.groupby(group_cols, as_index=False)
            .agg(**{col: ("_etr", "sum"), "policy_count": ("_policy_id", "nunique")})
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

    def _agg_events(rows: List[dict], col: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=group_cols + [col])
        df = pd.DataFrame(rows)
        return (
            df.groupby(group_cols, as_index=False)
            .agg(**{col: ("_claim_id", "count")})
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

    per_policy_df = (
        pd.DataFrame(per_policy_rows)
        if per_policy_rows
        else pd.DataFrame(
            columns=["policy_id", "healthy_etr", "sick_etr", "total_etr"]
        )
    )

    return DualETRResult(
        healthy_etr=_agg_etr(healthy_rows, "healthy_etr"),
        sick_etr=_agg_etr(sick_rows, "sick_etr"),
        new_claims=_agg_events(new_claim_rows, "new_claims"),
        claim_terminations=_agg_events(term_rows, "terminations"),
        per_policy=per_policy_df,
    )
