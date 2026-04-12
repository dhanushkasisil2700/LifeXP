"""Morbidity experience study: MorbidityStudy and MorbidityResults."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import gamma as _gamma
from scipy.stats import lognorm as _lognorm

from lifexp.core.data_model import ClaimDataset, PolicyDataset
from lifexp.core.date_utils import OutOfStudyError, age_at, days_in_study, policy_year_at
from lifexp.core.exposure import dual_etr, _sick_intervals_for_policy
from lifexp.core.study_period import AgeBasis, StudyPeriod
from lifexp.core.tables import MortalityTable, TableRegistry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _anniversary(base: date, year: int) -> date:
    try:
        return base.replace(year=year)
    except ValueError:
        return base.replace(year=year, day=28)


def _boundary_dates(anchor: date, start: date, end: date) -> List[date]:
    """Anniversary dates of *anchor* strictly within (start, end]."""
    result: List[date] = []
    for year in range(start.year - 1, end.year + 2):
        ann = _anniversary(anchor, year)
        if start < ann <= end:
            result.append(ann)
    return result


def _iter_sick_duration_segments(
    dob: date,
    claim_start: date,
    sick_start: date,
    sick_end: date,
    age_basis: AgeBasis,
):
    """Yield ``(age, duration_on_claim, days)`` splitting at birthday and
    claim-anniversary boundaries within ``[sick_start, sick_end]``.

    ``duration_on_claim`` = 1 during the first year of the claim, 2 during
    the second year, etc. (same convention as policy_year).
    """
    boundaries = sorted(
        {sick_start}
        | set(_boundary_dates(dob, sick_start, sick_end))
        | set(_boundary_dates(claim_start, sick_start, sick_end))
    )
    n = len(boundaries)
    for i, seg_start in enumerate(boundaries):
        is_last = i == n - 1
        seg_end = boundaries[i + 1] - timedelta(days=1) if not is_last else sick_end
        age = age_at(dob, seg_start, age_basis)
        duration = policy_year_at(claim_start, seg_start)
        days = (seg_end - seg_start).days + 1
        yield age, duration, days


def _build_termination_df(
    policy_dataset: PolicyDataset,
    claim_dataset: ClaimDataset,
    study: StudyPeriod,
    deferred_days: int,
    group_by: List[str],
    age_basis: AgeBasis,
) -> pd.DataFrame:
    """Sick ETR and termination counts segmented by (group, duration_on_claim, age)."""
    group_cols = group_by + ["duration_on_claim", "age"]
    sick_rows: List[dict] = []
    term_rows: List[dict] = []

    for record in policy_dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        group_vals = {f: getattr(record, f) for f in group_by}
        claims = claim_dataset.claims_for(record.policy_id)

        for claim in claims:
            eff_sick_start = claim.claim_start_date + timedelta(days=deferred_days)
            eff_sick_end = (
                claim.claim_end_date if claim.claim_end_date is not None else obs_end
            )
            sick_start = max(obs_start, eff_sick_start)
            sick_end = min(obs_end, eff_sick_end)

            if sick_start > sick_end:
                continue

            for age, duration, days in _iter_sick_duration_segments(
                record.date_of_birth, claim.claim_start_date,
                sick_start, sick_end, age_basis,
            ):
                sick_rows.append({
                    **group_vals,
                    "duration_on_claim": duration,
                    "age": age,
                    "_etr": days / 365.25,
                })

            # Termination: claim ends within the study AND past the deferred period
            if (claim.claim_end_date is not None
                    and obs_start <= claim.claim_end_date <= obs_end
                    and claim.claim_end_date >= eff_sick_start):
                t_dur = policy_year_at(claim.claim_start_date, claim.claim_end_date)
                t_age = age_at(record.date_of_birth, claim.claim_end_date, age_basis)
                is_death = (claim.claim_status == "CLOSED_DEATH")
                term_rows.append({
                    **group_vals,
                    "duration_on_claim": t_dur,
                    "age": t_age,
                    "_is_recovery": int(not is_death),
                    "_is_death": int(is_death),
                })

    if sick_rows:
        sick_agg = (
            pd.DataFrame(sick_rows)
            .groupby(group_cols, as_index=False)
            .agg(sick_etr=("_etr", "sum"))
        )
    else:
        sick_agg = pd.DataFrame(columns=group_cols + ["sick_etr"])

    if term_rows:
        term_agg = (
            pd.DataFrame(term_rows)
            .groupby(group_cols, as_index=False)
            .agg(
                recoveries=("_is_recovery", "sum"),
                deaths=("_is_death", "sum"),
            )
        )
    else:
        term_agg = pd.DataFrame(columns=group_cols + ["recoveries", "deaths"])

    detail = (
        sick_agg
        .merge(term_agg, on=group_cols, how="left")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    for col in ("recoveries", "deaths"):
        if col not in detail.columns:
            detail[col] = 0.0
        else:
            detail[col] = pd.to_numeric(detail[col], errors="coerce").fillna(0.0)

    detail["recovery_rate"] = np.where(
        detail["sick_etr"] > 0,
        detail["recoveries"] / detail["sick_etr"],
        np.nan,
    )
    detail["death_on_claim_rate"] = np.where(
        detail["sick_etr"] > 0,
        detail["deaths"] / detail["sick_etr"],
        np.nan,
    )
    return detail


def _build_severity_df(
    policy_dataset: PolicyDataset,
    claim_dataset: ClaimDataset,
    study: StudyPeriod,
    deferred_days: int,
    group_by: List[str],
) -> pd.DataFrame:
    """Summary statistics of incurred claim amounts, grouped by *group_by*."""
    amounts_by_group: dict = {}

    for record in policy_dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        gkey = tuple(getattr(record, f) for f in group_by)
        for claim in claim_dataset.claims_for(record.policy_id):
            incidence_date = claim.claim_start_date + timedelta(days=deferred_days)
            survives = (
                claim.claim_end_date is None
                or claim.claim_end_date >= incidence_date
            )
            if obs_start <= incidence_date <= obs_end and survives:
                amounts_by_group.setdefault(gkey, []).append(claim.claim_amount)

    rows = []
    for gkey, amounts in amounts_by_group.items():
        arr = np.array(amounts, dtype=float)
        row = {f: v for f, v in zip(group_by, gkey)}
        row["count"] = len(arr)
        row["mean"] = float(np.mean(arr))
        row["median"] = float(np.median(arr))
        row["p95"] = float(np.percentile(arr, 95))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=group_by + ["count", "mean", "median", "p95"])
    return pd.DataFrame(rows)


def _build_claim_costs_by_age(
    policy_dataset: PolicyDataset,
    claim_dataset: ClaimDataset,
    study: StudyPeriod,
    deferred_days: int,
    group_by: List[str],
    age_basis: AgeBasis,
) -> pd.DataFrame:
    """Total incurred claim amount per (group, age) cell."""
    group_cols = group_by + ["age"]
    rows: List[dict] = []
    amounts_all: List[float] = []

    for record in policy_dataset._records:
        try:
            obs_start, obs_end = days_in_study(record, study)
        except OutOfStudyError:
            continue

        group_vals = {f: getattr(record, f) for f in group_by}
        for claim in claim_dataset.claims_for(record.policy_id):
            incidence_date = claim.claim_start_date + timedelta(days=deferred_days)
            survives = (
                claim.claim_end_date is None
                or claim.claim_end_date >= incidence_date
            )
            if obs_start <= incidence_date <= obs_end and survives:
                a = age_at(record.date_of_birth, incidence_date, age_basis)
                rows.append({**group_vals, "age": a, "_cost": claim.claim_amount})
                amounts_all.append(claim.claim_amount)

    if not rows:
        return (
            pd.DataFrame(columns=group_cols + ["actual_cost"]),
            np.array([], dtype=float),
        )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(actual_cost=("_cost", "sum"))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return agg, np.array(amounts_all, dtype=float)


# ---------------------------------------------------------------------------
# MorbidityResults
# ---------------------------------------------------------------------------

class MorbidityResults:
    """Results produced by :meth:`MorbidityStudy.run`.

    Attributes
    ----------
    incidence_df : pd.DataFrame
        Columns: ``[*group_by, age, healthy_etr, new_claims, incidence_rate]``.
        One row per *(group, age)* cell.
    termination_df : pd.DataFrame
        Columns: ``[*group_by, duration_on_claim, age, sick_etr, recoveries,
        deaths, recovery_rate, death_on_claim_rate]``.
    severity_df : pd.DataFrame
        Columns: ``[*group_by, count, mean, median, p95]``.
        Summary statistics over incurred claim amounts.
    """

    def __init__(
        self,
        incidence_df: pd.DataFrame,
        termination_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        _ae_base_df: pd.DataFrame,     # (group, age, healthy_etr, sick_etr, new_claims)
        _claim_costs_df: pd.DataFrame,  # (group, age, actual_cost)
        _claim_amounts: np.ndarray,     # raw claim amounts for fitting
        _group_by: List[str],
    ) -> None:
        self.incidence_df = incidence_df
        self.termination_df = termination_df
        self.severity_df = severity_df
        self._ae_base_df = _ae_base_df
        self._claim_costs_df = _claim_costs_df
        self._claim_amounts = _claim_amounts
        self._group_by = list(_group_by)

    # ------------------------------------------------------------------
    # A/E on incidence
    # ------------------------------------------------------------------

    def ae_incidence(self, ri_table: MortalityTable) -> pd.DataFrame:
        """A/E ratio on incidence: actual new claims vs. ``ri_rate × healthy_etr``.

        Parameters
        ----------
        ri_table:
            Rate table providing expected incidence rates by age.

        Returns
        -------
        pd.DataFrame
            Extends :attr:`incidence_df` with ``ri_rate``, ``expected_claims``,
            ``ae_incidence``.
        """
        df = self.incidence_df.copy()
        df["ri_rate"] = df["age"].apply(lambda a: ri_table.qx(int(a)))
        df["expected_claims"] = df["ri_rate"] * df["healthy_etr"]
        df["ae_incidence"] = np.where(
            df["expected_claims"] > 0,
            df["new_claims"] / df["expected_claims"],
            np.nan,
        )
        return df

    # ------------------------------------------------------------------
    # A/E on cost
    # ------------------------------------------------------------------

    def ae_cost_ratio(
        self,
        ri_table: MortalityTable,
        mean_sa: Optional[float] = None,
        assumed_claim_duration: float = 1.0,
    ) -> pd.DataFrame:
        """A/E on claim cost.

        Two metrics are returned in the same DataFrame:

        **Simple cost ratio** — compares total incurred claim amounts against
        the expected cost assuming each claim costs one unit of *mean_sa*::

            ae_cost_simple = Σ(claim_amount) / (ri_rate × healthy_etr × mean_sa)

        **Duration-adjusted A/E** — for periodic (income-replacement) benefits,
        compares accumulated sick exposure to expected sick exposure under the
        assumed average claim duration::

            ae_duration_adjusted = sick_etr / (ri_rate × healthy_etr × assumed_claim_duration)

        When claims last longer than *assumed_claim_duration*,
        ``ae_duration_adjusted > ae_cost_simple``.

        Parameters
        ----------
        ri_table:
            Incidence rate table.
        mean_sa:
            Mean sum assured / benefit amount used to normalise the simple cost
            ratio.  Defaults to the mean of all incurred claim amounts.
        assumed_claim_duration:
            Assumed average claim duration in years for the duration-adjusted
            metric.  Default is 1.0 year.

        Returns
        -------
        pd.DataFrame
            Columns include ``ae_cost_simple`` and ``ae_duration_adjusted``.
        """
        if mean_sa is None:
            mean_sa = (
                float(np.mean(self._claim_amounts))
                if len(self._claim_amounts) > 0
                else 1.0
            )

        group_cols = self._group_by + ["age"]

        base = self._ae_base_df.copy()
        base["ri_rate"] = base["age"].apply(lambda a: ri_table.qx(int(a)))
        base["expected_claims"] = base["ri_rate"] * base["healthy_etr"]

        # Merge in actual costs
        if not self._claim_costs_df.empty:
            base = base.merge(self._claim_costs_df, on=group_cols, how="left")
            base["actual_cost"] = pd.to_numeric(
                base["actual_cost"], errors="coerce"
            ).fillna(0.0)
        else:
            base["actual_cost"] = 0.0

        # Simple cost ratio: actual_cost / (ri_rate × healthy_etr × mean_sa)
        base["expected_cost_simple"] = base["ri_rate"] * base["healthy_etr"] * mean_sa
        base["ae_cost_simple"] = np.where(
            base["expected_cost_simple"] > 0,
            base["actual_cost"] / base["expected_cost_simple"],
            np.nan,
        )

        # Duration-adjusted: sick_etr / (ri_rate × healthy_etr × assumed_duration)
        base["expected_sick_etr"] = (
            base["ri_rate"] * base["healthy_etr"] * assumed_claim_duration
        )
        base["ae_duration_adjusted"] = np.where(
            base["expected_sick_etr"] > 0,
            base["sick_etr"] / base["expected_sick_etr"],
            np.nan,
        )

        return base

    # ------------------------------------------------------------------
    # Severity fitting
    # ------------------------------------------------------------------

    def severity_fit(self, distribution: str = "lognormal") -> dict:
        """Fit a parametric distribution to incurred claim amounts.

        Parameters
        ----------
        distribution:
            ``'lognormal'`` (default) or ``'gamma'``.

        Returns
        -------
        dict
            Fitted parameters.  Keys depend on *distribution*:

            * lognormal: ``mu``, ``sigma``, ``loc``, ``scale``
            * gamma: ``shape``, ``loc``, ``scale``
        """
        amounts = self._claim_amounts
        if len(amounts) == 0:
            return {"distribution": distribution, "error": "no claims to fit"}

        if distribution == "lognormal":
            shape, loc, scale = _lognorm.fit(amounts, floc=0)
            return {
                "distribution": "lognormal",
                "mu": float(np.log(scale)),
                "sigma": float(shape),
                "loc": float(loc),
                "scale": float(scale),
            }
        if distribution == "gamma":
            a, loc, scale = _gamma.fit(amounts, floc=0)
            return {
                "distribution": "gamma",
                "shape": float(a),
                "loc": float(loc),
                "scale": float(scale),
            }
        raise ValueError(f"Unsupported distribution: {distribution!r}; use 'lognormal' or 'gamma'")


# ---------------------------------------------------------------------------
# MorbidityStudy
# ---------------------------------------------------------------------------

class MorbidityStudy:
    """Orchestrates a morbidity experience study.

    Parameters
    ----------
    policy_dataset:
        Collection of PolicyRecord objects.
    claim_dataset:
        Collection of ClaimRecord objects keyed by policy_id.
    study:
        Observation window.
    deferred_days:
        Waiting period length in calendar days (0 = no deferral).
    ri_table:
        Optional name of a reinsurance/standard incidence table in the
        :class:`~lifexp.core.tables.TableRegistry`.  Not used in
        :meth:`run`; pass a :class:`~lifexp.core.tables.MortalityTable`
        directly to the A/E methods instead.
    group_by:
        PolicyRecord field names to segment results by.
    age_basis:
        Age convention for birthday-boundary segmentation.
    """

    def __init__(
        self,
        policy_dataset: PolicyDataset,
        claim_dataset: ClaimDataset,
        study: StudyPeriod,
        deferred_days: int = 0,
        ri_table: Optional[str] = None,
        group_by: Optional[List[str]] = None,
        age_basis: AgeBasis = AgeBasis.LAST_BIRTHDAY,
    ) -> None:
        self._policy_dataset = policy_dataset
        self._claim_dataset = claim_dataset
        self._study = study
        self._deferred_days = deferred_days
        self._ri_table_name = ri_table
        self._group_by = list(group_by) if group_by is not None else []
        self._age_basis = age_basis

    # ------------------------------------------------------------------

    def run(self) -> MorbidityResults:
        """Execute the study and return :class:`MorbidityResults`.

        Steps
        -----
        1. Call :func:`~lifexp.core.exposure.dual_etr` to partition each
           policy's observation window into healthy and sick ETR.
        2. Aggregate healthy_etr, sick_etr, and new_claims to *(group, age)*
           level for the A/E base DataFrame.
        3. Build :attr:`~MorbidityResults.termination_df` with duration
           segmentation via :func:`_build_termination_df`.
        4. Build :attr:`~MorbidityResults.severity_df` and collect raw claim
           amounts for :meth:`~MorbidityResults.severity_fit`.
        """
        group_by = self._group_by
        age_basis = self._age_basis
        group_cols_py = group_by + ["age", "policy_year"]
        group_cols_age = group_by + ["age"]

        # ------------------------------------------------------------------
        # Step 1: dual ETR
        # ------------------------------------------------------------------
        etr = dual_etr(
            self._policy_dataset,
            self._claim_dataset,
            self._study,
            deferred_days=self._deferred_days,
            age_basis=age_basis,
            group_by=group_by,
        )

        # ------------------------------------------------------------------
        # Step 2: Aggregate to (group, age) for A/E base
        # ------------------------------------------------------------------
        def _agg_to_age(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=group_cols_age + [col])
            return (
                df.groupby(group_cols_age, as_index=False)
                .agg(**{col: (col, "sum")})
                .sort_values(group_cols_age)
                .reset_index(drop=True)
            )

        healthy_by_age = _agg_to_age(etr.healthy_etr, "healthy_etr")
        sick_by_age = _agg_to_age(etr.sick_etr, "sick_etr")

        # new_claims may use a different column name internally
        if not etr.new_claims.empty and "new_claims" in etr.new_claims.columns:
            claims_by_age = (
                etr.new_claims
                .groupby(group_cols_age, as_index=False)
                .agg(new_claims=("new_claims", "sum"))
                .sort_values(group_cols_age)
                .reset_index(drop=True)
            )
        else:
            claims_by_age = pd.DataFrame(columns=group_cols_age + ["new_claims"])

        # Merge to build incidence_df
        incidence_base = (
            healthy_by_age
            .merge(sick_by_age, on=group_cols_age, how="left")
            .merge(claims_by_age, on=group_cols_age, how="left")
        )
        incidence_base["sick_etr"] = pd.to_numeric(
            incidence_base.get("sick_etr"), errors="coerce"
        ).fillna(0.0)
        incidence_base["new_claims"] = pd.to_numeric(
            incidence_base.get("new_claims"), errors="coerce"
        ).fillna(0.0)

        incidence_df = incidence_base[group_cols_age + ["healthy_etr", "new_claims"]].copy()
        incidence_df["incidence_rate"] = np.where(
            incidence_df["healthy_etr"] > 0,
            incidence_df["new_claims"] / incidence_df["healthy_etr"],
            np.nan,
        )

        ae_base_df = incidence_base[group_cols_age + ["healthy_etr", "sick_etr", "new_claims"]].copy()

        # ------------------------------------------------------------------
        # Step 3: Termination DataFrame (duration-segmented sick ETR)
        # ------------------------------------------------------------------
        termination_df = _build_termination_df(
            self._policy_dataset,
            self._claim_dataset,
            self._study,
            self._deferred_days,
            group_by,
            age_basis,
        )

        # ------------------------------------------------------------------
        # Step 4: Severity DataFrame + raw claim amounts
        # ------------------------------------------------------------------
        severity_df = _build_severity_df(
            self._policy_dataset,
            self._claim_dataset,
            self._study,
            self._deferred_days,
            group_by,
        )

        claim_costs_df, claim_amounts = _build_claim_costs_by_age(
            self._policy_dataset,
            self._claim_dataset,
            self._study,
            self._deferred_days,
            group_by,
            age_basis,
        )

        return MorbidityResults(
            incidence_df=incidence_df,
            termination_df=termination_df,
            severity_df=severity_df,
            _ae_base_df=ae_base_df,
            _claim_costs_df=claim_costs_df,
            _claim_amounts=claim_amounts,
            _group_by=group_by,
        )
