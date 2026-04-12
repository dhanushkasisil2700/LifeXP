"""Lapse experience study: LapseStudy and LapseResults."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from lifexp.core.data_model import PolicyDataset
from lifexp.core.date_utils import OutOfStudyError, days_in_study, policy_year_at
from lifexp.core.study_period import StudyPeriod


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _anniversary(base_date: date, year: int) -> date:
    """Return the anniversary of *base_date* in *year*, handling Feb-29."""
    try:
        return base_date.replace(year=year)
    except ValueError:
        return base_date.replace(year=year, day=28)


def _py_boundaries(issue_date: date, obs_start: date, obs_end: date) -> List[date]:
    """Policy-year anniversary dates that fall strictly within (obs_start, obs_end]."""
    result: List[date] = []
    years_span = obs_end.year - issue_date.year + 2
    for offset in range(1, years_span + 1):
        ann = _anniversary(issue_date, issue_date.year + offset)
        if obs_start < ann <= obs_end:
            result.append(ann)
    return result


def _iter_py_segments(issue_date: date, obs_start: date, obs_end: date):
    """Yield ``(policy_year, days)`` for each policy-year segment in the window.

    The observation window [obs_start, obs_end] is split at every
    policy-anniversary that falls strictly inside it.  Each resulting
    sub-interval has a constant policy year.  ``days`` counts calendar days
    inclusive of both endpoints.
    """
    starts = sorted(
        {obs_start} | set(_py_boundaries(issue_date, obs_start, obs_end))
    )
    n = len(starts)
    for i, seg_start in enumerate(starts):
        is_last = i == n - 1
        seg_end = starts[i + 1] - timedelta(days=1) if not is_last else obs_end
        py = policy_year_at(issue_date, seg_start)
        days = (seg_end - seg_start).days + 1
        yield py, days


# ---------------------------------------------------------------------------
# LapseResults
# ---------------------------------------------------------------------------

class LapseResults:
    """Results produced by :meth:`LapseStudy.run`.

    Attributes
    ----------
    summary_df:
        One row per *(policy_year, group_by...)* cell.  Columns include
        ``policies_exposed``, ``lapses``, ``surrenders``, ``deaths``,
        ``gross_lapse_rate``, ``net_lapse_rate``.
    """

    def __init__(
        self,
        summary_df: pd.DataFrame,
        group_by: List[str],
    ) -> None:
        self.summary_df = summary_df
        self._group_by = list(group_by)

    # ------------------------------------------------------------------
    # Core views
    # ------------------------------------------------------------------

    def by_policy_year(self) -> pd.DataFrame:
        """Return per-(group, policy_year) lapse statistics.

        Returns
        -------
        pd.DataFrame
            Columns: group_by fields (if any), ``policy_year``,
            ``policies_exposed``, ``lapses``, ``surrenders``, ``deaths``,
            ``gross_lapse_rate``, ``net_lapse_rate``.
        """
        return self.summary_df.copy()

    # ------------------------------------------------------------------
    # Persistency
    # ------------------------------------------------------------------

    def persistency_table(self) -> pd.DataFrame:
        """Cumulative in-force percentage by policy year.

        ``persistency[k] = Π_{j=1}^{k} (1 − gross_lapse_rate[j])``

        Returns
        -------
        pd.DataFrame
            All columns from :meth:`by_policy_year` plus ``persistency``.
        """
        df = self.summary_df.copy()

        if self._group_by:
            frames: List[pd.DataFrame] = []
            for _, grp in df.groupby(self._group_by):
                grp = grp.sort_values("policy_year").reset_index(drop=True)
                grp["persistency"] = (
                    1.0 - grp["gross_lapse_rate"].fillna(0.0)
                ).cumprod()
                frames.append(grp)
            return (
                pd.concat(frames, ignore_index=True)
                .sort_values(self._group_by + ["policy_year"])
                .reset_index(drop=True)
            )

        df = df.sort_values("policy_year").reset_index(drop=True)
        df["persistency"] = (1.0 - df["gross_lapse_rate"].fillna(0.0)).cumprod()
        return df

    # ------------------------------------------------------------------
    # Kaplan-Meier survival curve
    # ------------------------------------------------------------------

    def survival_curve(self) -> pd.DataFrame:
        """Kaplan-Meier survival estimate at each policy year.

        Lapses are treated as events; deaths and surrenders are treated as
        censored (half-year adjustment).  The risk set for each period is::

            risk_set[k] = policies_exposed[k] − deaths[k]/2 − surrenders[k]/2

        and the KM increment is::

            km_increment[k] = 1 − lapses[k] / risk_set[k]

        Returns
        -------
        pd.DataFrame
            All columns from :meth:`by_policy_year` plus ``km_survival``.
        """
        df = self.summary_df.copy()

        def _km(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values("policy_year").reset_index(drop=True)
            risk_set = (
                grp["policies_exposed"]
                - grp["deaths"] / 2.0
                - grp["surrenders"] / 2.0
            )
            km_incr = np.where(
                risk_set > 0,
                1.0 - grp["lapses"].to_numpy() / risk_set.to_numpy(),
                1.0,
            )
            grp["km_survival"] = np.cumprod(km_incr)
            return grp

        if self._group_by:
            frames = [_km(g) for _, g in df.groupby(self._group_by)]
            return (
                pd.concat(frames, ignore_index=True)
                .sort_values(self._group_by + ["policy_year"])
                .reset_index(drop=True)
            )

        return _km(df)

    # ------------------------------------------------------------------
    # A/E vs assumption
    # ------------------------------------------------------------------

    def ae_vs_assumption(self, assumption_table: pd.DataFrame) -> pd.DataFrame:
        """A/E comparison against an assumed lapse rate table.

        Parameters
        ----------
        assumption_table:
            DataFrame with either a ``policy_year`` column and an
            ``assumed_lapse_rate`` column, or indexed by policy year with
            an ``assumed_lapse_rate`` column.

        Returns
        -------
        pd.DataFrame
            Extends :meth:`by_policy_year` with columns
            ``assumed_lapse_rate``, ``expected_lapses``, ``ae_ratio``.
        """
        df = self.by_policy_year()

        if "policy_year" in assumption_table.columns:
            asmp = assumption_table.set_index("policy_year")["assumed_lapse_rate"]
        else:
            asmp = assumption_table["assumed_lapse_rate"]

        df["assumed_lapse_rate"] = df["policy_year"].map(asmp)
        df["expected_lapses"] = df["assumed_lapse_rate"] * df["policies_exposed"]
        df["ae_ratio"] = np.where(
            df["expected_lapses"] > 0,
            df["lapses"] / df["expected_lapses"],
            np.nan,
        )
        return df


# ---------------------------------------------------------------------------
# LapseStudy
# ---------------------------------------------------------------------------

class LapseStudy:
    """Orchestrates a lapse experience study.

    Parameters
    ----------
    dataset:
        Collection of policy records to study.
    study:
        Observation window (start and end date).
    group_by:
        PolicyRecord field names to segment results by
        (e.g. ``['gender']``).  Pass ``[]`` or omit for aggregate only.
    """

    def __init__(
        self,
        dataset: PolicyDataset,
        study: StudyPeriod,
        group_by: Optional[List[str]] = None,
    ) -> None:
        self._dataset = dataset
        self._study = study
        self._group_by = list(group_by) if group_by is not None else []

    # ------------------------------------------------------------------

    def run(self) -> LapseResults:
        """Execute the study and return :class:`LapseResults`.

        Steps
        -----
        1. For each policy, determine the policy-year segments within the
           study window via anniversary-based splitting.
        2. Count distinct policies exposed in each (group, policy_year) cell.
        3. Identify exits (LAPSED, SURRENDERED, DEATH) and assign them to
           the policy year in which the exit date falls.
        4. Compute gross and net lapse rates.
        """
        group_by = self._group_by
        group_cols = group_by + ["policy_year"]

        exposed_rows: List[dict] = []
        exit_rows: List[dict] = []

        for record in self._dataset._records:
            try:
                obs_start, obs_end = days_in_study(record, self._study)
            except OutOfStudyError:
                continue

            group_vals = {f: getattr(record, f) for f in group_by}

            # Track which policy years this record is exposed to
            for py, _ in _iter_py_segments(record.issue_date, obs_start, obs_end):
                exposed_rows.append(
                    {**group_vals, "_policy_id": record.policy_id, "policy_year": py}
                )

            # Classify exits (lapse / surrender / death)
            if record.status in ("LAPSED", "SURRENDERED", "DEATH"):
                exit_py = policy_year_at(record.issue_date, obs_end)
                exit_rows.append({
                    **group_vals,
                    "_policy_id": record.policy_id,
                    "policy_year": exit_py,
                    "is_lapse":    int(record.status == "LAPSED"),
                    "is_surrender": int(record.status == "SURRENDERED"),
                    "is_death":    int(record.status == "DEATH"),
                })

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------

        if not exposed_rows:
            empty_cols = group_cols + [
                "policies_exposed", "lapses", "surrenders", "deaths",
                "gross_lapse_rate", "net_lapse_rate",
            ]
            return LapseResults(pd.DataFrame(columns=empty_cols), group_by)

        exp_df = pd.DataFrame(exposed_rows)
        exposed_agg = (
            exp_df.groupby(group_cols, as_index=False)
            .agg(policies_exposed=("_policy_id", "nunique"))
        )

        if exit_rows:
            exit_df = pd.DataFrame(exit_rows)
            exits_agg = (
                exit_df.groupby(group_cols, as_index=False)
                .agg(
                    lapses=("is_lapse", "sum"),
                    surrenders=("is_surrender", "sum"),
                    deaths=("is_death", "sum"),
                )
            )
        else:
            exits_agg = pd.DataFrame(
                columns=group_cols + ["lapses", "surrenders", "deaths"]
            )

        detail = (
            exposed_agg
            .merge(exits_agg, on=group_cols, how="left")
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        for col in ("lapses", "surrenders", "deaths"):
            if col not in detail.columns:
                detail[col] = 0.0
            else:
                detail[col] = pd.to_numeric(detail[col], errors="coerce").fillna(0.0)

        # ------------------------------------------------------------------
        # Rates
        # ------------------------------------------------------------------

        detail["gross_lapse_rate"] = np.where(
            detail["policies_exposed"] > 0,
            detail["lapses"] / detail["policies_exposed"],
            np.nan,
        )

        net_denom = (
            detail["policies_exposed"] - detail["deaths"] / 2.0
        )
        detail["net_lapse_rate"] = np.where(
            net_denom > 0,
            detail["lapses"] / net_denom,
            np.nan,
        )

        return LapseResults(detail, group_by)
