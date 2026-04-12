"""Reinsurance experience study: RIStudy and RIResults."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from lifexp.core.data_model import ClaimDataset, PolicyDataset
from lifexp.core.date_utils import OutOfStudyError, age_at, days_in_study
from lifexp.core.exposure import _iter_segments
from lifexp.core.study_period import AgeBasis, StudyPeriod
from lifexp.core.tables import MortalityTable


_AGE_BASIS = AgeBasis.LAST_BIRTHDAY


# ---------------------------------------------------------------------------
# RIResults
# ---------------------------------------------------------------------------

class RIResults:
    """Results produced by :meth:`RIStudy.run`.

    Attributes
    ----------
    summary_df : pd.DataFrame
        One row per *(group_by…, age)* cell.  Columns:

        ``naar_etr``
            NAAR-weighted central exposure (NAAR × years).
        ``etr``
            Plain central exposure (years).
        ``deaths``
            Observed death count.
        ``actual_ri_claims``
            Actual RI claim payments (Σ NAAR for deceased policies).
        ``expected_ri_claims``
            Expected RI cost = ``ri_rate × naar_etr``.
        ``earned_premium``
            RI premium earned = ``ri_rate × naar_etr`` (identical to
            expected under risk-premium pricing).
        ``ri_rate``
            RI table mortality rate for this age cell.
    """

    def __init__(
        self,
        summary_df: pd.DataFrame,
        ri_table: MortalityTable,
        retention_rate: float,
        treaty_type: str,
        group_by: List[str],
    ) -> None:
        self.summary_df = summary_df
        self._ri_table = ri_table
        self._retention_rate = retention_rate
        self._treaty_type = treaty_type
        self._group_by = list(group_by)

    # ------------------------------------------------------------------
    # A/E views
    # ------------------------------------------------------------------

    def ae_by_age(self) -> pd.DataFrame:
        """Actual vs expected RI claims by *(group_by…, age)* cell.

        Returns
        -------
        pd.DataFrame
            Extends :attr:`summary_df` with ``ae_ratio``.
        """
        group_cols = self._group_by + ["age"]
        df = (
            self.summary_df
            .groupby(group_cols, as_index=False)
            .agg(
                naar_etr=("naar_etr", "sum"),
                etr=("etr", "sum"),
                deaths=("deaths", "sum"),
                actual_ri_claims=("actual_ri_claims", "sum"),
                expected_ri_claims=("expected_ri_claims", "sum"),
                earned_premium=("earned_premium", "sum"),
            )
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        df["ri_rate"] = df["age"].apply(lambda a: self._ri_table.qx(int(a)))
        df["ae_ratio"] = np.where(
            df["expected_ri_claims"] > 0,
            df["actual_ri_claims"] / df["expected_ri_claims"],
            np.nan,
        )
        return df

    def ae_by_treaty(self) -> pd.DataFrame:
        """Actual vs expected RI claims by treaty segment.

        Uses *group_by* dimensions as the treaty segmentation.  If
        *group_by* is empty, returns a single aggregate row tagged with
        the treaty type.

        Returns
        -------
        pd.DataFrame
        """
        if self._group_by:
            df = (
                self.summary_df
                .groupby(self._group_by, as_index=False)
                .agg(
                    naar_etr=("naar_etr", "sum"),
                    deaths=("deaths", "sum"),
                    actual_ri_claims=("actual_ri_claims", "sum"),
                    expected_ri_claims=("expected_ri_claims", "sum"),
                    earned_premium=("earned_premium", "sum"),
                )
            )
        else:
            df = pd.DataFrame([{
                "treaty_type": self._treaty_type,
                "naar_etr": float(self.summary_df["naar_etr"].sum()),
                "deaths": float(self.summary_df["deaths"].sum()),
                "actual_ri_claims": float(self.summary_df["actual_ri_claims"].sum()),
                "expected_ri_claims": float(self.summary_df["expected_ri_claims"].sum()),
                "earned_premium": float(self.summary_df["earned_premium"].sum()),
            }])

        df["ae_ratio"] = np.where(
            df["expected_ri_claims"] > 0,
            df["actual_ri_claims"] / df["expected_ri_claims"],
            np.nan,
        )
        return df

    # ------------------------------------------------------------------
    # Scalar summary metrics
    # ------------------------------------------------------------------

    def loss_ratio(self) -> float:
        """Actual RI claims / earned RI premium (profit-share denominator).

        Returns 0.0 when earned premium is zero (no ceded risk).
        """
        total_claims  = float(self.summary_df["actual_ri_claims"].sum())
        total_premium = float(self.summary_df["earned_premium"].sum())
        return total_claims / total_premium if total_premium > 0.0 else 0.0

    def break_even_mortality(self) -> float:
        """Flat mortality rate at which actual RI cost equals earned premium.

        Defined as::

            q_be = Σ actual_ri_claims / Σ naar_etr

        Substituting *q_be* back as the RI rate makes loss_ratio = 1.0.

        Returns 0.0 when total NAAR-ETR is zero.
        """
        total_claims   = float(self.summary_df["actual_ri_claims"].sum())
        total_naar_etr = float(self.summary_df["naar_etr"].sum())
        return total_claims / total_naar_etr if total_naar_etr > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Analytical views
    # ------------------------------------------------------------------

    def implied_mortality(self) -> pd.DataFrame:
        """Back-calculated mortality rate implied by actual RI claims.

        Returns
        -------
        pd.DataFrame
            Extends :meth:`ae_by_age` with ``implied_rate``
            (= ``actual_ri_claims / naar_etr``).
        """
        df = self.ae_by_age().copy()
        df["implied_rate"] = np.where(
            df["naar_etr"] > 0,
            df["actual_ri_claims"] / df["naar_etr"],
            np.nan,
        )
        return df

    def cost_sensitivity(self, delta_pct: float) -> pd.DataFrame:
        """Impact of a ±*delta_pct* percentage-point shift in mortality on RI cost.

        The sensitivity is linear by construction::

            shocked_cost  = (1 + delta_pct / 100) × expected_ri_claims
            delta_cost    = (delta_pct / 100)      × expected_ri_claims

        Parameters
        ----------
        delta_pct :
            Percentage shift in mortality (e.g., ``10.0`` for +10 %).

        Returns
        -------
        pd.DataFrame
            Per-cell costs: ``expected_ri_claims``, ``shocked_cost``,
            ``delta_cost``, ``delta_pct``.
        """
        group_cols = self._group_by + ["age"]
        df = self.ae_by_age()[group_cols + ["expected_ri_claims"]].copy()
        shift = delta_pct / 100.0
        df["shocked_cost"] = (1.0 + shift) * df["expected_ri_claims"]
        df["delta_cost"]   = shift * df["expected_ri_claims"]
        df["delta_pct"]    = delta_pct
        return df


# ---------------------------------------------------------------------------
# RIStudy
# ---------------------------------------------------------------------------

class RIStudy:
    """Reinsurance experience study.

    Parameters
    ----------
    dataset :
        PolicyRecord collection.
    claim_data :
        ClaimRecord collection.  For each policy with ``status='DEATH'``,
        the study first looks for a ``CLOSED_DEATH`` claim in *claim_data*
        within the study window and uses its ``claim_amount`` as the actual
        RI settlement.  If no matching claim is found, the default NAAR
        (``sum_assured × (1 − retention_rate)``) is used instead.
    study :
        Observation window.
    ri_table :
        Reinsurer's mortality rate table.  Provides expected claim rates
        by age for pricing / A/E comparison.
    treaty_type :
        ``'YRT'``, ``'COINSURANCE'``, or ``'MODCO'``.  Stored for
        reporting; all treaty types use the same NAAR-ETR calculation in
        this implementation.
    retention_rate :
        Fraction of the sum assured *retained* by the cedant.
        ``1.0`` = keep all (no cession); ``0.0`` = cede all.
        Net Amount at Risk = ``sum_assured × (1 − retention_rate)``.
    group_by :
        PolicyRecord field names used to segment results
        (e.g., ``['gender', 'product_code']``).
    """

    def __init__(
        self,
        dataset: PolicyDataset,
        claim_data: ClaimDataset,
        study: StudyPeriod,
        ri_table: MortalityTable,
        treaty_type: Literal["YRT", "COINSURANCE", "MODCO"],
        retention_rate: float = 1.0,
        group_by: Optional[List[str]] = None,
    ) -> None:
        self._dataset = dataset
        self._claim_data = claim_data
        self._study = study
        self._ri_table = ri_table
        self._treaty_type = treaty_type
        self._retention_rate = float(retention_rate)
        self._group_by = list(group_by) if group_by is not None else []

    # ------------------------------------------------------------------

    def run(self) -> RIResults:
        """Execute the study and return :class:`RIResults`."""
        group_by  = self._group_by
        group_cols = group_by + ["age"]

        etr_rows:   List[dict] = []
        death_rows: List[dict] = []

        for record in self._dataset._records:
            try:
                obs_start, obs_end = days_in_study(record, self._study)
            except OutOfStudyError:
                continue

            group_vals = {f: getattr(record, f) for f in group_by}
            naar = record.sum_assured * (1.0 - self._retention_rate)

            # -------------------------------------------------------
            # Accumulate NAAR-weighted ETR in each (group, age) cell
            # -------------------------------------------------------
            for age, _py, days, _last in _iter_segments(
                record.date_of_birth, record.issue_date,
                obs_start, obs_end, _AGE_BASIS,
            ):
                etr = days / 365.25
                etr_rows.append({
                    **group_vals,
                    "age":       age,
                    "_naar_etr": naar * etr,
                    "_etr":      etr,
                })

            # -------------------------------------------------------
            # Death events within the observation window
            # -------------------------------------------------------
            if record.status == "DEATH" and record.exit_date is not None:
                exit_date = record.exit_date
                if obs_start <= exit_date <= obs_end:
                    # Prefer actual RI settlement from claim_data
                    ri_claims = [
                        c for c in self._claim_data.claims_for(record.policy_id)
                        if c.claim_status == "CLOSED_DEATH"
                        and obs_start <= c.claim_start_date <= obs_end
                    ]
                    actual_naar = (
                        float(ri_claims[0].claim_amount)
                        if ri_claims
                        else naar
                    )
                    death_age = age_at(
                        record.date_of_birth, exit_date, _AGE_BASIS
                    )
                    death_rows.append({
                        **group_vals,
                        "age":              death_age,
                        "_actual_ri_claim": actual_naar,
                    })

        # ---------------------------------------------------------------
        # Aggregate ETR
        # ---------------------------------------------------------------
        if etr_rows:
            etr_agg = (
                pd.DataFrame(etr_rows)
                .groupby(group_cols, as_index=False)
                .agg(naar_etr=("_naar_etr", "sum"), etr=("_etr", "sum"))
                .sort_values(group_cols)
                .reset_index(drop=True)
            )
        else:
            etr_agg = pd.DataFrame(columns=group_cols + ["naar_etr", "etr"])

        # ---------------------------------------------------------------
        # Aggregate death claims
        # ---------------------------------------------------------------
        if death_rows:
            death_agg = (
                pd.DataFrame(death_rows)
                .groupby(group_cols, as_index=False)
                .agg(
                    deaths=("_actual_ri_claim", "count"),
                    actual_ri_claims=("_actual_ri_claim", "sum"),
                )
                .sort_values(group_cols)
                .reset_index(drop=True)
            )
        else:
            death_agg = pd.DataFrame(
                columns=group_cols + ["deaths", "actual_ri_claims"]
            )

        # ---------------------------------------------------------------
        # Merge exposure and claims
        # ---------------------------------------------------------------
        summary = etr_agg.merge(death_agg, on=group_cols, how="left")
        summary["deaths"] = (
            pd.to_numeric(summary["deaths"], errors="coerce").fillna(0.0)
        )
        summary["actual_ri_claims"] = (
            pd.to_numeric(summary["actual_ri_claims"], errors="coerce").fillna(0.0)
        )

        # ---------------------------------------------------------------
        # Expected RI claims and earned premium
        # ---------------------------------------------------------------
        summary["ri_rate"] = summary["age"].apply(
            lambda a: self._ri_table.qx(int(a))
        )
        summary["expected_ri_claims"] = (
            summary["ri_rate"] * summary["naar_etr"]
        )
        # Under risk-premium pricing, earned premium = expected claims
        summary["earned_premium"] = summary["expected_ri_claims"]

        return RIResults(
            summary_df=summary,
            ri_table=self._ri_table,
            retention_rate=self._retention_rate,
            treaty_type=self._treaty_type,
            group_by=group_by,
        )
