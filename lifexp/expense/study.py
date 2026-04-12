"""Expense experience study: ExpenseStudy and ExpenseResults."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lifexp.core.data_model import PolicyDataset
from lifexp.core.study_period import StudyPeriod


# ---------------------------------------------------------------------------
# ExpenseResults
# ---------------------------------------------------------------------------

class ExpenseResults:
    """Results produced by :meth:`ExpenseStudy.run`.

    Attributes
    ----------
    expense_df : pd.DataFrame
        Aggregated expenses by ``(year, expense_type, cost_centre)``.
        Columns: ``year``, ``expense_type``, ``cost_centre``, ``amount``.
    unit_cost_df : pd.DataFrame
        Per-year unit costs.  Columns: ``year``, ``if_count``,
        ``new_policies``, ``total_premium``, ``total_sa``,
        ``renewal_expenses``, ``acquisition_expenses``,
        ``per_policy``, ``per_new_policy``, ``per_premium_pct``,
        ``per_sa_pct``.
    """

    def __init__(
        self,
        expense_df: pd.DataFrame,
        unit_cost_df: pd.DataFrame,
    ) -> None:
        self.expense_df  = expense_df
        self.unit_cost_df = unit_cost_df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def unit_costs(self) -> pd.DataFrame:
        """Return per-year unit costs.

        Returns
        -------
        pd.DataFrame
            Columns: ``year``, ``per_policy``, ``per_new_policy``,
            ``per_premium_pct``, ``per_sa_pct``.
        """
        cols = ["year", "per_policy", "per_new_policy", "per_premium_pct", "per_sa_pct"]
        return self.unit_cost_df[cols].copy()

    def ae_vs_assumption(self, assumption: Dict[str, float]) -> pd.DataFrame:
        """Actual vs assumed unit costs.

        Parameters
        ----------
        assumption :
            Dict mapping metric name to assumed value.  Supported keys:
            ``'per_policy'``, ``'per_new_policy'``, ``'per_premium_pct'``,
            ``'per_sa_pct'``.

        Returns
        -------
        pd.DataFrame
            Per year: ``year``, ``metric``, ``actual``, ``assumed``,
            ``ae_ratio``.
        """
        uc = self.unit_costs()
        rows: List[dict] = []
        for metric, assumed_val in assumption.items():
            if metric not in uc.columns:
                continue
            for _, row in uc.iterrows():
                actual = float(row[metric])
                if np.isnan(actual) or assumed_val == 0.0:
                    ae = np.nan
                else:
                    ae = actual / assumed_val
                rows.append({
                    "year":     int(row["year"]),
                    "metric":   metric,
                    "actual":   actual,
                    "assumed":  assumed_val,
                    "ae_ratio": ae,
                })
        return pd.DataFrame(rows, columns=["year", "metric", "actual", "assumed", "ae_ratio"])

    def inflation_analysis(self) -> pd.DataFrame:
        """Year-on-year change in per-policy unit cost.

        Returns
        -------
        pd.DataFrame
            Columns: ``year``, ``per_policy``, ``prior_per_policy``,
            ``yoy_change``, ``yoy_pct``.
        ``yoy_pct`` is ``NaN`` for the first year or when prior cost is zero.
        """
        uc = self.unit_cost_df[["year", "per_policy"]].sort_values("year").copy()
        uc["prior_per_policy"] = uc["per_policy"].shift(1)
        uc["yoy_change"] = uc["per_policy"] - uc["prior_per_policy"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            uc["yoy_pct"] = np.where(
                uc["prior_per_policy"].notna() & (uc["prior_per_policy"] != 0.0),
                (uc["per_policy"] - uc["prior_per_policy"]) / uc["prior_per_policy"],
                np.nan,
            )
        return uc.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ExpenseStudy
# ---------------------------------------------------------------------------

class ExpenseStudy:
    """Expense experience study.

    Parameters
    ----------
    expense_data : pd.DataFrame
        One row per cost line.  Required columns:
        ``cost_centre``, ``expense_type``, ``amount``, ``year``.
    policy_data : PolicyDataset
        Policy-level data used to compute exposure denominators.
    study : StudyPeriod
        Observation window (used to derive study years).
    allocation_keys : dict
        Maps ``expense_type`` → allocation basis.
        Recognised bases:

        * ``'if_policy_count'`` — allocate to renewal unit cost
          (``per_policy = renewal_expenses / if_count``).
        * ``'new_policies'`` — allocate to acquisition unit cost
          (``per_new_policy = acquisition_expenses / new_policies``).

        Unknown expense types are ignored in unit-cost allocation but are
        included in ``expense_df``.
    """

    def __init__(
        self,
        expense_data: pd.DataFrame,
        policy_data: PolicyDataset,
        study: StudyPeriod,
        allocation_keys: Dict[str, str],
    ) -> None:
        self._expense_data   = expense_data.copy()
        self._policy_data    = policy_data
        self._study          = study
        self._allocation_keys = dict(allocation_keys)

    # ------------------------------------------------------------------

    def run(self) -> ExpenseResults:
        """Execute the study and return :class:`ExpenseResults`."""
        # Determine study years from the expense data
        years = sorted(self._expense_data["year"].dropna().unique().astype(int))

        # ---------------------------------------------------------------
        # Aggregate expense_df
        # ---------------------------------------------------------------
        group_cols = ["year", "expense_type", "cost_centre"]
        expense_df = (
            self._expense_data
            .groupby(group_cols, as_index=False)
            .agg(amount=("amount", "sum"))
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------
        # Build denominators per year
        # ---------------------------------------------------------------
        unit_rows: List[dict] = []
        for year in years:
            year_end   = pd.Timestamp(year, 12, 31).date()
            year_start = pd.Timestamp(year,  1,  1).date()

            if_count      = 0
            new_policies  = 0
            total_premium = 0.0
            total_sa      = 0.0

            for rec in self._policy_data._records:
                # In-force at year-end: issued on or before year-end
                # and (not exited OR exited after year-end)
                issued_by_year_end = rec.issue_date <= year_end
                still_inforce = (
                    rec.exit_date is None or rec.exit_date > year_end
                )
                if issued_by_year_end and still_inforce:
                    if_count     += 1
                    total_premium += float(rec.annual_premium)
                    total_sa      += float(rec.sum_assured)

                # New policies: issued in the calendar year
                if year_start <= rec.issue_date <= year_end:
                    new_policies += 1

            # -------------------------------------------------------
            # Sum expenses by allocation basis
            # -------------------------------------------------------
            renewal_expenses     = 0.0
            acquisition_expenses = 0.0

            year_exp = expense_df[expense_df["year"] == year]
            for _, row in year_exp.iterrows():
                basis = self._allocation_keys.get(str(row["expense_type"]))
                amount = float(row["amount"])
                if basis == "if_policy_count":
                    renewal_expenses += amount
                elif basis == "new_policies":
                    acquisition_expenses += amount

            # -------------------------------------------------------
            # Unit costs (NaN + warn when denominator is zero)
            # -------------------------------------------------------
            if if_count > 0:
                per_policy = renewal_expenses / if_count
            else:
                warnings.warn(
                    f"Year {year}: if_count is zero; per_policy is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
                per_policy = np.nan

            if new_policies > 0:
                per_new_policy = acquisition_expenses / new_policies
            else:
                warnings.warn(
                    f"Year {year}: new_policies is zero; per_new_policy is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
                per_new_policy = np.nan

            if total_premium > 0:
                per_premium_pct = renewal_expenses / total_premium
            else:
                warnings.warn(
                    f"Year {year}: total_premium is zero; per_premium_pct is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
                per_premium_pct = np.nan

            if total_sa > 0:
                per_sa_pct = renewal_expenses / total_sa
            else:
                warnings.warn(
                    f"Year {year}: total_sa is zero; per_sa_pct is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
                per_sa_pct = np.nan

            unit_rows.append({
                "year":                year,
                "if_count":            if_count,
                "new_policies":        new_policies,
                "total_premium":       total_premium,
                "total_sa":            total_sa,
                "renewal_expenses":    renewal_expenses,
                "acquisition_expenses": acquisition_expenses,
                "per_policy":          per_policy,
                "per_new_policy":      per_new_policy,
                "per_premium_pct":     per_premium_pct,
                "per_sa_pct":          per_sa_pct,
            })

        unit_cost_df = pd.DataFrame(unit_rows)

        return ExpenseResults(
            expense_df=expense_df,
            unit_cost_df=unit_cost_df,
        )
