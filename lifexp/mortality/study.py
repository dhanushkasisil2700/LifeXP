"""Mortality experience study: MortalityStudy and MortalityResults."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from lifexp.core.data_model import PolicyDataset
from lifexp.core.exposure import central_etr as _central_etr
from lifexp.core.exposure import initial_etr as _initial_etr
from lifexp.core.study_period import AgeBasis, StudyPeriod
from lifexp.core.tables import MortalityTable, TableRegistry
from lifexp.graduation.whittaker import whittaker_1d


# ---------------------------------------------------------------------------
# MortalityResults
# ---------------------------------------------------------------------------

class MortalityResults:
    """Results produced by :meth:`MortalityStudy.run`.

    Attributes
    ----------
    summary_df:
        One row per *(age, group_by...)* cell.  Columns include
        ``central_etr``, ``initial_etr``, ``deaths``,
        ``crude_central_rate``, ``crude_initial_rate``,
        ``standard_rate``, ``expected_deaths``,
        ``ae_ratio``, ``ae_central``, ``ae_initial``.
    overall_ae:
        Aggregate A/E ratio = total_deaths / total_expected_deaths.
    total_deaths:
        Sum of observed deaths across all cells.
    total_etr:
        Sum of central ETR across all cells (years).
    """

    def __init__(
        self,
        summary_df: pd.DataFrame,
        overall_ae: float,
        total_deaths: float,
        total_etr: float,
        detail_df: pd.DataFrame,
        standard_table: MortalityTable,
        standard_table_name: str,
        group_by: List[str],
    ) -> None:
        self.summary_df = summary_df
        self.overall_ae = overall_ae
        self.total_deaths = total_deaths
        self.total_etr = total_etr
        self._detail_df = detail_df
        self._standard_table = standard_table
        self._standard_table_name = standard_table_name
        self._group_by = list(group_by)

    # ------------------------------------------------------------------
    # Aggregated views
    # ------------------------------------------------------------------

    def ae_by_age(self) -> pd.DataFrame:
        """A/E ratio collapsed to age only (group_by dimensions removed).

        Returns
        -------
        pd.DataFrame
            Columns: ``age, deaths, central_etr, expected_deaths,
            crude_central_rate, standard_rate, ae_ratio``.
        """
        df = (
            self.summary_df
            .groupby("age", as_index=False)
            .agg(
                deaths=("deaths", "sum"),
                central_etr=("central_etr", "sum"),
                expected_deaths=("expected_deaths", "sum"),
            )
            .sort_values("age")
            .reset_index(drop=True)
        )
        df["crude_central_rate"] = np.where(
            df["central_etr"] > 0,
            df["deaths"] / df["central_etr"],
            np.nan,
        )
        df["standard_rate"] = df["age"].apply(
            lambda a: self._standard_table.qx(int(a))
        )
        df["ae_ratio"] = np.where(
            df["expected_deaths"] > 0,
            df["deaths"] / df["expected_deaths"],
            np.nan,
        )
        return df

    def ae_by_policy_year(self) -> pd.DataFrame:
        """A/E ratio aggregated by policy year (duration).

        Returns
        -------
        pd.DataFrame
            Columns: ``policy_year, deaths, central_etr, expected_deaths, ae_ratio``.
        """
        df = (
            self._detail_df
            .groupby("policy_year", as_index=False)
            .agg(
                deaths=("deaths", "sum"),
                central_etr=("central_etr", "sum"),
                expected_deaths=("expected_deaths", "sum"),
            )
            .sort_values("policy_year")
            .reset_index(drop=True)
        )
        df["ae_ratio"] = np.where(
            df["expected_deaths"] > 0,
            df["deaths"] / df["expected_deaths"],
            np.nan,
        )
        return df

    # ------------------------------------------------------------------
    # Confidence intervals (Garwood exact Poisson)
    # ------------------------------------------------------------------

    def confidence_interval(self, level: float = 0.95) -> pd.DataFrame:
        """Poisson exact (Garwood) confidence interval on A/E per age cell.

        The Garwood CI for an observed Poisson count *d* at significance
        level α is::

            lower_count = χ²(α/2 ; 2d) / 2          (0 when d = 0)
            upper_count = χ²(1−α/2 ; 2(d+1)) / 2

        Dividing by *expected_deaths* converts to an A/E interval.

        Returns
        -------
        pd.DataFrame
            Extends :meth:`ae_by_age` with columns
            ``ae_ci_lower``, ``ae_ci_upper``, ``ci_width``.
        """
        alpha = 1.0 - level
        df = self.ae_by_age()

        lowers: list = []
        uppers: list = []

        for _, row in df.iterrows():
            d = float(row["deaths"])
            exp = float(row["expected_deaths"])

            if exp <= 0:
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue

            # Garwood lower: 0 when d = 0
            ci_lo = 0.0 if d == 0 else _chi2.ppf(alpha / 2.0, 2.0 * d) / 2.0
            ci_hi = _chi2.ppf(1.0 - alpha / 2.0, 2.0 * (d + 1.0)) / 2.0

            lowers.append(ci_lo / exp)
            uppers.append(ci_hi / exp)

        df["ae_ci_lower"] = lowers
        df["ae_ci_upper"] = uppers
        df["ci_width"] = df["ae_ci_upper"] - df["ae_ci_lower"]
        return df

    # ------------------------------------------------------------------
    # Graduation
    # ------------------------------------------------------------------

    def graduate(self, method: str = "whittaker", lam: float = 100.0) -> pd.DataFrame:
        """Graduate crude central mortality rates.

        Parameters
        ----------
        method:
            ``'whittaker'`` (default) uses 2nd-order Whittaker-Henderson.
        lam:
            Smoothing parameter passed to the graduation method.

        Returns
        -------
        pd.DataFrame
            Columns: ``age, crude_central_rate, standard_rate, graduated_rate,
            ae_ratio, ae_graduated``.
        """
        age_df = self.ae_by_age()

        crude_arr = age_df["crude_central_rate"].fillna(0.0).to_numpy()
        weights_arr = age_df["central_etr"].to_numpy()
        age_idx = age_df["age"].to_numpy()

        crude_s = pd.Series(crude_arr, index=age_idx)
        weights_s = pd.Series(weights_arr, index=age_idx)

        if method == "whittaker":
            graduated_s = whittaker_1d(crude_s, weights_s, lam=lam)
        else:
            raise ValueError(f"Unknown graduation method: {method!r}")

        result = age_df[["age", "crude_central_rate", "standard_rate", "ae_ratio"]].copy()
        result["graduated_rate"] = graduated_s.values
        result["ae_graduated"] = np.where(
            result["standard_rate"] > 0,
            result["graduated_rate"] / result["standard_rate"],
            np.nan,
        )
        return result

    # ------------------------------------------------------------------
    # Formatted A/E table
    # ------------------------------------------------------------------

    def ae_table(self, standard: str = "A_1967_70") -> pd.DataFrame:
        """Formatted A/E comparison table.

        If *standard* differs from the table used in this study, expected
        deaths are recomputed against the requested table.

        Returns
        -------
        pd.DataFrame
            Columns: ``age, deaths, central_etr, standard_rate,
            expected_deaths, ae``.
        """
        df = self.ae_by_age()

        if standard != self._standard_table_name:
            other_table = TableRegistry().get(standard)
            df["standard_rate"] = df["age"].apply(
                lambda a: other_table.qx(int(a))
            )
            df["expected_deaths"] = df["standard_rate"] * df["central_etr"]
            df["ae_ratio"] = np.where(
                df["expected_deaths"] > 0,
                df["deaths"] / df["expected_deaths"],
                np.nan,
            )

        return (
            df[["age", "deaths", "central_etr", "standard_rate",
                "expected_deaths", "ae_ratio"]]
            .rename(columns={"ae_ratio": "ae"})
        )


# ---------------------------------------------------------------------------
# MortalityStudy
# ---------------------------------------------------------------------------

class MortalityStudy:
    """Orchestrates a mortality experience study.

    Parameters
    ----------
    dataset:
        Collection of policy records to study.
    study:
        Observation window (start and end date).
    age_basis:
        How integer ages are assigned within time segments.
    standard_table:
        Name of the standard mortality table to compare against.
        Must be registered in :class:`~lifexp.core.tables.TableRegistry`.
    group_by:
        Additional PolicyRecord field names to segment results by
        (e.g. ``['gender', 'smoker_status']``).  Pass ``[]`` for
        age-only results.
    """

    def __init__(
        self,
        dataset: PolicyDataset,
        study: StudyPeriod,
        age_basis: AgeBasis = AgeBasis.LAST_BIRTHDAY,
        standard_table: str = "A_1967_70",
        group_by: Optional[List[str]] = None,
    ) -> None:
        self._dataset = dataset
        self._study = study
        self._age_basis = age_basis
        self._standard_table_name = standard_table
        self._group_by = list(group_by) if group_by is not None else []

    # ------------------------------------------------------------------

    def run(self) -> MortalityResults:
        """Execute the study and return :class:`MortalityResults`.

        Steps
        -----
        1. Compute central and initial ETR via :mod:`lifexp.core.exposure`.
        2. Merge on *(age, policy_year, group_by...)* to build ``detail_df``.
        3. Look up standard table rates and compute expected deaths.
        4. Aggregate to ``summary_df`` (one row per age/group cell).
        5. Derive crude rates and A/E columns.
        """
        registry = TableRegistry()
        table = registry.get(self._standard_table_name)

        # ----------------------------------------------------------
        # 1. Raw ETR computation
        # ----------------------------------------------------------
        central_df = _central_etr(
            self._dataset, self._study, self._age_basis, self._group_by
        )
        initial_df = _initial_etr(
            self._dataset, self._study, self._age_basis, self._group_by
        )

        # ----------------------------------------------------------
        # 2. Build per-(group, age, policy_year) detail DataFrame
        # ----------------------------------------------------------
        merge_keys = self._group_by + ["age", "policy_year"]

        # Drop duplicate policy_count column from initial_df before merge
        initial_for_merge = initial_df.drop(
            columns=["policy_count"], errors="ignore"
        )

        detail = (
            central_df
            .merge(initial_for_merge, on=merge_keys, how="outer")
            .fillna({"deaths": 0.0, "central_etr": 0.0, "initial_etr": 0.0})
        )

        # Standard rate lookup and expected deaths.
        # Ages outside the table range yield NaN standard_rate; those rows
        # are excluded from aggregate A/E via nansum logic below.
        import warnings as _warnings
        from lifexp.core.tables import AgeOutOfRangeError as _AgeOOR

        def _safe_qx(a: float) -> float:
            try:
                return table.qx(int(a))
            except _AgeOOR as exc:
                _warnings.warn(str(exc), UserWarning, stacklevel=6)
                return np.nan

        detail["standard_rate"] = detail["age"].apply(_safe_qx)
        # Drop rows where age is out of the table range so they do not
        # pollute exposure sums or A/E calculations.
        detail = detail.dropna(subset=["standard_rate"]).copy()

        detail["expected_deaths"] = detail["standard_rate"] * detail["central_etr"]
        detail["ae_ratio"] = np.where(
            detail["expected_deaths"] > 0,
            detail["deaths"] / detail["expected_deaths"],
            np.nan,
        )

        # ----------------------------------------------------------
        # 3. Collapse to summary_df (per age/group)
        # ----------------------------------------------------------
        agg_keys = self._group_by + ["age"]
        summary = (
            detail
            .groupby(agg_keys, as_index=False)
            .agg(
                central_etr=("central_etr", "sum"),
                initial_etr=("initial_etr", "sum"),
                deaths=("deaths", "sum"),
                expected_deaths=("expected_deaths", "sum"),
            )
            .sort_values(agg_keys)
            .reset_index(drop=True)
        )

        # Derived columns on the summary
        summary["standard_rate"] = summary["age"].apply(_safe_qx)
        summary["crude_central_rate"] = np.where(
            summary["central_etr"] > 0,
            summary["deaths"] / summary["central_etr"],
            np.nan,
        )
        summary["crude_initial_rate"] = np.where(
            summary["initial_etr"] > 0,
            summary["deaths"] / summary["initial_etr"],
            np.nan,
        )
        summary["ae_ratio"] = np.where(
            summary["expected_deaths"] > 0,
            summary["deaths"] / summary["expected_deaths"],
            np.nan,
        )
        # ae_central = A/E on central ETR basis (same as ae_ratio here)
        summary["ae_central"] = summary["ae_ratio"]
        # ae_initial = A/E on initial ETR basis
        _initial_expected = summary["standard_rate"] * summary["initial_etr"]
        summary["ae_initial"] = np.where(
            _initial_expected > 0,
            summary["deaths"] / _initial_expected,
            np.nan,
        )

        # ----------------------------------------------------------
        # 4. Aggregate scalars
        # ----------------------------------------------------------
        total_deaths = float(summary["deaths"].sum())
        total_etr = float(summary["central_etr"].sum())
        total_expected = float(summary["expected_deaths"].sum())
        overall_ae = (
            total_deaths / total_expected
            if total_expected > 0 else float("nan")
        )

        return MortalityResults(
            summary_df=summary,
            overall_ae=overall_ae,
            total_deaths=total_deaths,
            total_etr=total_etr,
            detail_df=detail,
            standard_table=table,
            standard_table_name=self._standard_table_name,
            group_by=self._group_by,
        )
