"""Commission experience study: CommissionStudy and CommissionResults."""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from lifexp.core.study_period import StudyPeriod


# ---------------------------------------------------------------------------
# CommissionResults
# ---------------------------------------------------------------------------

class CommissionResults:
    """Results produced by :meth:`CommissionStudy.run`.

    Attributes
    ----------
    summary_df : pd.DataFrame
        One row per agent.  Columns: ``agent_id``, ``total_commission``,
        ``policy_count``, ``commission_per_policy``.
    """

    def __init__(self, summary_df: pd.DataFrame) -> None:
        self.summary_df = summary_df

    # ------------------------------------------------------------------

    def flag_anomalies(
        self,
        method: Literal["zscore", "mad", "iqr", "ensemble"] = "ensemble",
        zscore_threshold: float = 3.0,
        mad_threshold: float = 3.5,
        iqr_multiplier: float = 1.5,
        min_ensemble_votes: int = 2,
    ) -> pd.DataFrame:
        """Identify agents with anomalous commission-per-policy values.

        Methods
        -------
        ``'zscore'``
            Standard Z-score: ``|x − μ| / σ > zscore_threshold``.
        ``'mad'``
            Modified Z-score using median absolute deviation:
            ``0.6745 × |x − median| / MAD > mad_threshold``.
            When MAD = 0 all modified Z-scores are set to 0 (no flags).
        ``'iqr'``
            Tukey fences: outside
            ``[Q1 − iqr_multiplier × IQR, Q3 + iqr_multiplier × IQR]``.
        ``'ensemble'``
            Flag agents flagged by ≥ *min_ensemble_votes* individual methods.

        Parameters
        ----------
        method :
            Detection method.
        zscore_threshold, mad_threshold, iqr_multiplier, min_ensemble_votes :
            Tuning parameters for each method.

        Returns
        -------
        pd.DataFrame
            Rows are flagged agents only.  Columns: ``agent_id``,
            ``commission_per_policy``, ``zscore_flag``, ``mad_flag``,
            ``iqr_flag``, ``ensemble_votes``.
        ``commission_per_policy`` is NaN for agents with no policies.

        Warns
        -----
        UserWarning
            When the number of agents is ≤ 1 (anomaly detection is
            undefined for a single point).  Returns empty DataFrame.
        """
        df = self.summary_df.copy()

        _COLS = ["agent_id", "commission_per_policy",
                 "zscore_flag", "mad_flag", "iqr_flag", "ensemble_votes"]

        n = len(df)
        if n <= 1:
            warnings.warn(
                f"Anomaly detection requires > 1 agent; got {n}. "
                "Returning empty result.",
                UserWarning,
                stacklevel=2,
            )
            return pd.DataFrame(columns=_COLS)

        x = df["commission_per_policy"].to_numpy(dtype=float)

        # --- Z-score ---
        mu    = np.nanmean(x)
        sigma = np.nanstd(x, ddof=1)
        if sigma > 0:
            zscores = np.abs(x - mu) / sigma
        else:
            zscores = np.zeros(n)
        zscore_flag = zscores > zscore_threshold

        # --- MAD modified Z-score ---
        median = np.nanmedian(x)
        mad    = np.nanmedian(np.abs(x - median))
        if mad > 0:
            mod_z = 0.6745 * np.abs(x - median) / mad
        else:
            mod_z = np.zeros(n)
        mad_flag = mod_z > mad_threshold

        # --- IQR ---
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        iqr_flag = (x < lower) | (x > upper)

        df["zscore_flag"]    = zscore_flag
        df["mad_flag"]       = mad_flag
        df["iqr_flag"]       = iqr_flag
        df["ensemble_votes"] = zscore_flag.astype(int) + mad_flag.astype(int) + iqr_flag.astype(int)

        if method == "zscore":
            flagged = df[df["zscore_flag"]]
        elif method == "mad":
            flagged = df[df["mad_flag"]]
        elif method == "iqr":
            flagged = df[df["iqr_flag"]]
        else:  # ensemble
            flagged = df[df["ensemble_votes"] >= min_ensemble_votes]

        return (
            flagged[_COLS]
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# CommissionStudy
# ---------------------------------------------------------------------------

class CommissionStudy:
    """Commission experience study.

    Parameters
    ----------
    commission_data : pd.DataFrame
        One row per commission payment.  Required columns:
        ``agent_id``, ``policy_id``, ``commission_amount``, ``payment_date``.
    schedule : dict
        Maps ``product_code`` → expected commission rate (fraction of premium).
        Used for future A/E analysis extensions; stored but not consumed
        in the current implementation.
    study : StudyPeriod
        Observation window.  Only payments with
        ``study.obs_start <= payment_date <= study.obs_end`` are included.
    """

    def __init__(
        self,
        commission_data: pd.DataFrame,
        schedule: Dict[str, float],
        study: StudyPeriod,
    ) -> None:
        self._commission_data = commission_data.copy()
        self._schedule        = dict(schedule)
        self._study           = study

    # ------------------------------------------------------------------

    def run(self) -> CommissionResults:
        """Execute the study and return :class:`CommissionResults`."""
        obs_start = self._study.start_date
        obs_end   = self._study.end_date

        data = self._commission_data.copy()

        # Filter to study window
        data["payment_date"] = pd.to_datetime(data["payment_date"]).dt.date
        mask = (data["payment_date"] >= obs_start) & (data["payment_date"] <= obs_end)
        data = data[mask].copy()

        if data.empty:
            summary = pd.DataFrame(columns=[
                "agent_id", "total_commission", "policy_count", "commission_per_policy"
            ])
            return CommissionResults(summary)

        # Aggregate by agent
        agg = (
            data.groupby("agent_id", as_index=False)
            .agg(
                total_commission=("commission_amount", "sum"),
                policy_count=("policy_id", "nunique"),
            )
        )

        agg["commission_per_policy"] = np.where(
            agg["policy_count"] > 0,
            agg["total_commission"] / agg["policy_count"],
            np.nan,
        )

        return CommissionResults(agg.reset_index(drop=True))
