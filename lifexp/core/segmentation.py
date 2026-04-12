"""Segmentation and crude-rate calculation for experience studies."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# Longley-Cook full-credibility threshold (95 % CI ±5 %)
_FULL_CREDIBILITY_THRESHOLD: int = 1082


def segment(
    etr_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    group_by: List[str],
) -> pd.DataFrame:
    """Merge exposed-to-risk and death counts, then compute crude rates.

    Both input DataFrames are first aggregated (summed) over any ``policy_year``
    or other non-key columns, leaving one row per ``(age, *group_by)`` cell.

    Parameters
    ----------
    etr_df:
        Output of :func:`lifexp.core.exposure.central_etr`.  Must contain
        ``age`` and ``central_etr`` columns, plus all fields listed in
        *group_by*.
    deaths_df:
        DataFrame with ``age``, ``deaths``, and all fields in *group_by*.
        Typically the output of :func:`lifexp.core.exposure.initial_etr`.
    group_by:
        Additional PolicyRecord field names used as segmentation dimensions
        (e.g. ``['gender', 'smoker_status']``).  Pass ``[]`` for age-only.

    Returns
    -------
    pd.DataFrame
        Columns: ``[*group_by, age, central_etr, policy_count, deaths,
        crude_rate, is_credible]``.

        ``crude_rate``
            deaths / central_etr.  NaN when central_etr = 0.
        ``is_credible``
            True when deaths ≥ the Longley-Cook full-credibility threshold
            (1 082 deaths for a 95 % CI ±5 %).
    """
    merge_keys = group_by + ["age"]

    # Aggregate ETR over policy_year (and anything else that isn't a key/metric)
    etr_agg = (
        etr_df.groupby(merge_keys, as_index=False)
        .agg(central_etr=("central_etr", "sum"), policy_count=("policy_count", "sum"))
    )

    # Aggregate deaths similarly
    deaths_agg = (
        deaths_df.groupby(merge_keys, as_index=False)
        .agg(deaths=("deaths", "sum"))
    )

    merged = (
        etr_agg.merge(deaths_agg, on=merge_keys, how="outer")
        .fillna({"deaths": 0, "central_etr": 0.0, "policy_count": 0})
    )

    # Crude rate: NaN where there is no exposure (central_etr = 0)
    merged["crude_rate"] = np.where(
        merged["central_etr"] > 0,
        merged["deaths"] / merged["central_etr"],
        np.nan,
    )

    # Credibility flag using Longley-Cook threshold
    merged["is_credible"] = merged["deaths"] >= _FULL_CREDIBILITY_THRESHOLD

    return (
        merged.sort_values(merge_keys)
        .reset_index(drop=True)
    )
