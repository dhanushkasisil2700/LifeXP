"""Classical credibility weighting for life experience studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from lifexp.core.tables import MortalityTable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CredibilityConfig:
    """Parameters governing how experience rates are blended with a standard table.

    Attributes
    ----------
    method:
        ``'classical'`` uses the Longley-Cook square-root formula.
        ``'buhlmann'`` is reserved for a future Bühlmann-Straub implementation.
    full_credibility_threshold:
        Minimum observed deaths for Z = 1.0.  The Longley-Cook value of 1082
        gives a 95 % confidence interval of ±5 % around the true rate.
    partial_formula:
        Formula used for partial credibility.  Currently only ``'square_root'``
        is supported.
    """

    method: Literal["classical", "buhlmann"] = "classical"
    full_credibility_threshold: int = 1082
    partial_formula: Literal["square_root"] = "square_root"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def classical_credibility(deaths: float, config: CredibilityConfig) -> float:
    """Return the classical credibility factor Z ∈ [0, 1].

    Formula: Z = min( sqrt(deaths / threshold), 1.0 )

    Parameters
    ----------
    deaths:
        Observed death count (may be fractional for weighted data).
    config:
        Credibility configuration specifying the full-credibility threshold.

    Returns
    -------
    float
        Z in [0.0, 1.0].  Returns 0.0 when deaths ≤ 0.
    """
    if deaths <= 0.0:
        return 0.0
    return min(math.sqrt(deaths / config.full_credibility_threshold), 1.0)


def blend_rates(
    experience_rate: float,
    standard_rate: float,
    credibility_z: float,
) -> float:
    """Blend an experience rate with a standard (table) rate using credibility weight Z.

    Formula: blended = Z × experience + (1 − Z) × standard

    Special cases
    -------------
    * If Z ≤ 0 **or** *experience_rate* is NaN (no observed exposure), the
      result is *standard_rate* (full reliance on the table).
    * If Z ≥ 1 the result is *experience_rate* exactly.
    * When *experience_rate* == *standard_rate* the result equals both,
      regardless of Z (blending invariant).

    Parameters
    ----------
    experience_rate:
        Crude observed rate from the study data.
    standard_rate:
        Published table rate to fall back on.
    credibility_z:
        Weight in [0, 1] from :func:`classical_credibility`.
    """
    if credibility_z <= 0.0 or (isinstance(experience_rate, float) and math.isnan(experience_rate)):
        return standard_rate
    if credibility_z >= 1.0:
        return experience_rate
    return credibility_z * experience_rate + (1.0 - credibility_z) * standard_rate


# ---------------------------------------------------------------------------
# DataFrame-level application
# ---------------------------------------------------------------------------

def apply_credibility(
    segment_df: pd.DataFrame,
    standard_table: MortalityTable,
    config: CredibilityConfig,
) -> pd.DataFrame:
    """Append credibility columns to a segmented experience DataFrame.

    The input *segment_df* must contain at least ``age``, ``deaths``, and
    ``crude_rate`` columns (typically the output of
    :func:`lifexp.core.segmentation.segment`).

    Parameters
    ----------
    segment_df:
        DataFrame produced by the segmentation step.
    standard_table:
        Mortality table used as the standard (fallback) rate for each age.
    config:
        Credibility configuration.

    Returns
    -------
    pd.DataFrame
        A copy of *segment_df* with three additional columns:

        ``credibility_z``
            Classical credibility factor Z ∈ [0, 1].
        ``blended_rate``
            Z × crude_rate + (1 − Z) × standard_rate.
        ``is_fully_credible``
            True when Z = 1.0 (deaths ≥ threshold).
    """
    required = {"age", "deaths", "crude_rate"}
    missing = required - set(segment_df.columns)
    if missing:
        raise ValueError(
            f"segment_df is missing required columns: {sorted(missing)}"
        )

    df = segment_df.copy()

    df["credibility_z"] = df["deaths"].apply(
        lambda d: classical_credibility(float(d), config)
    )

    # Look up the standard rate for each age; propagate AgeOutOfRangeError
    df["_standard_rate"] = df["age"].apply(lambda a: standard_table.qx(int(a)))

    df["blended_rate"] = df.apply(
        lambda row: blend_rates(
            float(row["crude_rate"]),
            float(row["_standard_rate"]),
            float(row["credibility_z"]),
        ),
        axis=1,
    )

    df["is_fully_credible"] = df["credibility_z"] >= 1.0

    df = df.drop(columns=["_standard_rate"])
    return df
