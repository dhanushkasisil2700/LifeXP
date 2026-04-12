"""Core data model: PolicyRecord dataclass and PolicyDataset container."""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from datetime import date
from typing import Dict, List, Literal, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# PolicyRecord
# ---------------------------------------------------------------------------

@dataclass
class PolicyRecord:
    """Represents a single life insurance policy record."""

    policy_id: str
    date_of_birth: date
    issue_date: date
    gender: Literal["M", "F", "U"]
    smoker_status: Literal["S", "NS", "U"]
    sum_assured: float
    annual_premium: float
    product_code: str
    channel: str
    status: Literal["IF", "LAPSED", "DEATH", "SURRENDERED", "MATURED", "PU"]
    exit_date: Optional[date]
    exit_reason: Optional[str]

    def validate(self) -> List[str]:
        """Return a list of validation error strings; empty list means valid."""
        errors: List[str] = []

        if self.issue_date < self.date_of_birth:
            errors.append(
                f"policy_id={self.policy_id}: issue_date ({self.issue_date}) "
                f"must be >= date_of_birth ({self.date_of_birth})"
            )

        if self.exit_date is not None and self.exit_date <= self.issue_date:
            errors.append(
                f"policy_id={self.policy_id}: exit_date ({self.exit_date}) "
                f"must be > issue_date ({self.issue_date})"
            )

        if self.status != "IF" and self.exit_date is None:
            errors.append(
                f"policy_id={self.policy_id}: exit_date must not be None "
                f"when status='{self.status}'"
            )

        if self.sum_assured <= 0:
            errors.append(
                f"policy_id={self.policy_id}: sum_assured ({self.sum_assured}) "
                f"must be > 0"
            )

        if self.annual_premium < 0:
            errors.append(
                f"policy_id={self.policy_id}: annual_premium ({self.annual_premium}) "
                f"must be >= 0"
            )

        return errors


# ---------------------------------------------------------------------------
# PolicyDataset
# ---------------------------------------------------------------------------

_FIELD_NAMES = {f.name for f in fields(PolicyRecord)}

_DATE_FIELDS = {"date_of_birth", "issue_date", "exit_date"}


class PolicyDataset:
    """Container for a collection of PolicyRecord objects."""

    def __init__(self, records: List[PolicyRecord]) -> None:
        self._records: List[PolicyRecord] = list(records)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        field_map: Optional[Dict[str, str]] = None,
    ) -> "PolicyDataset":
        """Build a PolicyDataset from a DataFrame.

        Parameters
        ----------
        df:
            Source DataFrame.
        field_map:
            Optional mapping ``{df_column: PolicyRecord_field}``.  If omitted
            the DataFrame column names are assumed to match PolicyRecord field
            names exactly.
        """
        if field_map:
            df = df.rename(columns=field_map)

        records: List[PolicyRecord] = []
        for _, row in df.iterrows():
            kwargs: dict = {}
            for field_name in _FIELD_NAMES:
                if field_name not in row.index:
                    # missing columns default to None for Optional fields
                    kwargs[field_name] = None
                    continue
                value = row[field_name]
                # Coerce NaN / NaT to None for Optional fields
                if field_name in _DATE_FIELDS:
                    # pd.isna() covers None, float NaN, and pd.NaT uniformly.
                    try:
                        _is_na = pd.isna(value)
                    except (TypeError, ValueError):
                        _is_na = False
                    if _is_na:
                        kwargs[field_name] = None
                    elif isinstance(value, date):
                        kwargs[field_name] = value
                    else:
                        # pandas Timestamp or string
                        try:
                            result = pd.Timestamp(value).date()
                            # Guard: pd.NaT.date() may return NaT rather than raise
                            kwargs[field_name] = None if pd.isna(result) else result
                        except Exception:
                            kwargs[field_name] = None
                else:
                    try:
                        import math
                        if isinstance(value, float) and math.isnan(value):
                            kwargs[field_name] = None
                        else:
                            kwargs[field_name] = value
                    except (TypeError, ValueError):
                        kwargs[field_name] = value

            records.append(PolicyRecord(**kwargs))

        return cls(records)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Collect and return all validation errors across every record.

        Does not raise; returns an empty list when everything is valid.
        """
        errors: List[str] = []
        for record in self._records:
            errors.extend(record.validate())
        return errors

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all records to a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=[f.name for f in fields(PolicyRecord)])
        rows = [asdict(r) for r in self._records]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print record counts grouped by status."""
        all_statuses = ["IF", "LAPSED", "DEATH", "SURRENDERED", "MATURED", "PU"]
        counts: Dict[str, int] = {s: 0 for s in all_statuses}
        for record in self._records:
            counts[record.status] = counts.get(record.status, 0) + 1

        total = len(self._records)
        print(f"PolicyDataset — {total} record(s)")
        print(f"  {'Status':<12}  {'Count':>6}")
        print(f"  {'-'*12}  {'-'*6}")
        for status, count in counts.items():
            print(f"  {status:<12}  {count:>6}")


# ---------------------------------------------------------------------------
# ClaimRecord
# ---------------------------------------------------------------------------

@dataclass
class ClaimRecord:
    """Represents a single insurance claim record."""

    claim_id: str
    policy_id: str
    claim_start_date: date
    claim_end_date: Optional[date]      # None if still on claim
    claim_status: Literal[
        "OPEN", "CLOSED_RECOVERY", "CLOSED_DEATH", "CLOSED_MATURITY"
    ]
    benefit_type: Literal["LUMP_SUM", "PERIODIC"]
    claim_amount: float
    benefit_period_days: Optional[int]  # None for LUMP_SUM


# ---------------------------------------------------------------------------
# ClaimDataset
# ---------------------------------------------------------------------------

_CLAIM_FIELD_NAMES = {f.name for f in fields(ClaimRecord)}
_CLAIM_DATE_FIELDS = {"claim_start_date", "claim_end_date"}


class ClaimDataset:
    """Container for a collection of ClaimRecord objects."""

    def __init__(self, records: List[ClaimRecord]) -> None:
        self._records: List[ClaimRecord] = list(records)
        self._by_policy: Dict[str, List[ClaimRecord]] = {}
        for r in self._records:
            self._by_policy.setdefault(r.policy_id, []).append(r)

    def claims_for(self, policy_id: str) -> List[ClaimRecord]:
        """Return all ClaimRecords for the given policy_id (empty list if none)."""
        return self._by_policy.get(policy_id, [])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        field_map: Optional[Dict[str, str]] = None,
    ) -> "ClaimDataset":
        """Build a ClaimDataset from a DataFrame.

        Parameters
        ----------
        df:
            Source DataFrame.
        field_map:
            Optional ``{df_column: ClaimRecord_field}`` name mapping.
        """
        if field_map:
            df = df.rename(columns=field_map)

        records: List[ClaimRecord] = []
        for _, row in df.iterrows():
            kwargs: dict = {}
            for field_name in _CLAIM_FIELD_NAMES:
                if field_name not in row.index:
                    kwargs[field_name] = None
                    continue
                value = row[field_name]
                if field_name in _CLAIM_DATE_FIELDS:
                    if value is None or (
                        hasattr(value, "__class__")
                        and value.__class__.__name__ == "NaTType"
                    ):
                        kwargs[field_name] = None
                    elif isinstance(value, date):
                        kwargs[field_name] = value
                    else:
                        try:
                            kwargs[field_name] = pd.Timestamp(value).date()
                        except Exception:
                            kwargs[field_name] = None
                else:
                    try:
                        import math
                        if isinstance(value, float) and math.isnan(value):
                            kwargs[field_name] = None
                        else:
                            kwargs[field_name] = value
                    except (TypeError, ValueError):
                        kwargs[field_name] = value
            records.append(ClaimRecord(**kwargs))
        return cls(records)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all records to a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=[f.name for f in fields(ClaimRecord)])
        rows = [asdict(r) for r in self._records]
        return pd.DataFrame(rows)
