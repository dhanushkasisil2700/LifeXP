"""Mortality table registry and interpolation utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgeOutOfRangeError(Exception):
    """Raised when an age lookup falls outside the table's defined range."""


class DataError(Exception):
    """Raised when a mortality table CSV contains invalid or incomplete data."""


# ---------------------------------------------------------------------------
# MortalityTable
# ---------------------------------------------------------------------------

@dataclass
class MortalityTable:
    """A single mortality table with lookup and interpolation methods.

    Attributes
    ----------
    name:
        Unique identifier used for registry lookups.
    age_min, age_max:
        Inclusive bounds of the table's age range.
    data:
        DataFrame with at least ``age`` and ``qx`` columns.
        Select-ultimate tables may also contain a ``select_year`` column.
    table_type:
        ``'ultimate'`` or ``'select_ultimate'``.
    basis:
        Source identifier, e.g. ``'A_1967_70'`` or ``'custom'``.
    """

    name: str
    age_min: int
    age_max: int
    data: pd.DataFrame = field(compare=False, repr=False)
    table_type: Literal["ultimate", "select_ultimate"] = "ultimate"
    basis: str = "custom"

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def qx(self, age: int, select_year: int = 0) -> float:
        """Return the annual mortality rate for an exact integer age.

        Parameters
        ----------
        age:
            Integer attained age.
        select_year:
            Duration since selection (only used for select-ultimate tables).

        Raises
        ------
        AgeOutOfRangeError
            If *age* is outside ``[age_min, age_max]``.
        """
        if age < self.age_min or age > self.age_max:
            raise AgeOutOfRangeError(
                f"Age {age} is out of range [{self.age_min}, {self.age_max}] "
                f"for table '{self.name}'."
            )
        if (
            self.table_type == "select_ultimate"
            and "select_year" in self.data.columns
            and select_year > 0
        ):
            mask = (self.data["age"] == age) & (self.data["select_year"] == select_year)
        else:
            mask = self.data["age"] == age

        rows = self.data.loc[mask]
        if rows.empty:
            raise AgeOutOfRangeError(
                f"Age {age} has no entry in table '{self.name}'."
            )
        return float(rows["qx"].iloc[0])

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        age: float,
        method: Literal["linear", "cubic"] = "linear",
    ) -> float:
        """Return an interpolated qx for a fractional age.

        Parameters
        ----------
        age:
            Age (may be fractional, e.g. 40.5).
        method:
            ``'linear'`` uses piecewise-linear interpolation (numpy.interp);
            ``'cubic'`` uses a natural cubic spline (scipy CubicSpline).

        Raises
        ------
        AgeOutOfRangeError
            If *age* is outside ``[age_min, age_max]``.
        """
        if age < self.age_min or age > self.age_max:
            raise AgeOutOfRangeError(
                f"Age {age:.4f} is out of range [{self.age_min}, {self.age_max}] "
                f"for table '{self.name}'."
            )
        ages_arr = self.data["age"].to_numpy(dtype=float)
        qx_arr = self.data["qx"].to_numpy(dtype=float)

        if method == "linear":
            return float(np.interp(age, ages_arr, qx_arr))
        if method == "cubic":
            cs = CubicSpline(ages_arr, qx_arr)
            return float(cs(float(age)))
        raise ValueError(f"Unknown interpolation method '{method}'. Use 'linear' or 'cubic'.")

    # ------------------------------------------------------------------
    # Force of mortality
    # ------------------------------------------------------------------

    def mu(self, age: float) -> float:
        """Return the force of mortality at *age* (constant-force approximation).

        Uses linear interpolation of qx then applies mu ≈ −ln(1 − qx).
        """
        qx_val = self.interpolate(age, method="linear")
        # Guard against qx ≥ 1 (e.g., at very old ages) to prevent log(0)
        qx_val = min(qx_val, 1.0 - 1e-10)
        return float(-np.log(1.0 - qx_val))


# ---------------------------------------------------------------------------
# TableRegistry
# ---------------------------------------------------------------------------

class TableRegistry:
    """Registry that maps table names to :class:`MortalityTable` instances.

    Each :class:`TableRegistry` instance starts pre-populated with built-in
    tables (currently ``'A_1967_70'``).  Calling :meth:`register` adds or
    replaces tables on that specific instance; built-in tables in other
    instances are not affected.
    """

    def __init__(self) -> None:
        # Each instance receives an independent copy of the shared built-ins.
        self._tables: Dict[str, MortalityTable] = dict(_BUILTIN_TABLES)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, table: MortalityTable) -> None:
        """Add *table* to this registry.

        If a table with the same name already exists a :class:`UserWarning`
        is issued and the old table is replaced.
        """
        if table.name in self._tables:
            warnings.warn(
                f"Table '{table.name}' already exists in this registry; overwriting.",
                UserWarning,
                stacklevel=2,
            )
        self._tables[table.name] = table

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> MortalityTable:
        """Return the table registered under *name*.

        Raises
        ------
        KeyError
            If no table with that name exists.
        """
        if name not in self._tables:
            raise KeyError(
                f"Table '{name}' not found. "
                f"Available tables: {self.list_tables()}"
            )
        return self._tables[name]

    def list_tables(self) -> List[str]:
        """Return a sorted list of registered table names."""
        return sorted(self._tables.keys())

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_csv(
        path: str | Path,
        name: str,
        table_type: Literal["ultimate", "select_ultimate"] = "ultimate",
        basis: str = "custom",
    ) -> MortalityTable:
        """Create a :class:`MortalityTable` from a CSV file.

        The CSV must contain ``age`` and ``qx`` columns.  Ages must cover a
        contiguous range with no gaps; any missing ages raise
        :class:`DataError` listing the problematic ages.

        Parameters
        ----------
        path:
            File-system path to the CSV.
        name:
            Name to assign to the resulting table.
        table_type:
            ``'ultimate'`` or ``'select_ultimate'``.
        basis:
            Source identifier string.

        Returns
        -------
        MortalityTable
            The loaded table (not yet registered; call
            :meth:`register` to add it to a registry).

        Raises
        ------
        DataError
            If required columns are missing or the age range has gaps.
        """
        df = pd.read_csv(path)

        missing_cols = {"age", "qx"} - set(df.columns)
        if missing_cols:
            raise DataError(
                f"CSV '{path}' is missing required columns: {sorted(missing_cols)}"
            )

        df = df.copy()
        df["age"] = df["age"].astype(int)

        if df.empty:
            raise DataError(f"CSV '{path}' contains no data rows.")

        ages = sorted(df["age"].tolist())
        age_min, age_max = ages[0], ages[-1]
        expected = set(range(age_min, age_max + 1))
        missing_ages = sorted(expected - set(ages))
        if missing_ages:
            raise DataError(
                f"CSV '{path}' has gaps in the age range {age_min}–{age_max}. "
                f"Missing ages: {missing_ages}"
            )

        df = df.sort_values("age").reset_index(drop=True)
        return MortalityTable(
            name=name,
            age_min=age_min,
            age_max=age_max,
            data=df,
            table_type=table_type,
            basis=basis,
        )


# ---------------------------------------------------------------------------
# Built-in table bootstrap  (runs once at module import)
# ---------------------------------------------------------------------------

_BUILTIN_TABLES: Dict[str, MortalityTable] = {}

def _bootstrap_builtins() -> None:
    """Load built-in tables into _BUILTIN_TABLES at import time."""
    csv_path = Path(__file__).resolve().parent.parent / "datasets" / "a_1967_70.csv"
    try:
        table = TableRegistry.load_from_csv(csv_path, "A_1967_70", "ultimate", "A_1967_70")
        _BUILTIN_TABLES["A_1967_70"] = table
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Could not load built-in table 'A_1967_70' from {csv_path}: {exc}",
            RuntimeWarning,
            stacklevel=1,
        )


_bootstrap_builtins()
