"""Lee-Carter mortality projection and improvement factor application."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from lifexp.core.tables import MortalityTable, TableRegistry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InsufficientDataError(Exception):
    """Raised when the mortality matrix has too few years to estimate a trend."""


# ---------------------------------------------------------------------------
# Lee-Carter model
# ---------------------------------------------------------------------------

class LeeCarter:
    """Lee-Carter (1992) age-period mortality model.

    The model decomposes log-mortality as::

        ln q_{x,t} = a_x + b_x * k_t + ε_{x,t}

    * ``a_x`` — age-specific constants (row means of the input matrix)
    * ``b_x`` — age-specific sensitivity to the mortality index (normalised
      so ``Σ b_x = 1``)
    * ``k_t`` — time-varying mortality index estimated via SVD rank-1
      approximation of the centred log-mortality matrix

    Parameters are not set until :meth:`fit` is called.

    Attributes set after fitting
    ----------------------------
    ax : pd.Series  — estimated a_x, indexed by age
    bx : pd.Series  — estimated b_x, indexed by age
    kt : pd.Series  — estimated k_t, indexed by calendar year
    """

    def fit(self, mortality_matrix: pd.DataFrame) -> "LeeCarter":
        """Estimate ax, bx, kt from a historical log-mortality matrix.

        Parameters
        ----------
        mortality_matrix:
            DataFrame with rows = age, columns = calendar year,
            values = ln(q_{x,t}).  Must contain at least **2** calendar
            years so that a kt drift can be estimated for projection.

        Returns
        -------
        self
            For method chaining.

        Raises
        ------
        InsufficientDataError
            If *mortality_matrix* has fewer than 2 columns.
        """
        if mortality_matrix.shape[1] < 2:
            raise InsufficientDataError(
                "Lee-Carter fitting requires at least 2 calendar years "
                f"to estimate a kt trend; got {mortality_matrix.shape[1]}."
            )

        ln_qx = mortality_matrix.to_numpy(dtype=float)  # (n_age, n_year)
        ages = mortality_matrix.index
        years = mortality_matrix.columns

        # Step 1: a_x = row means (average log-mortality over calendar years)
        ax = ln_qx.mean(axis=1)

        # Step 2: centred matrix Z = ln(q) - a_x
        Z = ln_qx - ax[:, np.newaxis]

        # Step 3: rank-1 SVD approximation of Z
        # Z ≈ σ₁ · u₁ · v₁ᵀ  →  b_x = u₁, k_t = σ₁ · v₁
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)

        bx_raw = U[:, 0]                 # first left singular vector
        kt_raw = s[0] * Vt[0, :]        # first right component scaled by σ₁

        # Step 4: normalise so Σ b_x = 1
        scale = bx_raw.sum()
        if abs(scale) < 1e-14:
            scale = 1.0
        bx = bx_raw / scale
        kt = kt_raw * scale

        self.ax = pd.Series(ax, index=ages, name="ax")
        self.bx = pd.Series(bx, index=ages, name="bx")
        self.kt = pd.Series(kt, index=years, name="kt")
        self._mortality_matrix = mortality_matrix
        return self

    # ------------------------------------------------------------------

    def project(
        self,
        n_years: int,
        kt_model: str = "arima",
    ) -> pd.DataFrame:
        """Project the mortality surface ``n_years`` beyond the fitted data.

        ``k_t`` is extrapolated using an ARIMA(0,1,0) random walk with drift::

            k_{T+h} = k_T + h · d̂    where  d̂ = (k_T − k_1) / (T − 1)

        Parameters
        ----------
        n_years:
            Number of years to project forward (must be ≥ 1).
        kt_model:
            Only ``'arima'`` (random walk with drift) is supported.

        Returns
        -------
        pd.DataFrame
            Rows = age, columns = projected calendar years,
            values = projected **q_x** (not log-mortality).
        """
        if n_years < 1:
            raise ValueError(f"n_years must be ≥ 1; got {n_years}.")
        if kt_model != "arima":
            raise ValueError(f"Unsupported kt_model {kt_model!r}; use 'arima'.")

        kt_arr = self.kt.to_numpy(dtype=float)
        years = self.kt.index.to_numpy()

        # Estimate drift from the fitted kt series
        T = len(kt_arr)
        drift = (kt_arr[-1] - kt_arr[0]) / (T - 1) if T > 1 else 0.0

        last_year = int(years[-1])
        proj_years = [last_year + h for h in range(1, n_years + 1)]
        proj_kt = kt_arr[-1] + np.arange(1, n_years + 1) * drift

        # Reconstruct projected ln(qx): (n_age, n_proj_year)
        # = a_x[:, None] + outer(b_x, k_t_proj)
        ax = self.ax.to_numpy()
        bx = self.bx.to_numpy()

        ln_qx_proj = ax[:, np.newaxis] + np.outer(bx, proj_kt)

        # Convert to q_x, clip to [0, 1]
        qx_proj = np.clip(np.exp(ln_qx_proj), 0.0, 1.0)

        return pd.DataFrame(
            qx_proj,
            index=self.ax.index,
            columns=proj_years,
        )

    # ------------------------------------------------------------------

    def life_expectancy(self, from_age: int, projected_year: int) -> float:
        """Curtate expectation of life e_x at *from_age* in *projected_year*.

        Uses the age range covered by the fitted mortality matrix.
        For years beyond the fitted data, mortality rates are obtained
        via :meth:`project`.

        Parameters
        ----------
        from_age:
            Starting age x.  Must be present in the fitted age range.
        projected_year:
            Calendar year whose mortality rates are used.

        Returns
        -------
        float
            e_x = Σ_{k=1}^{ω−x} ₖpₓ  (curtate expectation)
        """
        ages = self.ax.index.tolist()
        if from_age not in ages:
            raise ValueError(
                f"from_age {from_age} is not in the fitted age range "
                f"{ages[0]}–{ages[-1]}."
            )

        # Retrieve q_x for the requested year
        if projected_year in self._mortality_matrix.columns:
            # Historical year: use fitted matrix directly
            ln_qx = self._mortality_matrix[projected_year].to_numpy(dtype=float)
            qx_year = np.clip(np.exp(ln_qx), 0.0, 1.0)
        else:
            last_year = int(self.kt.index[-1])
            n = projected_year - last_year
            if n <= 0:
                raise ValueError(
                    f"projected_year {projected_year} is before the last "
                    f"fitted year {last_year}.  Only forward projection is "
                    "supported."
                )
            proj_df = self.project(n)
            qx_year = proj_df[projected_year].to_numpy(dtype=float)

        # Curtate life expectancy: e_x = Σ ₖpₓ for k = 1, 2, …
        # ₖpₓ = Π_{j=0}^{k-1} (1 − q_{x+j})
        start_idx = ages.index(from_age)
        px = 1.0 - qx_year[start_idx:]          # p_x, p_{x+1}, …
        survival = np.cumprod(px)                # ₁pₓ, ₂pₓ, …
        return float(survival.sum())


# ---------------------------------------------------------------------------
# Improvement factor application
# ---------------------------------------------------------------------------

def apply_improvement_factors(
    base_table: MortalityTable,
    factors: pd.DataFrame,
    base_year: int,
    target_year: int,
) -> MortalityTable:
    """Apply annual improvement factors to project a mortality table forward.

    The improvement model is::

        q_x(target) = q_x(base) × ∏_{t=base_year}^{target_year−1} (1 − r_{x,t})

    where ``r_{x,t}`` is the annual improvement rate for age *x* in year *t*.
    Missing (age, year) combinations in *factors* default to zero improvement.

    Parameters
    ----------
    base_table:
        Starting :class:`~lifexp.core.tables.MortalityTable`.
    factors:
        DataFrame with rows = age, columns = calendar year,
        values = annual improvement rate (e.g. 0.01 = 1 % improvement).
        Ages not present in *factors* receive zero improvement.
    base_year:
        Calendar year corresponding to *base_table*.
    target_year:
        Target calendar year (must be > *base_year*).

    Returns
    -------
    MortalityTable
        New table for *target_year* with the same age range as *base_table*.

    Raises
    ------
    ValueError
        If *target_year* ≤ *base_year*.
    """
    if target_year <= base_year:
        raise ValueError(
            f"target_year ({target_year}) must be strictly greater than "
            f"base_year ({base_year})."
        )

    ages = list(range(base_table.age_min, base_table.age_max + 1))

    # Initialise from base table
    qx = {age: base_table.qx(age) for age in ages}

    # Apply one year at a time
    for year in range(base_year, target_year):
        for age in ages:
            if year in factors.columns and age in factors.index:
                rate = float(factors.loc[age, year])
            else:
                rate = 0.0
            qx[age] = max(0.0, qx[age] * (1.0 - rate))

    # Build the new MortalityTable
    new_data = pd.DataFrame({
        "age": ages,
        "qx": [min(1.0, qx[a]) for a in ages],
    })

    return MortalityTable(
        name=f"{base_table.name}_{target_year}",
        age_min=base_table.age_min,
        age_max=base_table.age_max,
        data=new_data,
        table_type=base_table.table_type,
        basis=f"projected_{target_year}",
    )
