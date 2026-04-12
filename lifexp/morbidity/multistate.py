"""Healthy-Sick-Dead (HSD) multi-state Markov model for disability insurance."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from lifexp.morbidity.study import MorbidityResults
from lifexp.mortality.study import MortalityResults


# ---------------------------------------------------------------------------
# HSDModel
# ---------------------------------------------------------------------------

class HSDModel:
    """Three-state continuous-time Markov model for disability insurance.

    States
    ------
    H : Healthy (not on claim)
    S : Sick (on claim, deferred period elapsed)
    D : Dead (absorbing)

    Transition intensities
    ----------------------
    σ(x) : H → S  incidence rate at age x
    ρ(x) : S → H  recovery rate at age x
    μ(x) : H → D  healthy-lives mortality at age x
    ν(x) : S → D  sick-lives mortality at age x

    Each intensity is stored as a dict ``{integer_age: annual_rate}`` and
    looked up as a step function (rate for ``floor(age)``).

    Parameters
    ----------
    sigma, rho, mu, nu :
        Optional dicts mapping integer age → annual intensity.  Can also be
        populated by calling :meth:`fit`.
    ode_rtol, ode_atol :
        Tolerances passed to the ODE solver.  Tight defaults ensure the
        conservation law Σ P = 1 holds to better than 1 × 10⁻⁶.
    """

    def __init__(
        self,
        sigma: Optional[Dict[int, float]] = None,
        rho:   Optional[Dict[int, float]] = None,
        mu:    Optional[Dict[int, float]] = None,
        nu:    Optional[Dict[int, float]] = None,
        ode_rtol: float = 1e-8,
        ode_atol: float = 1e-10,
    ) -> None:
        self._sigma: Dict[int, float] = dict(sigma or {})
        self._rho:   Dict[int, float] = dict(rho   or {})
        self._mu:    Dict[int, float] = dict(mu    or {})
        self._nu:    Dict[int, float] = dict(nu    or {})
        self._ode_rtol = ode_rtol
        self._ode_atol = ode_atol

    # ------------------------------------------------------------------
    # Fitting from experience study results
    # ------------------------------------------------------------------

    def fit(
        self,
        morbidity_results: MorbidityResults,
        mortality_results: Optional[MortalityResults] = None,
    ) -> "HSDModel":
        """Populate transition intensities from experience study results.

        Parameters
        ----------
        morbidity_results :
            Output of :meth:`~lifexp.morbidity.study.MorbidityStudy.run`.
            Provides σ(x) from ``incidence_df`` and ρ(x)/ν(x) from
            ``termination_df`` (aggregated over claim duration).
        mortality_results :
            Output of :meth:`~lifexp.mortality.study.MortalityStudy.run`.
            Provides μ(x) from ``summary_df``.  If *None*, μ(x) = 0.

        Returns
        -------
        HSDModel
            ``self``, for chaining.
        """
        # σ(x) — incidence rate
        for _, row in morbidity_results.incidence_df.iterrows():
            age = int(row["age"])
            rate = float(row["incidence_rate"])
            self._sigma[age] = 0.0 if np.isnan(rate) else rate

        # ρ(x) and ν(x) — aggregate sick_etr over duration cells per age
        term = morbidity_results.termination_df
        if not term.empty:
            agg = (
                term.groupby("age", as_index=False)
                .agg(
                    sick_etr=("sick_etr", "sum"),
                    recoveries=("recoveries", "sum"),
                    deaths=("deaths", "sum"),
                )
            )
            for _, row in agg.iterrows():
                age = int(row["age"])
                etr = float(row["sick_etr"])
                self._rho[age] = float(row["recoveries"]) / etr if etr > 0 else 0.0
                self._nu[age]  = float(row["deaths"])     / etr if etr > 0 else 0.0

        # μ(x) — crude central mortality rate for healthy lives
        if mortality_results is not None:
            for _, row in mortality_results.summary_df.iterrows():
                age = int(row["age"])
                rate = float(row["crude_central_rate"])
                self._mu[age] = 0.0 if np.isnan(rate) else rate

        return self

    # ------------------------------------------------------------------
    # Intensity lookup (step function)
    # ------------------------------------------------------------------

    def _rate(self, d: Dict[int, float], age: float) -> float:
        """Step-function lookup: return intensity for floor(age), else 0.0."""
        return d.get(int(age), 0.0)

    def sigma(self, age: float) -> float:
        """Incidence rate σ(x) at *age*."""
        return self._rate(self._sigma, age)

    def rho(self, age: float) -> float:
        """Recovery rate ρ(x) at *age*."""
        return self._rate(self._rho, age)

    def mu(self, age: float) -> float:
        """Healthy-lives mortality μ(x) at *age*."""
        return self._rate(self._mu, age)

    def nu(self, age: float) -> float:
        """Sick-lives mortality ν(x) at *age*."""
        return self._rate(self._nu, age)

    # ------------------------------------------------------------------
    # Kolmogorov forward equations
    # ------------------------------------------------------------------

    def state_probabilities(
        self,
        age_from: int,
        age_to: int,
        initial_state: str = "H",
        n_steps: int = 1000,
    ) -> pd.DataFrame:
        """Solve the Kolmogorov forward equations numerically.

        The state vector P(t) = [P_xH(t), P_xS(t)] (x = initial state)
        satisfies the system::

            dP_xH/dt = -(σ + μ) P_xH + ρ P_xS
            dP_xS/dt =   σ P_xH − (ρ + ν) P_xS
            P_xD(t)  = 1 − P_xH(t) − P_xS(t)

        Parameters
        ----------
        age_from :
            Starting age (initial condition applied here).
        age_to :
            Final age.
        initial_state :
            ``'H'`` — start in healthy state (default).
            ``'S'`` — start in sick state.
        n_steps :
            Number of output time points.

        Returns
        -------
        pd.DataFrame
            Columns ``age``, ``P_HH``/``P_SH``, ``P_HS``/``P_SS``,
            ``P_HD``/``P_SD`` (prefix matches *initial_state*).
        """
        if initial_state == "H":
            y0 = [1.0, 0.0]
        elif initial_state == "S":
            y0 = [0.0, 1.0]
        else:
            raise ValueError(
                f"initial_state must be 'H' or 'S', got {initial_state!r}"
            )

        t_eval = np.linspace(float(age_from), float(age_to), n_steps + 1)

        def _rhs(t: float, y: list) -> list:
            p_h, p_s = y
            s = self.sigma(t)
            r = self.rho(t)
            m = self.mu(t)
            v = self.nu(t)
            return [
                -(s + m) * p_h + r * p_s,
                 s * p_h - (r + v) * p_s,
            ]

        sol = solve_ivp(
            _rhs,
            (float(age_from), float(age_to)),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=self._ode_rtol,
            atol=self._ode_atol,
        )

        p_h = sol.y[0]
        p_s = sol.y[1]
        p_d = 1.0 - p_h - p_s

        x = initial_state
        return pd.DataFrame({
            "age":   sol.t,
            f"P_{x}H": p_h,
            f"P_{x}S": p_s,
            f"P_{x}D": p_d,
        })

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def active_prevalence(self, age: int, age_from: Optional[int] = None) -> float:
        """Probability of being in the sick state at *age*, starting from *age_from*.

        Parameters
        ----------
        age :
            Target age.
        age_from :
            Starting age; defaults to ``min(sigma)`` if not supplied.

        Returns
        -------
        float
            P_HS(age_from → age).
        """
        if age_from is None:
            if not self._sigma:
                raise ValueError(
                    "No intensities set; call fit() or supply sigma dict."
                )
            age_from = min(self._sigma)
        if age <= age_from:
            return 0.0
        df = self.state_probabilities(age_from, age)
        return float(df["P_HS"].iloc[-1])

    def expected_claim_cost(
        self,
        age_from: int,
        age_to: int,
        benefit_pa: float,
        interest_rate: float,
        n_steps: int = 1000,
    ) -> float:
        """Expected present value of a disability annuity.

        Computes::

            EPV = ∫_{age_from}^{age_to} benefit_pa × P_HS(t) × e^{-δ(t − age_from)} dt

        where δ = *interest_rate* (continuous force of interest).

        Parameters
        ----------
        age_from :
            Start age (also the valuation date).
        age_to :
            End age (benefit ceases).
        benefit_pa :
            Annual benefit amount.
        interest_rate :
            Continuous force of interest (e.g., ``ln(1.05) ≈ 0.0488`` for a
            5 % effective rate).
        n_steps :
            ODE resolution.

        Returns
        -------
        float
            EPV of the disability annuity.
        """
        df = self.state_probabilities(
            age_from, age_to, initial_state="H", n_steps=n_steps
        )
        ages = df["age"].to_numpy()
        p_hs = df["P_HS"].to_numpy()
        discount = np.exp(-interest_rate * (ages - age_from))
        return float(np.trapezoid(benefit_pa * p_hs * discount, ages))
