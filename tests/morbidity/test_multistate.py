"""Tests for HSDModel in lifexp.morbidity.multistate.

Checkpoint S15: the sum-to-1 conservation law P_HH + P_HS + P_HD = 1 at
every time step validates the Kolmogorov ODE solver numerically.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from lifexp.morbidity.multistate import HSDModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_model(
    sigma: float,
    rho: float,
    mu: float = 0.002,
    nu: float = 0.005,
    ages: range = range(40, 65),
    **kwargs,
) -> HSDModel:
    """Create an HSDModel with flat (age-independent) transition intensities."""
    return HSDModel(
        sigma={a: sigma for a in ages},
        rho={a: rho   for a in ages},
        mu={a: mu     for a in ages},
        nu={a: nu     for a in ages},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1 — State probabilities sum to 1.0  (Checkpoint S15)
# ---------------------------------------------------------------------------

class TestSumToOne:
    """P_HH + P_HS + P_HD = 1 at every age step."""

    AGE_FROM = 40
    AGE_TO   = 64

    @pytest.fixture(scope="class")
    def df(self):
        model = _flat_model(sigma=0.05, rho=0.4, mu=0.003, nu=0.008)
        return model.state_probabilities(self.AGE_FROM, self.AGE_TO)

    def test_columns_present(self, df):
        assert {"age", "P_HH", "P_HS", "P_HD"} <= set(df.columns)

    def test_sum_equals_one_everywhere(self, df):
        """Checkpoint S15: conservation law of the ODE system."""
        row_sums = df["P_HH"] + df["P_HS"] + df["P_HD"]
        deviation = np.abs(row_sums - 1.0)
        assert (deviation < 1e-6).all(), (
            f"Max deviation from 1.0: {deviation.max():.3e}"
        )

    def test_no_negative_probabilities(self, df):
        assert (df["P_HH"] >= -1e-10).all()
        assert (df["P_HS"] >= -1e-10).all()
        assert (df["P_HD"] >= -1e-10).all()

    def test_p_hd_non_decreasing(self, df):
        """Dead state is absorbing — its probability must not decrease."""
        diffs = np.diff(df["P_HD"].to_numpy())
        assert (diffs >= -1e-8).all()


# ---------------------------------------------------------------------------
# Test 2 — Initial condition: P_HH(0) = 1.0
# ---------------------------------------------------------------------------

class TestInitialCondition:
    """At t = age_from the state vector must equal (1, 0, 0)."""

    @pytest.fixture(scope="class")
    def df(self):
        model = _flat_model(sigma=0.05, rho=0.3)
        return model.state_probabilities(40, 60)

    def test_phh_at_start_is_one(self, df):
        assert df["P_HH"].iloc[0] == pytest.approx(1.0, abs=1e-10)

    def test_phs_at_start_is_zero(self, df):
        assert df["P_HS"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_phd_at_start_is_zero(self, df):
        assert df["P_HD"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_first_age_matches_age_from(self, df):
        assert df["age"].iloc[0] == pytest.approx(40.0)

    def test_last_age_matches_age_to(self, df):
        assert df["age"].iloc[-1] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Test 3 — Zero incidence → no sick state (degenerate to pure mortality)
# ---------------------------------------------------------------------------

class TestZeroIncidence:
    """σ(x) = 0 → P_HS = 0 everywhere; model reduces to pure H/D survival."""

    @pytest.fixture(scope="class")
    def df(self):
        model = _flat_model(sigma=0.0, rho=0.4, mu=0.003, nu=0.008)
        return model.state_probabilities(40, 64)

    def test_p_hs_is_zero_everywhere(self, df):
        assert (np.abs(df["P_HS"]) < 1e-10).all()

    def test_phh_plus_phd_equals_one(self, df):
        total = df["P_HH"] + df["P_HD"]
        assert (np.abs(total - 1.0) < 1e-8).all()

    def test_phh_decreases_monotonically(self, df):
        """Without recovery, survival probability must fall with age."""
        diffs = np.diff(df["P_HH"].to_numpy())
        assert (diffs <= 1e-10).all()

    def test_sum_to_one_still_holds(self, df):
        row_sums = df["P_HH"] + df["P_HS"] + df["P_HD"]
        assert (np.abs(row_sums - 1.0) < 1e-8).all()


# ---------------------------------------------------------------------------
# Test 4 — EPV decreases with higher interest rate
# ---------------------------------------------------------------------------

class TestEPVInterestRate:
    """EPV(5 %) < EPV(3 %) for the same benefit stream."""

    MODEL = _flat_model(sigma=0.05, rho=0.4, mu=0.003, nu=0.008)
    AGE_FROM = 40
    AGE_TO   = 65
    BENEFIT  = 1.0

    def test_epv_lower_at_higher_rate(self):
        epv_low  = self.MODEL.expected_claim_cost(
            self.AGE_FROM, self.AGE_TO, self.BENEFIT, interest_rate=0.03
        )
        epv_high = self.MODEL.expected_claim_cost(
            self.AGE_FROM, self.AGE_TO, self.BENEFIT, interest_rate=0.05
        )
        assert epv_high < epv_low

    def test_epv_positive(self):
        epv = self.MODEL.expected_claim_cost(
            self.AGE_FROM, self.AGE_TO, self.BENEFIT, interest_rate=0.04
        )
        assert epv > 0.0

    def test_epv_scales_with_benefit(self):
        epv_1 = self.MODEL.expected_claim_cost(
            self.AGE_FROM, self.AGE_TO, 1.0, interest_rate=0.04
        )
        epv_2 = self.MODEL.expected_claim_cost(
            self.AGE_FROM, self.AGE_TO, 2.0, interest_rate=0.04
        )
        assert epv_2 == pytest.approx(2.0 * epv_1, rel=1e-6)

    def test_epv_strictly_decreases_with_interest(self):
        """Check monotonicity over a wider grid of rates."""
        rates = [0.01, 0.02, 0.04, 0.06, 0.10]
        epvs = [
            self.MODEL.expected_claim_cost(
                self.AGE_FROM, self.AGE_TO, self.BENEFIT, interest_rate=r
            )
            for r in rates
        ]
        assert all(epvs[i] > epvs[i + 1] for i in range(len(epvs) - 1))


# ---------------------------------------------------------------------------
# Test 5 — Very high recovery rate → P_HS ≈ 0 (model stability)
# ---------------------------------------------------------------------------

class TestHighRecoveryRate:
    """ρ = 100 (near-instant recovery): P_HS stays tiny; solver remains stable."""

    # At near-instant recovery the steady-state sick fraction is:
    #   P_HS_ss ≈ σ / (σ + ρ + ν) ≈ 0.05 / 100.06 ≈ 5 × 10⁻⁴
    AGE_FROM = 40
    AGE_TO   = 60

    @pytest.fixture(scope="class")
    def df(self):
        model = _flat_model(sigma=0.05, rho=100.0, mu=0.003, nu=0.008)
        return model.state_probabilities(self.AGE_FROM, self.AGE_TO)

    def test_p_hs_near_zero(self, df):
        """P_HS well below 0.1 % at all times."""
        assert (df["P_HS"] < 1e-3).all()

    def test_p_hs_non_negative(self, df):
        assert (df["P_HS"] >= -1e-8).all()

    def test_sum_to_one_stable(self, df):
        """Conservation law holds even under near-instant recovery."""
        row_sums = df["P_HH"] + df["P_HS"] + df["P_HD"]
        deviation = np.abs(row_sums - 1.0)
        assert (deviation < 1e-6).all(), (
            f"Max deviation from 1.0 with ρ=100: {deviation.max():.3e}"
        )

    def test_model_does_not_diverge(self, df):
        """Probabilities remain in [0, 1]; no numerical blow-up."""
        assert (df[["P_HH", "P_HS", "P_HD"]].values <= 1.0 + 1e-8).all()
