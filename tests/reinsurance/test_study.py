"""Tests for RIStudy and RIResults (lifexp.reinsurance.study).

Checkpoint S16: cost_sensitivity is verified to be linear for ±5 % shifts.
break_even_mortality is validated by substituting it back into the cost
formula and confirming loss_ratio = 1.0.
"""

from __future__ import annotations

from datetime import date
from typing import List

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import ClaimDataset, PolicyDataset
from lifexp.core.study_period import StudyPeriod
from lifexp.core.tables import MortalityTable
from lifexp.reinsurance.study import RIStudy


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Study: 2021-01-01 to 2021-12-31 (365 days, non-leap year).
# DOB = 1971-01-01 → age 50 throughout; no birthday strictly inside the window
# (anniversary is exactly obs_start so not a boundary by the strict-< rule).
# Issue = 2020-01-01 → anniversary at 2021-01-01 = obs_start, not a boundary.
_STUDY = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
_DOB   = date(1971, 1, 1)
_ISSUE = date(2020, 1, 1)
_SA    = 100_000.0
_RETENTION = 0.5         # cede 50 % → NAAR = 50 000
_NAAR  = _SA * (1.0 - _RETENTION)   # 50 000
_RI_RATE = 0.05          # flat rate at age 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ri_table(rate: float = _RI_RATE) -> MortalityTable:
    ages = list(range(40, 61))
    return MortalityTable(
        name="test_ri",
        age_min=40,
        age_max=60,
        data=pd.DataFrame({"age": ages, "qx": [rate] * len(ages)}),
    )


def _pol(pid: str, status: str = "IF", exit_date=None) -> dict:
    return {
        "policy_id":      pid,
        "date_of_birth":  _DOB,
        "issue_date":     _ISSUE,
        "gender":         "M",
        "smoker_status":  "NS",
        "sum_assured":    _SA,
        "annual_premium": 100.0,
        "product_code":   "TERM",
        "channel":        "DIRECT",
        "status":         status,
        "exit_date":      exit_date,
        "exit_reason":    status if status != "IF" else None,
    }


def _clm(cid: str, pid: str, claim_date: date, amount: float = _NAAR) -> dict:
    return {
        "claim_id":          cid,
        "policy_id":         pid,
        "claim_start_date":  claim_date,
        "claim_end_date":    claim_date,
        "claim_status":      "CLOSED_DEATH",
        "benefit_type":      "LUMP_SUM",
        "claim_amount":      amount,
        "benefit_period_days": None,
    }


def _pds(rows: List[dict]) -> PolicyDataset:
    return PolicyDataset.from_dataframe(pd.DataFrame(rows))


def _cds(rows: List[dict]) -> ClaimDataset:
    if not rows:
        empty = pd.DataFrame(columns=[
            "claim_id", "policy_id", "claim_start_date", "claim_end_date",
            "claim_status", "benefit_type", "claim_amount", "benefit_period_days",
        ])
        return ClaimDataset.from_dataframe(empty)
    return ClaimDataset.from_dataframe(pd.DataFrame(rows))


def _ri_study(pols, clms, retention=_RETENTION, ri_rate=_RI_RATE):
    return RIStudy(
        _pds(pols),
        _cds(clms),
        _STUDY,
        _make_ri_table(ri_rate),
        treaty_type="YRT",
        retention_rate=retention,
    ).run()


# ETR for a policy running the full study (365 / 365.25)
_FULL_ETR = 365 / 365.25


# ---------------------------------------------------------------------------
# Test 1 — A/E ≈ 1.0 when experience matches RI rates
# ---------------------------------------------------------------------------

class TestAEEqualsOne:
    """With n_deaths ≈ ri_rate × n_policies, ae_ratio ≈ 1.0."""

    # 5 deaths at year-end (all policies still contribute full ETR).
    N_TOTAL  = 100
    N_DEATHS = 5   # ≈ 0.05 × 100 × 0.9993 year exposure

    @pytest.fixture(scope="class")
    def results(self):
        death_date = date(2021, 12, 31)   # last day → full-year ETR
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        return _ri_study(pols, [])

    def test_overall_ae_near_one(self, results):
        """Aggregate A/E ≈ 1.0 (deviation < 0.2 % from ETR calendar effect)."""
        df = results.ae_by_age()
        total_actual   = df["actual_ri_claims"].sum()
        total_expected = df["expected_ri_claims"].sum()
        overall_ae = total_actual / total_expected
        assert overall_ae == pytest.approx(1.0, abs=0.002)

    def test_deaths_attributed_correctly(self, results):
        df = results.ae_by_age()
        assert df["deaths"].sum() == pytest.approx(self.N_DEATHS, abs=0.5)

    def test_actual_claims_equals_n_deaths_times_naar(self, results):
        df = results.ae_by_age()
        assert df["actual_ri_claims"].sum() == pytest.approx(
            self.N_DEATHS * _NAAR, rel=1e-6
        )

    def test_expected_claims_formula(self, results):
        """expected = ri_rate × naar_etr = ri_rate × NAAR × n × etr_per_policy."""
        df = results.ae_by_age()
        expected_manual = _RI_RATE * self.N_TOTAL * _NAAR * _FULL_ETR
        assert df["expected_ri_claims"].sum() == pytest.approx(expected_manual, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 2 — Cost sensitivity is linear  (Checkpoint S16a)
# ---------------------------------------------------------------------------

class TestCostSensitivity:
    """±delta % shift in mortality → ±delta % change in RI cost; exactly linear."""

    N_TOTAL  = 100
    N_DEATHS = 5

    @pytest.fixture(scope="class")
    def results(self):
        death_date = date(2021, 12, 31)
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        return _ri_study(pols, [])

    def test_positive_shift_increases_cost(self, results):
        df = results.cost_sensitivity(10.0)
        base = results.ae_by_age()["expected_ri_claims"].sum()
        shocked = df["shocked_cost"].sum()
        assert shocked == pytest.approx(1.10 * base, rel=1e-6)

    def test_negative_shift_decreases_cost(self, results):
        df = results.cost_sensitivity(-10.0)
        base = results.ae_by_age()["expected_ri_claims"].sum()
        shocked = df["shocked_cost"].sum()
        assert shocked == pytest.approx(0.90 * base, rel=1e-6)

    def test_linearity_positive_5pct(self, results):
        """Checkpoint S16: +5 % shift → cost scales exactly by 1.05."""
        df5  = results.cost_sensitivity(5.0)
        df10 = results.cost_sensitivity(10.0)
        ratio = df10["shocked_cost"].sum() / df5["shocked_cost"].sum()
        # shocked_10 / shocked_5 = (1.10 × base) / (1.05 × base) = 1.10/1.05
        assert ratio == pytest.approx(1.10 / 1.05, rel=1e-6)

    def test_linearity_plus_minus_symmetric(self, results):
        """±delta produces equal-magnitude but opposite-sign delta_cost."""
        dfp = results.cost_sensitivity(5.0)
        dfm = results.cost_sensitivity(-5.0)
        assert dfp["delta_cost"].sum() == pytest.approx(
            -dfm["delta_cost"].sum(), rel=1e-6
        )

    def test_zero_shift_is_identity(self, results):
        df = results.cost_sensitivity(0.0)
        assert (df["shocked_cost"] == df["expected_ri_claims"]).all()

    def test_delta_cost_column(self, results):
        df = results.cost_sensitivity(10.0)
        base = results.ae_by_age()["expected_ri_claims"].sum()
        assert df["delta_cost"].sum() == pytest.approx(0.10 * base, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 3 — Loss ratio < 1.0 when profitable
# ---------------------------------------------------------------------------

class TestLossRatioProfitable:
    """Fewer deaths than expected → loss ratio < 1.0."""

    N_TOTAL  = 100
    N_DEATHS = 3   # below expected ≈ 5

    @pytest.fixture(scope="class")
    def results(self):
        death_date = date(2021, 12, 31)
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        return _ri_study(pols, [])

    def test_loss_ratio_below_one(self, results):
        assert results.loss_ratio() < 1.0

    def test_loss_ratio_positive(self, results):
        assert results.loss_ratio() > 0.0

    def test_ae_below_one(self, results):
        df = results.ae_by_age()
        ae = df["actual_ri_claims"].sum() / df["expected_ri_claims"].sum()
        assert ae < 1.0

    def test_loss_ratio_equals_overall_ae(self, results):
        """Under risk-premium pricing, loss_ratio = overall A/E."""
        df = results.ae_by_age()
        ae = df["actual_ri_claims"].sum() / df["expected_ri_claims"].sum()
        assert results.loss_ratio() == pytest.approx(ae, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 4 — Zero claims: ae = 0.0, loss_ratio = 0.0
# ---------------------------------------------------------------------------

class TestZeroClaims:
    """No deaths in study period — all RI metrics degenerate gracefully."""

    N_TOTAL = 100

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:03d}") for i in range(self.N_TOTAL)]
        return _ri_study(pols, [])

    def test_deaths_zero(self, results):
        assert results.summary_df["deaths"].sum() == 0

    def test_actual_claims_zero(self, results):
        assert results.summary_df["actual_ri_claims"].sum() == 0.0

    def test_ae_ratio_zero(self, results):
        df = results.ae_by_age()
        assert (df["ae_ratio"] == 0.0).all()

    def test_loss_ratio_zero(self, results):
        assert results.loss_ratio() == 0.0

    def test_expected_claims_positive(self, results):
        """Exposure is non-zero even with no deaths."""
        assert results.summary_df["expected_ri_claims"].sum() > 0.0

    def test_no_exception_in_all_methods(self, results):
        ri_table = _make_ri_table()
        results.ae_by_age()
        results.ae_by_treaty()
        results.loss_ratio()
        results.implied_mortality()
        results.cost_sensitivity(5.0)
        results.break_even_mortality()


# ---------------------------------------------------------------------------
# Test 5 — 100% retention: NAAR = 0, RI exposure is nil
# ---------------------------------------------------------------------------

class TestFullRetention:
    """retention_rate = 1.0 → no cession, RI amounts all zero."""

    N_TOTAL  = 100
    N_DEATHS = 5

    @pytest.fixture(scope="class")
    def results(self):
        death_date = date(2021, 12, 31)
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        return _ri_study(pols, [], retention=1.0)

    def test_naar_etr_zero(self, results):
        assert results.summary_df["naar_etr"].sum() == 0.0

    def test_actual_ri_claims_zero(self, results):
        assert results.summary_df["actual_ri_claims"].sum() == 0.0

    def test_expected_ri_claims_zero(self, results):
        assert results.summary_df["expected_ri_claims"].sum() == 0.0

    def test_loss_ratio_zero(self, results):
        assert results.loss_ratio() == 0.0

    def test_ae_ratio_nan(self, results):
        """With zero expected, ae_ratio should be NaN (not an error)."""
        df = results.ae_by_age()
        assert df["ae_ratio"].isna().all()

    def test_zero_retention_equals_direct_mortality_ae(self):
        """Checkpoint S16: with 0% retention (cede all), RI ae_ratio equals
        the direct mortality A/E (actual / expected at full SA)."""
        death_date = date(2021, 12, 31)
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        # 0% retention: NAAR = SA = 100 000
        ri_res = _ri_study(pols, [], retention=0.0)
        df = ri_res.ae_by_age()
        ri_ae = df["actual_ri_claims"].sum() / df["expected_ri_claims"].sum()

        # Direct calculation: n_deaths × SA / (ri_rate × n_total × SA × etr)
        direct_ae = (self.N_DEATHS * _SA) / (_RI_RATE * self.N_TOTAL * _SA * _FULL_ETR)
        assert ri_ae == pytest.approx(direct_ae, rel=1e-6)


# ---------------------------------------------------------------------------
# Checkpoint S16 — Break-even mortality validates to loss_ratio = 1.0
# ---------------------------------------------------------------------------

class TestBreakEvenMortality:
    """Substituting break_even_mortality() back gives loss_ratio = 1.0."""

    N_TOTAL  = 100
    N_DEATHS = 7   # some arbitrary count (not equal to expected ≈ 5)

    @pytest.fixture(scope="class")
    def results(self):
        death_date = date(2021, 12, 31)
        pols = (
            [_pol(f"D{i:02d}", "DEATH", death_date) for i in range(self.N_DEATHS)]
            + [_pol(f"P{i:03d}") for i in range(self.N_TOTAL - self.N_DEATHS)]
        )
        return _ri_study(pols, [])

    def test_break_even_definition(self, results):
        """q_be = actual_claims / naar_etr."""
        q_be = results.break_even_mortality()
        total_claims   = results.summary_df["actual_ri_claims"].sum()
        total_naar_etr = results.summary_df["naar_etr"].sum()
        assert q_be == pytest.approx(total_claims / total_naar_etr, rel=1e-9)

    def test_substitution_gives_loss_ratio_one(self, results):
        """Checkpoint S16: substituting q_be as RI rate → loss_ratio = 1.0."""
        q_be = results.break_even_mortality()
        total_claims   = results.summary_df["actual_ri_claims"].sum()
        total_naar_etr = results.summary_df["naar_etr"].sum()
        # If we reprice at q_be, earned_premium_new = q_be × naar_etr = actual_claims
        new_earned_premium = q_be * total_naar_etr
        new_loss_ratio = total_claims / new_earned_premium
        assert new_loss_ratio == pytest.approx(1.0, abs=1e-8)

    def test_break_even_above_ri_rate_when_ae_above_one(self, results):
        """With 7 deaths vs expected ≈ 5, q_be > contracted RI rate."""
        q_be = results.break_even_mortality()
        assert q_be > _RI_RATE

    def test_break_even_zero_when_no_claims(self):
        pols = [_pol(f"P{i:03d}") for i in range(50)]
        res = _ri_study(pols, [])
        assert res.break_even_mortality() == 0.0
