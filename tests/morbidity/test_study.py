"""Tests for MorbidityStudy and MorbidityResults (lifexp.morbidity.study)."""

from __future__ import annotations

from datetime import date
from typing import List

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import ClaimDataset, PolicyDataset
from lifexp.core.study_period import StudyPeriod
from lifexp.core.tables import MortalityTable
from lifexp.morbidity.study import MorbidityStudy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_table(rate: float, age_min: int = 35, age_max: int = 70) -> MortalityTable:
    """Flat incidence-rate table for testing."""
    ages = list(range(age_min, age_max + 1))
    return MortalityTable(
        name="test_flat",
        age_min=age_min,
        age_max=age_max,
        data=pd.DataFrame({"age": ages, "qx": [rate] * len(ages)}),
    )


def _pol(
    pid: str,
    dob: date = date(1970, 1, 1),
    issue: date = date(2019, 1, 1),
    status: str = "IF",
    exit_date=None,
    sa: float = 100_000.0,
) -> dict:
    return {
        "policy_id": pid,
        "date_of_birth": dob,
        "issue_date": issue,
        "gender": "M",
        "smoker_status": "NS",
        "sum_assured": sa,
        "annual_premium": 100.0,
        "product_code": "IP",
        "channel": "DIRECT",
        "status": status,
        "exit_date": exit_date,
        "exit_reason": status if status != "IF" else None,
    }


def _clm(
    cid: str,
    pid: str,
    start: date,
    end: date | None = None,
    amount: float = 1_000.0,
) -> dict:
    return {
        "claim_id": cid,
        "policy_id": pid,
        "claim_start_date": start,
        "claim_end_date": end,
        "claim_status": "OPEN" if end is None else "CLOSED_RECOVERY",
        "benefit_type": "PERIODIC",
        "claim_amount": amount,
        "benefit_period_days": None,
    }


def _pds(rows: List[dict]) -> PolicyDataset:
    return PolicyDataset.from_dataframe(pd.DataFrame(rows))


def _cds(rows: List[dict]) -> ClaimDataset:
    if not rows:
        empty = pd.DataFrame(
            columns=[
                "claim_id", "policy_id", "claim_start_date", "claim_end_date",
                "claim_status", "benefit_type", "claim_amount", "benefit_period_days",
            ]
        )
        return ClaimDataset.from_dataframe(empty)
    return ClaimDataset.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Test 1 — Incidence rate ≈ 0.05
# ---------------------------------------------------------------------------

class TestIncidenceRate:
    """50 new claims out of ~1 000 healthy person-years gives rate ≈ 0.05."""

    # Study: 2020 (non-leap, 365 days).  All policies have DOB 1970-01-01 so
    # no birthday boundary fires inside the window → single age cell (age 50).
    STUDY = StudyPeriod(date(2020, 1, 1), date(2020, 12, 31))
    DOB = date(1970, 1, 1)
    ISSUE = date(2019, 1, 1)

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:04d}", self.DOB, self.ISSUE) for i in range(1_000)]
        # 50 policies each have a 1-day claim on 2020-06-01
        clms = [
            _clm(f"C{i:04d}", f"P{i:04d}", date(2020, 6, 1), date(2020, 6, 1))
            for i in range(50)
        ]
        return MorbidityStudy(_pds(pols), _cds(clms), self.STUDY).run()

    def test_new_claims_count(self, results):
        assert results.incidence_df["new_claims"].sum() == pytest.approx(50, abs=0.5)

    def test_single_age_cell(self, results):
        assert len(results.incidence_df) == 1

    def test_incidence_rate(self, results):
        # 50 claims / ~999 healthy person-years ≈ 0.05004
        rate = results.incidence_df["incidence_rate"].iloc[0]
        assert rate == pytest.approx(0.05, abs=0.005)


# ---------------------------------------------------------------------------
# Test 2 — Simple cost A/E ≈ 1.0
# ---------------------------------------------------------------------------

class TestAECostSimple:
    """When n_claims ≈ ri_rate × healthy_etr, ae_cost_simple should be ≈ 1.0."""

    STUDY = StudyPeriod(date(2020, 1, 1), date(2020, 12, 31))
    DOB = date(1970, 1, 1)
    ISSUE = date(2019, 1, 1)
    MEAN_SA = 100_000.0
    RI_RATE = 0.05

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:03d}", self.DOB, self.ISSUE, sa=self.MEAN_SA) for i in range(100)]
        # 5 policies with 1-day claim; claim_amount = mean_sa so each costs exactly mean_sa
        clms = [
            _clm(f"C{i:03d}", f"P{i:03d}", date(2020, 1, 1), date(2020, 1, 1), amount=self.MEAN_SA)
            for i in range(5)
        ]
        return MorbidityStudy(_pds(pols), _cds(clms), self.STUDY).run()

    def test_ae_cost_simple_near_one(self, results):
        ri_table = _make_flat_table(self.RI_RATE, 40, 60)
        df = results.ae_cost_ratio(ri_table, mean_sa=self.MEAN_SA)
        # Single age cell (age 50); 5 claims at exactly ri_rate × healthy_etr × mean_sa
        ae = df["ae_cost_simple"].iloc[0]
        assert ae == pytest.approx(1.0, abs=0.02)

    def test_expected_cost_formula_used(self, results):
        """expected_cost_simple = ri_rate × healthy_etr × mean_sa."""
        ri_table = _make_flat_table(self.RI_RATE, 40, 60)
        df = results.ae_cost_ratio(ri_table, mean_sa=self.MEAN_SA)
        # expected_cost_simple column should be positive (non-zero denominator)
        assert (df["expected_cost_simple"] > 0).all()


# ---------------------------------------------------------------------------
# Test 3 — Duration-adjusted A/E > simple cost A/E  (Checkpoint S14)
# ---------------------------------------------------------------------------

class TestDurationAdjustedAE:
    """Checkpoint S14: ae_duration_adjusted > ae_cost_simple when claims
    last longer than the assumed claim duration."""

    # 4-year study so that 2-year claims are fully observed.
    STUDY = StudyPeriod(date(2020, 1, 1), date(2023, 12, 31))
    DOB = date(1980, 1, 1)   # ages 40-43 during the study
    ISSUE = date(2019, 1, 1)
    MEAN_SA = 1_000.0
    RI_RATE = 0.05
    ASSUMED_DURATION = 1.0   # 1-year assumed; actual claims last ~2 years

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:03d}", self.DOB, self.ISSUE, sa=self.MEAN_SA) for i in range(200)]
        # 50 policies have claims spanning exactly 2 calendar years
        clms = [
            _clm(
                f"C{i:03d}", f"P{i:03d}",
                date(2020, 1, 1), date(2021, 12, 31),
                amount=self.MEAN_SA,
            )
            for i in range(50)
        ]
        return MorbidityStudy(_pds(pols), _cds(clms), self.STUDY).run()

    def test_duration_adjusted_exceeds_simple(self, results):
        """Aggregate ae_duration_adjusted must be strictly greater than ae_cost_simple."""
        total_actual_cost = results._claim_costs_df["actual_cost"].sum()
        total_healthy_etr = results._ae_base_df["healthy_etr"].sum()
        total_sick_etr = results._ae_base_df["sick_etr"].sum()

        denom = self.RI_RATE * total_healthy_etr
        overall_ae_simple = total_actual_cost / (denom * self.MEAN_SA)
        overall_ae_duration = total_sick_etr / (denom * self.ASSUMED_DURATION)

        assert overall_ae_duration > overall_ae_simple

    def test_duration_ratio_approximately_two(self, results):
        """With 2-year claims and 1-year assumed duration the ratio should be ~2."""
        total_actual_cost = results._claim_costs_df["actual_cost"].sum()
        total_healthy_etr = results._ae_base_df["healthy_etr"].sum()
        total_sick_etr = results._ae_base_df["sick_etr"].sum()

        denom = self.RI_RATE * total_healthy_etr
        overall_ae_simple = total_actual_cost / (denom * self.MEAN_SA)
        overall_ae_duration = total_sick_etr / (denom * self.ASSUMED_DURATION)

        assert overall_ae_duration == pytest.approx(2.0 * overall_ae_simple, rel=0.1)

    def test_ae_cost_ratio_runs_without_error(self, results):
        ri_table = _make_flat_table(self.RI_RATE, 39, 45)
        df = results.ae_cost_ratio(
            ri_table, mean_sa=self.MEAN_SA, assumed_claim_duration=self.ASSUMED_DURATION
        )
        assert not df.empty
        assert "ae_cost_simple" in df.columns
        assert "ae_duration_adjusted" in df.columns


# ---------------------------------------------------------------------------
# Test 4 — Lognormal severity fit within 5 % of generating parameters
# ---------------------------------------------------------------------------

class TestLognormalSeverityFit:
    """severity_fit recovers lognormal parameters from 500 generated claims."""

    STUDY = StudyPeriod(date(2020, 1, 1), date(2020, 12, 31))
    DOB = date(1980, 1, 1)   # age 40 throughout (no birthday crossing)
    ISSUE = date(2019, 1, 1)
    MU = 10.0
    SIGMA = 0.5
    N = 500

    @pytest.fixture(scope="class")
    def results(self):
        rng = np.random.default_rng(42)
        amounts = rng.lognormal(mean=self.MU, sigma=self.SIGMA, size=self.N)
        pols = [_pol(f"P{i:03d}", self.DOB, self.ISSUE) for i in range(self.N)]
        clms = [
            _clm(f"C{i:03d}", f"P{i:03d}", date(2020, 6, 1), date(2020, 6, 1), amount=float(amounts[i]))
            for i in range(self.N)
        ]
        return MorbidityStudy(_pds(pols), _cds(clms), self.STUDY).run()

    def test_fit_returns_dict(self, results):
        fitted = results.severity_fit("lognormal")
        assert "error" not in fitted
        assert fitted["distribution"] == "lognormal"

    def test_mu_within_tolerance(self, results):
        fitted = results.severity_fit("lognormal")
        assert fitted["mu"] == pytest.approx(self.MU, rel=0.02)

    def test_sigma_within_tolerance(self, results):
        fitted = results.severity_fit("lognormal")
        assert fitted["sigma"] == pytest.approx(self.SIGMA, abs=0.05)


# ---------------------------------------------------------------------------
# Test 5 — Zero claims: rate = 0.0, no division errors
# ---------------------------------------------------------------------------

class TestZeroClaims:
    """With no claims, incidence rates are 0 and A/E methods complete cleanly."""

    STUDY = StudyPeriod(date(2020, 1, 1), date(2020, 12, 31))
    DOB = date(1970, 1, 1)
    ISSUE = date(2019, 1, 1)

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:03d}", self.DOB, self.ISSUE) for i in range(100)]
        return MorbidityStudy(_pds(pols), _cds([]), self.STUDY).run()

    def test_new_claims_zero(self, results):
        assert (results.incidence_df["new_claims"] == 0).all()

    def test_incidence_rate_zero(self, results):
        assert (results.incidence_df["incidence_rate"] == 0.0).all()

    def test_ae_cost_simple_zero_no_error(self, results):
        ri_table = _make_flat_table(0.05, 40, 60)
        df = results.ae_cost_ratio(ri_table, mean_sa=100_000.0)
        assert not df.empty
        assert (df["ae_cost_simple"] == 0.0).all()

    def test_severity_fit_handles_no_claims(self, results):
        fitted = results.severity_fit("lognormal")
        assert "error" in fitted


# ---------------------------------------------------------------------------
# Test 6 — All claims open: recovery_rate = 0.0, sick_etr > 0
# ---------------------------------------------------------------------------

class TestAllClaimsOpen:
    """Open claims produce sick ETR but zero terminations."""

    STUDY = StudyPeriod(date(2020, 1, 1), date(2020, 12, 31))
    DOB = date(1970, 1, 1)
    ISSUE = date(2019, 1, 1)
    N = 50

    @pytest.fixture(scope="class")
    def results(self):
        pols = [_pol(f"P{i:02d}", self.DOB, self.ISSUE) for i in range(self.N)]
        # Open claims starting mid-year: policies have healthy time in H1 and
        # sick time in H2 (no end date → sick until obs_end).
        clms = [_clm(f"C{i:02d}", f"P{i:02d}", date(2020, 6, 1)) for i in range(self.N)]
        return MorbidityStudy(_pds(pols), _cds(clms), self.STUDY).run()

    def test_termination_df_has_rows(self, results):
        # Sick ETR rows are present even though no claim ended
        assert not results.termination_df.empty

    def test_recoveries_zero(self, results):
        assert (results.termination_df["recoveries"] == 0).all()

    def test_recovery_rate_zero(self, results):
        # recovery_rate = recoveries / sick_etr = 0 / positive = 0.0
        assert (results.termination_df["recovery_rate"] == 0.0).all()

    def test_sick_etr_positive(self, results):
        # termination_df is populated from _build_termination_df, which always
        # has sick_etr rows regardless of whether there is any healthy_etr.
        assert results.termination_df["sick_etr"].sum() > 0

    def test_new_claims_counted(self, results):
        # Claims start 2020-06-01 so there is healthy ETR in H1; incidence_df
        # is anchored on healthy_etr rows and therefore captures the 50 claims.
        assert results.incidence_df["new_claims"].sum() == pytest.approx(self.N, abs=0.5)
