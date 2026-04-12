"""Tests for ExpenseStudy / ExpenseResults and CommissionStudy / CommissionResults.

Checkpoint S17: Full suite run (~75+ tests).
Commission anomaly detection regression fixture: seed=42, 20 normal agents
+ 1 outlier → exactly 1 agent flagged by the ensemble method.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from lifexp.core.data_model import PolicyDataset
from lifexp.core.study_period import StudyPeriod
from lifexp.expense.commission import CommissionStudy, CommissionResults
from lifexp.expense.study import ExpenseStudy, ExpenseResults


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_STUDY = StudyPeriod(date(2022, 1, 1), date(2022, 12, 31))

_ALLOCATION_KEYS = {
    "renewal":     "if_policy_count",
    "acquisition": "new_policies",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pol(pid: str, issue: date, exit_date=None, status="IF",
         premium: float = 1_000.0, sa: float = 100_000.0) -> dict:
    return {
        "policy_id":      pid,
        "date_of_birth":  date(1980, 1, 1),
        "issue_date":     issue,
        "gender":         "M",
        "smoker_status":  "NS",
        "sum_assured":    sa,
        "annual_premium": premium,
        "product_code":   "TERM",
        "channel":        "DIRECT",
        "status":         status,
        "exit_date":      exit_date,
        "exit_reason":    status if status != "IF" else None,
    }


def _pds(rows) -> PolicyDataset:
    return PolicyDataset.from_dataframe(pd.DataFrame(rows))


def _expense_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cost_centre", "expense_type", "amount", "year"])


def _commission_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["agent_id", "policy_id", "commission_amount", "payment_date"])


# ---------------------------------------------------------------------------
# Test 1 — Per-policy unit cost arithmetic
# ---------------------------------------------------------------------------

class TestPerPolicyUnitCost:
    """500 IF policies, 1 000 000 renewal expenses → per_policy = 2 000."""

    N_POLICIES = 500
    RENEWAL_TOTAL = 1_000_000.0

    @pytest.fixture(scope="class")
    def results(self):
        # All policies issued in 2021 and still in force at year-end 2022
        pols = [_pol(f"P{i:04d}", date(2021, 1, 1)) for i in range(self.N_POLICIES)]
        # Single renewal expense row
        exp = _expense_df([
            ("ops", "renewal", self.RENEWAL_TOTAL, 2022),
        ])
        return ExpenseStudy(exp, _pds(pols), _STUDY, _ALLOCATION_KEYS).run()

    def test_per_policy_correct(self, results):
        uc = results.unit_costs()
        row = uc[uc["year"] == 2022].iloc[0]
        assert row["per_policy"] == pytest.approx(
            self.RENEWAL_TOTAL / self.N_POLICIES, rel=1e-9
        )

    def test_if_count_correct(self, results):
        row = results.unit_cost_df[results.unit_cost_df["year"] == 2022].iloc[0]
        assert row["if_count"] == self.N_POLICIES

    def test_renewal_expenses_match(self, results):
        row = results.unit_cost_df[results.unit_cost_df["year"] == 2022].iloc[0]
        assert row["renewal_expenses"] == pytest.approx(self.RENEWAL_TOTAL, rel=1e-9)

    def test_unit_costs_columns(self, results):
        uc = results.unit_costs()
        assert {"year", "per_policy", "per_new_policy",
                "per_premium_pct", "per_sa_pct"} <= set(uc.columns)

    def test_ae_vs_assumption(self, results):
        """A/E ratio = actual / assumed."""
        ae = results.ae_vs_assumption({"per_policy": 1_800.0})
        row = ae[(ae["year"] == 2022) & (ae["metric"] == "per_policy")].iloc[0]
        assert row["ae_ratio"] == pytest.approx(
            (self.RENEWAL_TOTAL / self.N_POLICIES) / 1_800.0, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Test 2 — Inflation analysis (YoY change)
# ---------------------------------------------------------------------------

class TestInflationRate:
    """Two years of data: 2021 per_policy = 2000, 2022 per_policy = 2120 → +6%."""

    N_POLICIES = 500
    RENEWAL_2021 = 1_000_000.0   # per_policy = 2000
    RENEWAL_2022 = 1_060_000.0   # per_policy = 2120

    @pytest.fixture(scope="class")
    def results(self):
        # Policies issued 2020, in force both year-ends
        pols = [_pol(f"P{i:04d}", date(2020, 1, 1)) for i in range(self.N_POLICIES)]
        exp = _expense_df([
            ("ops", "renewal", self.RENEWAL_2021, 2021),
            ("ops", "renewal", self.RENEWAL_2022, 2022),
        ])
        study = StudyPeriod(date(2021, 1, 1), date(2022, 12, 31))
        return ExpenseStudy(exp, _pds(pols), study, _ALLOCATION_KEYS).run()

    def test_yoy_pct_correct(self, results):
        ia = results.inflation_analysis()
        row = ia[ia["year"] == 2022].iloc[0]
        assert row["yoy_pct"] == pytest.approx(
            (self.RENEWAL_2022 - self.RENEWAL_2021) / self.RENEWAL_2021, rel=1e-9
        )

    def test_first_year_yoy_is_nan(self, results):
        ia = results.inflation_analysis()
        row = ia[ia["year"] == 2021].iloc[0]
        assert pd.isna(row["yoy_pct"])

    def test_yoy_change_correct(self, results):
        ia = results.inflation_analysis()
        row = ia[ia["year"] == 2022].iloc[0]
        expected_change = (
            self.RENEWAL_2022 / self.N_POLICIES
            - self.RENEWAL_2021 / self.N_POLICIES
        )
        assert row["yoy_change"] == pytest.approx(expected_change, rel=1e-9)

    def test_two_rows_returned(self, results):
        ia = results.inflation_analysis()
        assert len(ia) == 2


# ---------------------------------------------------------------------------
# Test 3 — Commission anomaly detection
# ---------------------------------------------------------------------------

class TestCommissionAnomaly:
    """20 normal agents + 1 outlier: all three methods flag the outlier."""

    N_NORMAL  = 20
    NORMAL_COMMISSION = 1_000.0   # per policy
    OUTLIER_COMMISSION = 50_000.0  # 50× the normal rate — extreme outlier

    @pytest.fixture(scope="class")
    def results(self):
        rng = np.random.default_rng(42)
        rows = []
        # Normal agents: commission_per_policy near 1 000
        normal_amounts = rng.integers(950, 1050, size=self.N_NORMAL).tolist()
        for i, amt in enumerate(normal_amounts):
            rows.append({
                "agent_id":         f"A{i:03d}",
                "policy_id":        f"POL{i:04d}",
                "commission_amount": float(amt),
                "payment_date":     date(2022, 6, 1),
            })
        # Outlier agent
        rows.append({
            "agent_id":         "OUTLIER",
            "policy_id":        "POL_OUT",
            "commission_amount": self.OUTLIER_COMMISSION,
            "payment_date":     date(2022, 6, 1),
        })
        df = pd.DataFrame(rows)
        return CommissionStudy(df, {}, _STUDY).run()

    def test_outlier_flagged_zscore(self, results):
        flagged = results.flag_anomalies(method="zscore")
        assert "OUTLIER" in flagged["agent_id"].values

    def test_outlier_flagged_mad(self, results):
        flagged = results.flag_anomalies(method="mad")
        assert "OUTLIER" in flagged["agent_id"].values

    def test_outlier_flagged_iqr(self, results):
        flagged = results.flag_anomalies(method="iqr")
        assert "OUTLIER" in flagged["agent_id"].values

    def test_outlier_flagged_ensemble(self, results):
        flagged = results.flag_anomalies(method="ensemble")
        assert "OUTLIER" in flagged["agent_id"].values

    def test_normal_agents_not_flagged_ensemble(self, results):
        flagged = results.flag_anomalies(method="ensemble")
        normal_flagged = [a for a in flagged["agent_id"].values if a != "OUTLIER"]
        assert len(normal_flagged) == 0

    def test_flagged_columns_present(self, results):
        flagged = results.flag_anomalies(method="ensemble")
        assert {"agent_id", "commission_per_policy",
                "zscore_flag", "mad_flag", "iqr_flag",
                "ensemble_votes"} <= set(flagged.columns)


# ---------------------------------------------------------------------------
# Test 4 — Zero IF policies → NaN + UserWarning
# ---------------------------------------------------------------------------

class TestZeroPolicies:
    """No policies in the study year → per_policy is NaN and UserWarning raised."""

    @pytest.fixture(scope="class")
    def results(self):
        # Policy issued and exited BEFORE 2022 year-end → if_count = 0
        pols = [_pol("P001", date(2021, 1, 1), exit_date=date(2021, 6, 1), status="LAPSED")]
        exp = _expense_df([("ops", "renewal", 500_000.0, 2022)])
        return ExpenseStudy(exp, _pds(pols), _STUDY, _ALLOCATION_KEYS).run()

    def test_per_policy_is_nan(self, results):
        row = results.unit_cost_df[results.unit_cost_df["year"] == 2022].iloc[0]
        assert pd.isna(row["per_policy"])

    def test_userwarning_raised(self):
        pols = [_pol("P001", date(2021, 1, 1), exit_date=date(2021, 6, 1), status="LAPSED")]
        exp = _expense_df([("ops", "renewal", 500_000.0, 2022)])
        with pytest.warns(UserWarning, match="if_count is zero"):
            ExpenseStudy(exp, _pds(pols), _STUDY, _ALLOCATION_KEYS).run()

    def test_ae_ratio_is_nan_when_actual_nan(self, results):
        ae = results.ae_vs_assumption({"per_policy": 2_000.0})
        if len(ae) > 0:
            row = ae[(ae["year"] == 2022) & (ae["metric"] == "per_policy")]
            if len(row) > 0:
                assert pd.isna(row.iloc[0]["ae_ratio"])


# ---------------------------------------------------------------------------
# Test 5 — Single commission agent → empty result + UserWarning
# ---------------------------------------------------------------------------

class TestSingleAgent:
    """Only one agent: anomaly detection undefined → warn + empty DataFrame."""

    @pytest.fixture(scope="class")
    def results(self):
        rows = [{
            "agent_id":         "SOLO",
            "policy_id":        "POL001",
            "commission_amount": 1_000.0,
            "payment_date":     date(2022, 6, 1),
        }]
        return CommissionStudy(pd.DataFrame(rows), {}, _STUDY).run()

    def test_summary_has_one_row(self, results):
        assert len(results.summary_df) == 1

    def test_flag_anomalies_warns(self, results):
        with pytest.warns(UserWarning, match="Anomaly detection requires"):
            results.flag_anomalies()

    def test_flag_anomalies_returns_empty(self, results):
        with pytest.warns(UserWarning):
            flagged = results.flag_anomalies()
        assert len(flagged) == 0

    def test_empty_result_has_correct_columns(self, results):
        with pytest.warns(UserWarning):
            flagged = results.flag_anomalies()
        assert {"agent_id", "commission_per_policy",
                "zscore_flag", "mad_flag", "iqr_flag",
                "ensemble_votes"} <= set(flagged.columns)


# ---------------------------------------------------------------------------
# Regression fixture — Checkpoint S17
# ---------------------------------------------------------------------------

class TestCommissionRegressionFixture:
    """Regression: seed=42, 20 normal agents (integers 950–1050) + 1 outlier
    at 50 000 → ensemble flags exactly 1 agent (the outlier)."""

    @pytest.fixture(scope="class")
    def flagged(self):
        rng = np.random.default_rng(42)
        rows = []
        normal_amounts = rng.integers(950, 1050, size=20).tolist()
        for i, amt in enumerate(normal_amounts):
            rows.append({
                "agent_id":         f"A{i:03d}",
                "policy_id":        f"POL{i:04d}",
                "commission_amount": float(amt),
                "payment_date":     date(2022, 6, 1),
            })
        rows.append({
            "agent_id":         "OUTLIER",
            "policy_id":        "POL_OUT",
            "commission_amount": 50_000.0,
            "payment_date":     date(2022, 6, 1),
        })
        res = CommissionStudy(pd.DataFrame(rows), {}, _STUDY).run()
        return res.flag_anomalies(method="ensemble")

    def test_exactly_one_flagged(self, flagged):
        assert len(flagged) == 1

    def test_flagged_agent_is_outlier(self, flagged):
        assert flagged.iloc[0]["agent_id"] == "OUTLIER"

    def test_all_three_votes(self, flagged):
        assert flagged.iloc[0]["ensemble_votes"] == 3

    def test_commission_per_policy_correct(self, flagged):
        assert flagged.iloc[0]["commission_per_policy"] == pytest.approx(50_000.0, rel=1e-9)
