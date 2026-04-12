"""Tests for lifexp.core.data_model: PolicyRecord and PolicyDataset."""

from datetime import date, timedelta
import random
import string

import pandas as pd
import pytest

from lifexp.core.data_model import PolicyDataset, PolicyRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOB = date(1980, 6, 15)
ISSUE = date(2010, 1, 1)
EXIT = date(2020, 3, 10)


def make_record(**overrides) -> PolicyRecord:
    """Return a valid in-force PolicyRecord, with optional field overrides."""
    defaults = dict(
        policy_id="P001",
        date_of_birth=DOB,
        issue_date=ISSUE,
        gender="M",
        smoker_status="NS",
        sum_assured=100_000.0,
        annual_premium=1_200.0,
        product_code="TERM10",
        channel="AGENT",
        status="IF",
        exit_date=None,
        exit_reason=None,
    )
    defaults.update(overrides)
    return PolicyRecord(**defaults)


# ---------------------------------------------------------------------------
# Normal: valid in-force record
# ---------------------------------------------------------------------------

def test_valid_inforce_record():
    record = make_record()
    errors = record.validate()
    assert errors == [], f"Expected no errors, got: {errors}"


# ---------------------------------------------------------------------------
# Normal: valid death record
# ---------------------------------------------------------------------------

def test_valid_death_record():
    record = make_record(status="DEATH", exit_date=EXIT, exit_reason="DEATH")
    errors = record.validate()
    assert errors == [], f"Expected no errors, got: {errors}"


# ---------------------------------------------------------------------------
# Normal: DataFrame round-trip
# ---------------------------------------------------------------------------

def test_dataframe_round_trip():
    records = [
        make_record(policy_id="P001"),
        make_record(
            policy_id="P002",
            status="DEATH",
            exit_date=EXIT,
            exit_reason="DEATH",
            gender="F",
            smoker_status="S",
        ),
    ]
    dataset = PolicyDataset(records)
    df_out = dataset.to_dataframe()

    # Re-build from the DataFrame with no field_map
    dataset2 = PolicyDataset.from_dataframe(df_out)
    df_out2 = dataset2.to_dataframe()

    # Compare string representations to avoid date type mismatches
    assert list(df_out.columns) == list(df_out2.columns)
    assert len(df_out) == len(df_out2)
    for col in df_out.columns:
        for i in range(len(df_out)):
            v1 = df_out.iloc[i][col]
            v2 = df_out2.iloc[i][col]
            assert str(v1) == str(v2), f"Mismatch in col={col} row={i}: {v1!r} != {v2!r}"


# ---------------------------------------------------------------------------
# Edge: death without exit_date
# ---------------------------------------------------------------------------

def test_death_without_exit_date():
    record = make_record(status="DEATH", exit_date=None)
    errors = record.validate()
    assert any("exit_date" in e for e in errors), (
        f"Expected an exit_date error, got: {errors}"
    )


# ---------------------------------------------------------------------------
# Edge: exit before issue
# ---------------------------------------------------------------------------

def test_exit_before_issue():
    record = make_record(
        status="DEATH",
        exit_date=ISSUE - timedelta(days=1),
        exit_reason="DEATH",
    )
    errors = record.validate()
    assert any("exit_date" in e for e in errors), (
        f"Expected a chronology error, got: {errors}"
    )


# ---------------------------------------------------------------------------
# Edge: zero sum assured
# ---------------------------------------------------------------------------

def test_zero_sum_assured():
    record = make_record(sum_assured=0.0)
    errors = record.validate()
    assert any("sum_assured" in e for e in errors), (
        f"Expected a sum_assured error, got: {errors}"
    )


# ---------------------------------------------------------------------------
# Boundary: issue date == date of birth (newborn policy)
# ---------------------------------------------------------------------------

def test_issue_date_equals_dob():
    record = make_record(date_of_birth=ISSUE, issue_date=ISSUE)
    errors = record.validate()
    assert errors == [], f"Newborn policy should be valid, got: {errors}"


# ---------------------------------------------------------------------------
# Edge: empty dataset summary
# ---------------------------------------------------------------------------

def test_empty_dataset_summary(capsys):
    dataset = PolicyDataset([])
    dataset.summary()  # must not raise
    captured = capsys.readouterr()
    assert "0" in captured.out


# ---------------------------------------------------------------------------
# Bulk: 1000-row DataFrame with mixed statuses
# ---------------------------------------------------------------------------

def _random_policy_id(n: int) -> str:
    return "P" + str(n).zfill(6)


def test_bulk_1000_rows():
    """from_dataframe handles 1000 rows with mixed statuses without error."""
    rng = random.Random(42)
    statuses_with_exit = ["LAPSED", "DEATH", "SURRENDERED", "MATURED", "PU"]
    rows = []
    for i in range(1000):
        use_exit = rng.random() < 0.4
        status = rng.choice(statuses_with_exit) if use_exit else "IF"
        exit_d = EXIT + timedelta(days=rng.randint(1, 365)) if use_exit else None
        rows.append(
            dict(
                policy_id=_random_policy_id(i),
                date_of_birth=DOB,
                issue_date=ISSUE,
                gender=rng.choice(["M", "F", "U"]),
                smoker_status=rng.choice(["S", "NS", "U"]),
                sum_assured=float(rng.randint(50_000, 500_000)),
                annual_premium=float(rng.randint(0, 5_000)),
                product_code="TERM10",
                channel="AGENT",
                status=status,
                exit_date=exit_d,
                exit_reason="EXIT" if use_exit else None,
            )
        )

    df = pd.DataFrame(rows)
    dataset = PolicyDataset.from_dataframe(df)
    assert len(dataset._records) == 1000

    errors = dataset.validate()
    assert errors == [], f"Expected no errors on clean data, got {len(errors)} errors"

    df_out = dataset.to_dataframe()
    assert len(df_out) == 1000
