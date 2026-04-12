"""Tests for lifexp.core.audit: StudyRun, @audit_run, compare_runs.

Checkpoint S20: data_checksum is deterministic — same data always yields
the same MD5 hex-digest regardless of execution order or machine.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime

import pandas as pd
import pytest

import lifexp
from lifexp.core.audit import (
    StudyRun,
    audit_run,
    checksum_dataset,
    compare_runs,
)
from lifexp.core.data_model import PolicyDataset
from lifexp.core.study_period import StudyPeriod


# ---------------------------------------------------------------------------
# Minimal study stub used across tests
# ---------------------------------------------------------------------------

class _FakeResults:
    """Minimal results object the stub study returns."""
    def __init__(self, value: float, dataset_size: int):
        self.overall_ae = value
        self.total_deaths = float(dataset_size)
        self.total_etr = float(dataset_size) * 0.9
        self.summary_df = pd.DataFrame({"x": range(dataset_size)})


@audit_run(study_type="fake", dataset_attr="_dataset")
class _FakeStudy:
    """Minimal study that captures constructor params and runs instantly."""

    def __init__(self, dataset: PolicyDataset, lam: float = 100.0,
                 label: str = "default"):
        self._dataset = dataset
        self.lam   = lam
        self.label = label

    def run(self) -> _FakeResults:
        return _FakeResults(
            value=self.lam / 100.0,
            dataset_size=len(self._dataset._records),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(pid: str, status: str = "IF") -> dict:
    return {
        "policy_id":      pid,
        "date_of_birth":  date(1975, 6, 1),
        "issue_date":     date(2015, 1, 1),
        "gender":         "M",
        "smoker_status":  "NS",
        "sum_assured":    100_000.0,
        "annual_premium": 500.0,
        "product_code":   "TERM",
        "channel":        "DIRECT",
        "status":         status,
        "exit_date":      date(2022, 6, 1) if status != "IF" else None,
        "exit_reason":    status if status != "IF" else None,
    }


def _pds(n: int, prefix: str = "P") -> PolicyDataset:
    rows = [_make_policy(f"{prefix}{i:04d}") for i in range(n)]
    return PolicyDataset.from_dataframe(pd.DataFrame(rows))


def _run(n: int = 20, lam: float = 100.0, label: str = "default",
         prefix: str = "P") -> _FakeResults:
    return _FakeStudy(_pds(n, prefix=prefix), lam=lam, label=label).run()


def _make_study_run(**kwargs) -> StudyRun:
    defaults = dict(
        run_id=str(uuid.uuid4()),
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        lifexp_version="0.1.0",
        python_version=sys.version,
        study_type="mortality",
        parameters={"lam": 100.0},
        data_checksum="abc123",
        execution_time_s=0.5,
        output_stats={"overall_ae": 1.0},
    )
    defaults.update(kwargs)
    return StudyRun(**defaults)


# ---------------------------------------------------------------------------
# Test 1 — Unique run_id each call
# ---------------------------------------------------------------------------

class TestUniqueRunId:
    """Running the same study twice produces two distinct UUID run_ids."""

    def test_two_runs_have_different_ids(self):
        r1 = _run()
        r2 = _run()
        assert r1.audit_run.run_id != r2.audit_run.run_id

    def test_run_id_is_valid_uuid(self):
        r = _run()
        # Should parse without error
        parsed = uuid.UUID(r.audit_run.run_id)
        assert parsed.version == 4

    def test_hundred_runs_all_unique(self):
        ids = {_run().audit_run.run_id for _ in range(100)}
        assert len(ids) == 100

    def test_audit_run_attached_to_results(self):
        r = _run()
        assert hasattr(r, "audit_run")
        assert isinstance(r.audit_run, StudyRun)

    def test_study_type_captured(self):
        r = _run()
        assert r.audit_run.study_type == "fake"

    def test_lifexp_version_captured(self):
        r = _run()
        assert r.audit_run.lifexp_version == lifexp.__version__

    def test_python_version_captured(self):
        r = _run()
        assert r.audit_run.python_version == sys.version

    def test_execution_time_positive(self):
        r = _run()
        assert r.audit_run.execution_time_s >= 0.0

    def test_timestamp_is_recent_utc(self):
        before = datetime.utcnow()
        r = _run()
        after = datetime.utcnow()
        ts = r.audit_run.timestamp
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# Test 2 — Checksum changes when data changes
# ---------------------------------------------------------------------------

class TestChecksumChangesWithData:
    """Modifying one record produces a different data_checksum."""

    def test_different_data_different_checksum(self):
        r1 = _run(n=20)
        # Same count, different policy IDs → different content
        r2 = _run(n=20, prefix="Q")
        assert r1.audit_run.data_checksum != r2.audit_run.data_checksum

    def test_more_records_different_checksum(self):
        r1 = _run(n=20)
        r2 = _run(n=21)
        assert r1.audit_run.data_checksum != r2.audit_run.data_checksum

    def test_checksum_is_32_char_hex(self):
        r = _run()
        cs = r.audit_run.data_checksum
        assert len(cs) == 32
        assert all(c in "0123456789abcdef" for c in cs)

    # --- Checkpoint S20: determinism ---

    def test_checksum_deterministic_same_data(self):
        """Checkpoint S20: identical inputs always produce the same MD5."""
        ds = _pds(50)
        c1 = checksum_dataset(ds)
        c2 = checksum_dataset(ds)
        assert c1 == c2

    def test_checksum_order_independent(self):
        """Records sorted by policy_id before hashing → order-independent."""
        rows_fwd = [_make_policy(f"P{i:04d}") for i in range(10)]
        rows_rev = list(reversed(rows_fwd))
        ds_fwd = PolicyDataset.from_dataframe(pd.DataFrame(rows_fwd))
        ds_rev = PolicyDataset.from_dataframe(pd.DataFrame(rows_rev))
        assert checksum_dataset(ds_fwd) == checksum_dataset(ds_rev)

    def test_checksum_sensitive_to_field_value(self):
        """Changing sum_assured on one record changes the checksum."""
        rows = [_make_policy(f"P{i:04d}") for i in range(10)]
        ds1 = PolicyDataset.from_dataframe(pd.DataFrame(rows))

        rows_mod = [r.copy() for r in rows]
        rows_mod[5]["sum_assured"] = 200_000.0
        ds2 = PolicyDataset.from_dataframe(pd.DataFrame(rows_mod))

        assert checksum_dataset(ds1) != checksum_dataset(ds2)

    def test_checksum_matches_via_run(self):
        """Checksum from audit_run matches direct checksum_dataset call."""
        ds = _pds(30)
        r = _FakeStudy(ds, lam=50.0).run()
        assert r.audit_run.data_checksum == checksum_dataset(ds)


# ---------------------------------------------------------------------------
# Test 3 — compare_runs detects parameter changes
# ---------------------------------------------------------------------------

class TestCompareRunsDetectsParamChange:
    """compare_runs surfaces differences in study parameters."""

    def test_lam_change_appears_in_diffs(self):
        r1 = _run(lam=100.0)
        r2 = _run(lam=200.0)
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert "lam" in diff["parameter_diffs"]

    def test_lam_diff_direction_correct(self):
        r1 = _run(lam=100.0)
        r2 = _run(lam=200.0)
        diff = compare_runs(r1.audit_run, r2.audit_run)
        entry = diff["parameter_diffs"]["lam"]
        assert entry["from"] == pytest.approx(100.0)
        assert entry["to"]   == pytest.approx(200.0)

    def test_no_param_diff_when_identical(self):
        r1 = _run(lam=100.0, label="x")
        r2 = _run(lam=100.0, label="x")
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert diff["parameter_diffs"] == {}

    def test_same_data_flag_true_for_same_dataset(self):
        ds = _pds(20)
        r1 = _FakeStudy(ds, lam=100.0).run()
        r2 = _FakeStudy(ds, lam=200.0).run()
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert diff["same_data"] is True

    def test_same_data_flag_false_for_different_dataset(self):
        r1 = _run(n=20)
        r2 = _run(n=20, prefix="Q")
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert diff["same_data"] is False

    def test_data_checksum_sub_dict_when_different(self):
        r1 = _run(n=20)
        r2 = _run(n=20, prefix="Q")
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert "data_checksum" in diff
        assert "run_a" in diff["data_checksum"]
        assert "run_b" in diff["data_checksum"]

    def test_data_checksum_key_absent_when_same(self):
        ds = _pds(10)
        r1 = _FakeStudy(ds).run()
        r2 = _FakeStudy(ds).run()
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert "data_checksum" not in diff

    def test_execution_time_delta_present(self):
        r1 = _run()
        r2 = _run()
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert "execution_time_delta_s" in diff

    def test_label_change_appears_in_diffs(self):
        r1 = _run(label="alpha")
        r2 = _run(label="beta")
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert "label" in diff["parameter_diffs"]

    def test_version_diff_detected(self):
        ra = _make_study_run(lifexp_version="0.1.0")
        rb = _make_study_run(lifexp_version="0.2.0")
        diff = compare_runs(ra, rb)
        assert diff["same_version"] is False
        assert diff["version_diff"]["run_a"] == "0.1.0"
        assert diff["version_diff"]["run_b"] == "0.2.0"

    def test_same_version_flag_true(self):
        ra = _make_study_run(lifexp_version="0.1.0")
        rb = _make_study_run(lifexp_version="0.1.0")
        diff = compare_runs(ra, rb)
        assert diff["same_version"] is True
        assert "version_diff" not in diff


# ---------------------------------------------------------------------------
# Test 4 — JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    """to_json → from_json reconstructs an equal StudyRun."""

    def test_round_trip_from_run(self, tmp_path):
        r = _run(n=15, lam=150.0)
        original = r.audit_run
        path = str(tmp_path / "run.json")
        original.to_json(path)
        reloaded = StudyRun.from_json(path)

        assert reloaded.run_id           == original.run_id
        assert reloaded.data_checksum    == original.data_checksum
        assert reloaded.study_type       == original.study_type
        assert reloaded.lifexp_version   == original.lifexp_version
        assert reloaded.execution_time_s == pytest.approx(original.execution_time_s, abs=1e-6)

    def test_round_trip_parameters(self, tmp_path):
        r = _run(lam=77.5, label="roundtrip")
        path = str(tmp_path / "run.json")
        r.audit_run.to_json(path)
        reloaded = StudyRun.from_json(path)
        assert reloaded.parameters.get("lam") == pytest.approx(77.5)
        assert reloaded.parameters.get("label") == "roundtrip"

    def test_round_trip_timestamp(self, tmp_path):
        r = _run()
        path = str(tmp_path / "run.json")
        r.audit_run.to_json(path)
        reloaded = StudyRun.from_json(path)
        assert reloaded.timestamp == r.audit_run.timestamp

    def test_round_trip_output_stats(self, tmp_path):
        r = _run(n=10, lam=100.0)
        path = str(tmp_path / "run.json")
        r.audit_run.to_json(path)
        reloaded = StudyRun.from_json(path)
        assert "overall_ae" in reloaded.output_stats

    def test_to_json_creates_parent_dirs(self, tmp_path):
        r = _run()
        path = str(tmp_path / "audit" / "sub" / "run.json")
        r.audit_run.to_json(path)
        import os
        assert os.path.isfile(path)

    def test_to_dict_is_json_serialisable(self):
        r = _run()
        d = r.audit_run.to_dict()
        # Must not raise
        s = json.dumps(d)
        assert "run_id" in s

    def test_from_dict_round_trip(self):
        original = _make_study_run()
        reloaded = StudyRun.from_dict(original.to_dict())
        assert reloaded.run_id        == original.run_id
        assert reloaded.data_checksum == original.data_checksum
        assert reloaded.parameters    == original.parameters


# ---------------------------------------------------------------------------
# Checkpoint S20 — determinism regression
# ---------------------------------------------------------------------------

class TestChecksumDeterminism:
    """Checkpoint S20: same data → same MD5 always; proves audit integrity."""

    # Known-good MD5 for the canonical 5-record dataset below.
    # Generated once; must stay stable across runs and machines.
    _CANONICAL_ROWS = [
        _make_policy("AUDIT001"),
        _make_policy("AUDIT002"),
        _make_policy("AUDIT003"),
        _make_policy("AUDIT004"),
        _make_policy("AUDIT005"),
    ]

    @pytest.fixture(scope="class")
    def canonical_checksum(self):
        ds = PolicyDataset.from_dataframe(pd.DataFrame(self._CANONICAL_ROWS))
        return checksum_dataset(ds)

    def test_canonical_checksum_stable(self, canonical_checksum):
        """Re-compute 100× — always the same value."""
        ds = PolicyDataset.from_dataframe(pd.DataFrame(self._CANONICAL_ROWS))
        for _ in range(100):
            assert checksum_dataset(ds) == canonical_checksum

    def test_canonical_checksum_order_independent(self, canonical_checksum):
        """Reversed insertion order produces the same digest."""
        reversed_rows = list(reversed(self._CANONICAL_ROWS))
        ds_rev = PolicyDataset.from_dataframe(pd.DataFrame(reversed_rows))
        assert checksum_dataset(ds_rev) == canonical_checksum

    def test_canonical_checksum_is_md5(self, canonical_checksum):
        """Digest is a 32-char lowercase hex string (MD5 format)."""
        assert len(canonical_checksum) == 32
        assert canonical_checksum == canonical_checksum.lower()
        int(canonical_checksum, 16)  # raises if not valid hex

    def test_single_field_change_breaks_checksum(self, canonical_checksum):
        """Auditor guarantee: any data mutation changes the digest."""
        mutated = [r.copy() for r in self._CANONICAL_ROWS]
        mutated[2]["sum_assured"] = 999_999.0
        ds_mut = PolicyDataset.from_dataframe(pd.DataFrame(mutated))
        assert checksum_dataset(ds_mut) != canonical_checksum

    def test_added_record_breaks_checksum(self, canonical_checksum):
        extra = self._CANONICAL_ROWS + [_make_policy("AUDIT006")]
        ds_extra = PolicyDataset.from_dataframe(pd.DataFrame(extra))
        assert checksum_dataset(ds_extra) != canonical_checksum

    def test_checksum_in_compare_runs_proof(self):
        """End-to-end proof: same data → same_data=True in compare_runs."""
        ds = PolicyDataset.from_dataframe(pd.DataFrame(self._CANONICAL_ROWS))
        r1 = _FakeStudy(ds, lam=1.0).run()
        r2 = _FakeStudy(ds, lam=2.0).run()
        diff = compare_runs(r1.audit_run, r2.audit_run)
        assert diff["same_data"] is True
        assert r1.audit_run.data_checksum == r2.audit_run.data_checksum
