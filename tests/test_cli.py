"""Tests for lifexp CLI (lifexp.cli) and public API (__init__.py).

Checkpoint S21: all public-API imports exercised from the installed package.
CLI tests use click.testing.CliRunner (in-process, no subprocess overhead).
"""

from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

import lifexp
from lifexp.cli import cli


# ---------------------------------------------------------------------------
# Synthetic CSV helper
# ---------------------------------------------------------------------------

def _write_policies_csv(path: str, n_if: int = 30, n_deaths: int = 3) -> None:
    """Write a minimal policies CSV to *path*."""
    rows = []
    for i in range(n_if):
        rows.append({
            "policy_id":      f"P{i:04d}",
            "date_of_birth":  "1975-01-01",
            "issue_date":     "2018-01-01",
            "gender":         "M",
            "smoker_status":  "NS",
            "sum_assured":    100_000.0,
            "annual_premium": 500.0,
            "product_code":   "TERM",
            "channel":        "DIRECT",
            "status":         "IF",
            "exit_date":      None,
            "exit_reason":    None,
        })
    for j in range(n_deaths):
        rows.append({
            "policy_id":      f"D{j:04d}",
            "date_of_birth":  "1975-01-01",
            "issue_date":     "2018-01-01",
            "gender":         "M",
            "smoker_status":  "NS",
            "sum_assured":    100_000.0,
            "annual_premium": 500.0,
            "product_code":   "TERM",
            "channel":        "DIRECT",
            "status":         "DEATH",
            "exit_date":      "2022-06-15",
            "exit_reason":    "DEATH",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Test 0 — Public API imports (Checkpoint S21)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """All symbols in __all__ are importable from lifexp directly."""

    def test_policy_dataset_importable(self):
        from lifexp import PolicyDataset
        assert PolicyDataset is not None

    def test_claim_dataset_importable(self):
        from lifexp import ClaimDataset
        assert ClaimDataset is not None

    def test_study_period_importable(self):
        from lifexp import StudyPeriod
        assert StudyPeriod is not None

    def test_mortality_study_importable(self):
        from lifexp import MortalityStudy
        assert MortalityStudy is not None

    def test_lapse_study_importable(self):
        from lifexp import LapseStudy
        assert LapseStudy is not None

    def test_morbidity_study_importable(self):
        from lifexp import MorbidityStudy
        assert MorbidityStudy is not None

    def test_ri_study_importable(self):
        from lifexp import RIStudy
        assert RIStudy is not None

    def test_expense_study_importable(self):
        from lifexp import ExpenseStudy
        assert ExpenseStudy is not None

    def test_table_registry_importable(self):
        from lifexp import TableRegistry
        assert TableRegistry is not None

    def test_html_report_importable(self):
        from lifexp import HTMLReport
        assert HTMLReport is not None

    def test_excel_report_importable(self):
        from lifexp import ExcelReport
        assert ExcelReport is not None

    def test_all_exports_listed(self):
        expected = {
            "PolicyDataset", "ClaimDataset", "StudyPeriod",
            "MortalityStudy", "LapseStudy", "MorbidityStudy",
            "RIStudy", "ExpenseStudy",
            "TableRegistry",
            "HTMLReport", "ExcelReport",
        }
        assert expected <= set(lifexp.__all__)

    def test_star_import_symbols_accessible(self):
        """Every name in __all__ is a real attribute on the package."""
        for name in lifexp.__all__:
            assert hasattr(lifexp, name), f"lifexp.{name} not found"


# ---------------------------------------------------------------------------
# Test 1 — lifexp version
# ---------------------------------------------------------------------------

class TestVersionCommand:
    """lifexp version prints the version string and exits 0."""

    def test_version_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0

    def test_version_prints_correct_string(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert lifexp.__version__ in result.output

    def test_version_flag_also_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert lifexp.__version__ in result.output
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 2 — mortality end-to-end
# ---------------------------------------------------------------------------

class TestMortalityCLI:
    """lifexp mortality with a synthetic CSV produces output files."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("mort")
        csv_path = str(tmp / "policies.csv")
        out_dir  = str(tmp / "results")
        _write_policies_csv(csv_path, n_if=50, n_deaths=2)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",        csv_path,
            "--study-start", "2022-01-01",
            "--study-end",   "2022-12-31",
            "--table",       "A_1967_70",
            "--output",      out_dir,
            "--format",      "html,excel",
        ])
        return result, out_dir

    def test_exit_code_zero(self, run_result):
        result, _ = run_result
        assert result.exit_code == 0, result.output

    def test_html_file_created(self, run_result):
        _, out_dir = run_result
        assert os.path.isfile(os.path.join(out_dir, "mortality.html"))

    def test_excel_file_created(self, run_result):
        _, out_dir = run_result
        assert os.path.isfile(os.path.join(out_dir, "mortality.xlsx"))

    def test_audit_json_created(self, run_result):
        _, out_dir = run_result
        assert os.path.isfile(os.path.join(out_dir, "mortality_audit.json"))

    def test_audit_json_valid(self, run_result):
        _, out_dir = run_result
        with open(os.path.join(out_dir, "mortality_audit.json")) as fh:
            audit = json.load(fh)
        assert audit["study_type"] == "mortality"
        assert len(audit["data_checksum"]) == 32
        assert audit["lifexp_version"] == lifexp.__version__

    def test_output_mentions_records(self, run_result):
        result, _ = run_result
        assert "records" in result.output.lower() or "loaded" in result.output.lower()

    def test_output_mentions_ae(self, run_result):
        result, _ = run_result
        assert "A/E" in result.output or "ae" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 3 — Missing required flag → exit 1 with clear message
# ---------------------------------------------------------------------------

class TestMissingRequiredFlag:
    """Missing --data causes exit code 2 (click UsageError) or 1."""

    def test_missing_data_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--study-start", "2022-01-01",
            "--study-end",   "2022-12-31",
        ])
        assert result.exit_code != 0
        assert "--data" in result.output.lower() or "data" in result.output.lower()

    def test_missing_study_start(self, tmp_path):
        csv_path = str(tmp_path / "p.csv")
        _write_policies_csv(csv_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",      csv_path,
            "--study-end", "2022-12-31",
        ])
        assert result.exit_code != 0

    def test_missing_study_end(self, tmp_path):
        csv_path = str(tmp_path / "p.csv")
        _write_policies_csv(csv_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",        csv_path,
            "--study-start", "2022-01-01",
        ])
        assert result.exit_code != 0

    def test_tables_no_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tables"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Test 4 — Invalid date format → clear error message
# ---------------------------------------------------------------------------

class TestInvalidDateFormat:
    """Wrong date separator produces a helpful UsageError."""

    def test_slash_separator_rejected(self, tmp_path):
        csv_path = str(tmp_path / "p.csv")
        _write_policies_csv(csv_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",        csv_path,
            "--study-start", "2022/01/01",
            "--study-end",   "2022-12-31",
        ])
        assert result.exit_code != 0
        # Message must mention the expected format
        combined = result.output + (result.exception.__str__() if result.exception else "")
        assert "YYYY-MM-DD" in combined or "format" in combined.lower()

    def test_ddmmyyyy_rejected(self, tmp_path):
        csv_path = str(tmp_path / "p.csv")
        _write_policies_csv(csv_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",        csv_path,
            "--study-start", "01-01-2022",
            "--study-end",   "2022-12-31",
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Test 5 — YAML config file
# ---------------------------------------------------------------------------

class TestYAMLConfig:
    """Passing --config study.yaml is equivalent to passing flags individually."""

    @pytest.fixture(scope="class")
    def flag_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("yaml_flags")
        csv = str(tmp / "p.csv")
        out = str(tmp / "out_flags")
        _write_policies_csv(csv, n_if=20, n_deaths=1)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mortality",
            "--data",        csv,
            "--study-start", "2022-01-01",
            "--study-end",   "2022-12-31",
            "--table",       "A_1967_70",
            "--output",      out,
            "--format",      "html,excel",
        ])
        return result, out, csv

    @pytest.fixture(scope="class")
    def yaml_result(self, tmp_path_factory, flag_result):
        _, _, csv = flag_result
        tmp = tmp_path_factory.mktemp("yaml_config")
        out = str(tmp / "out_yaml")
        cfg_path = str(tmp / "study.yaml")
        with open(cfg_path, "w") as fh:
            yaml.dump({
                "data":         csv,
                "study_start":  "2022-01-01",
                "study_end":    "2022-12-31",
                "table":        "A_1967_70",
                "output":       out,
                "fmt":          "html,excel",
            }, fh)
        runner = CliRunner()
        result = runner.invoke(cli, ["mortality", "--config", cfg_path])
        return result, out

    def test_yaml_exit_zero(self, yaml_result):
        result, _ = yaml_result
        assert result.exit_code == 0, result.output

    def test_yaml_html_created(self, yaml_result):
        _, out = yaml_result
        assert os.path.isfile(os.path.join(out, "mortality.html"))

    def test_yaml_excel_created(self, yaml_result):
        _, out = yaml_result
        assert os.path.isfile(os.path.join(out, "mortality.xlsx"))

    def test_yaml_audit_created(self, yaml_result):
        _, out = yaml_result
        assert os.path.isfile(os.path.join(out, "mortality_audit.json"))

    def test_yaml_and_flags_same_checksum(self, flag_result, yaml_result):
        """Both runs use the same CSV → same data_checksum in audit JSON."""
        _, out_flags, _ = flag_result
        _, out_yaml     = yaml_result
        with open(os.path.join(out_flags, "mortality_audit.json")) as fh:
            audit_flags = json.load(fh)
        with open(os.path.join(out_yaml, "mortality_audit.json")) as fh:
            audit_yaml = json.load(fh)
        assert audit_flags["data_checksum"] == audit_yaml["data_checksum"]


# ---------------------------------------------------------------------------
# Test 6 — tables command
# ---------------------------------------------------------------------------

class TestTablesCommand:
    """lifexp tables --list shows built-in tables."""

    def test_list_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tables", "--list"])
        assert result.exit_code == 0

    def test_list_contains_builtin(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tables", "--list"])
        assert "A_1967_70" in result.output

    def test_register_requires_name(self, tmp_path):
        csv = str(tmp_path / "t.csv")
        pd.DataFrame({"age": [40, 41], "qx": [0.001, 0.0012]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(cli, ["tables", "--register", csv])
        assert result.exit_code != 0
        assert "name" in result.output.lower()

    def test_register_valid_table(self, tmp_path):
        ages = list(range(40, 71))
        csv = str(tmp_path / "custom.csv")
        pd.DataFrame({"age": ages, "qx": [0.001 * (1 + 0.05 * i) for i in range(len(ages))]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "tables", "--register", csv, "--name", "CUSTOM_2024"
        ])
        assert result.exit_code == 0
        assert "CUSTOM_2024" in result.output

    def test_lapse_command_exits_zero(self, tmp_path):
        csv = str(tmp_path / "p.csv")
        out = str(tmp_path / "out")
        _write_policies_csv(csv, n_if=20, n_deaths=0)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "lapse",
            "--data",        csv,
            "--study-start", "2022-01-01",
            "--study-end",   "2022-12-31",
            "--output",      out,
        ])
        assert result.exit_code == 0, result.output
        assert os.path.isfile(os.path.join(out, "lapse_audit.json"))
