"""Command-line interface for lifexp.

Entry point: ``lifexp`` (registered via ``pyproject.toml``).

Commands
--------
lifexp mortality  --data FILE --study-start DATE --study-end DATE
                  --table NAME --output DIR [--format html,excel]
                  [--config FILE]

lifexp lapse      --data FILE --study-start DATE --study-end DATE
                  --output DIR [--format html,excel] [--config FILE]

lifexp tables     --list
lifexp tables     --register FILE --name NAME

lifexp version
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date
from typing import Optional

import click

import lifexp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(ctx, param, value: Optional[str]) -> Optional[date]:
    """Click callback: parse YYYY-MM-DD; raise UsageError on bad format."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise click.UsageError(
            f"Expected YYYY-MM-DD format for {param.name!r}, got {value!r}"
        )


def _load_config(config_path: Optional[str]) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    if config_path is None:
        return {}
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise click.UsageError(
            "PyYAML is required for --config support. "
            "Install it with: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _merge(defaults: dict, config: dict, overrides: dict) -> dict:
    """Merge: config values override defaults; CLI overrides override config."""
    merged = {**defaults, **config}
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _echo(msg: str) -> None:
    click.echo(msg)


def _load_policy_data(data_path: str):
    """Load a CSV as a PolicyDataset."""
    import pandas as pd
    from lifexp.core.data_model import PolicyDataset

    df = pd.read_csv(data_path, parse_dates=False)
    _echo(f"  Loaded {len(df):,} records from {data_path}")
    return PolicyDataset.from_dataframe(df)


def _write_audit(run, output_dir: str, prefix: str) -> None:
    """Save audit JSON alongside study results."""
    path = os.path.join(output_dir, f"{prefix}_audit.json")
    run.to_json(path)
    _echo(f"  Audit log written to {path}")


def _write_outputs(results, output_dir: str, formats: list,
                   prefix: str, title: str) -> None:
    """Render HTML and/or Excel reports from a results summary_df."""
    os.makedirs(output_dir, exist_ok=True)

    if "html" in formats:
        from lifexp.reporting.html_report import HTMLReport
        rpt = HTMLReport(title, {})
        rpt.add_ae_table(results.summary_df, caption=title)
        path = os.path.join(output_dir, f"{prefix}.html")
        rpt.render(path)
        _echo(f"  HTML report written to {path}")

    if "excel" in formats:
        from lifexp.reporting.excel_report import ExcelReport
        rpt = ExcelReport()
        rpt.add_ae_sheet(results.summary_df, sheet_name="Results")
        path = os.path.join(output_dir, f"{prefix}.xlsx")
        rpt.render(path)
        _echo(f"  Excel report written to {path}")


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=lifexp.__version__, prog_name="lifexp")
def cli():
    """lifexp — Actuarial life experience analysis toolkit."""


# ---------------------------------------------------------------------------
# lifexp version
# ---------------------------------------------------------------------------

@cli.command("version")
def cmd_version():
    """Print the lifexp version and exit."""
    click.echo(lifexp.__version__)


# ---------------------------------------------------------------------------
# lifexp mortality
# ---------------------------------------------------------------------------

@cli.command("mortality")
@click.option("--data",         default=None, help="Path to policies CSV file.")
@click.option("--study-start",  default=None, callback=_parse_date, is_eager=False,
              expose_value=True, help="Study start date (YYYY-MM-DD).")
@click.option("--study-end",    default=None, callback=_parse_date, is_eager=False,
              expose_value=True, help="Study end date (YYYY-MM-DD).")
@click.option("--table",        default=None, help="Mortality table name (default: A_1967_70).")
@click.option("--output",       default=None, help="Output directory (default: results).")
@click.option("--format",       "fmt", default=None,
              help="Comma-separated output formats: html, excel (default: html,excel).")
@click.option("--config",       default=None, help="YAML config file.")
def cmd_mortality(data, study_start, study_end, table, output, fmt, config):
    """Run a mortality experience study."""
    try:
        cfg = _load_config(config)
        params = _merge(
            defaults={"table": "A_1967_70", "output": "results", "fmt": "html,excel"},
            config=cfg,
            overrides={
                "data": data, "study_start": study_start,
                "study_end": study_end, "table": table,
                "output": output, "fmt": fmt,
            },
        )

        # Validate required fields
        if not params.get("data"):
            raise click.UsageError("--data is required (path to policies CSV).")
        if not params.get("study_start"):
            raise click.UsageError("--study-start is required (YYYY-MM-DD).")
        if not params.get("study_end"):
            raise click.UsageError("--study-end is required (YYYY-MM-DD).")

        # Coerce dates from config (YAML gives strings)
        s_start = params["study_start"]
        s_end   = params["study_end"]
        if isinstance(s_start, str):
            s_start = _parse_date(None, _FakeDateParam("study_start"), s_start)
        if isinstance(s_end, str):
            s_end = _parse_date(None, _FakeDateParam("study_end"), s_end)

        formats = [f.strip().lower() for f in str(params["fmt"]).split(",")]

        from lifexp.core.study_period import StudyPeriod
        from lifexp.mortality.study import MortalityStudy
        from lifexp.core.audit import checksum_dataset, StudyRun
        import uuid
        from datetime import datetime

        _echo("lifexp mortality study")
        _echo(f"  Study period : {s_start} → {s_end}")
        _echo(f"  Table        : {params['table']}")

        dataset = _load_policy_data(params["data"])
        study   = StudyPeriod(s_start, s_end)

        _echo("  Computing ETR and expected deaths ...")
        # MortalityStudy.standard_table accepts a table name string
        ms = MortalityStudy(dataset, study, standard_table=str(params["table"]))
        results = ms.run()

        _echo(f"  Overall A/E  : {results.overall_ae:.4f}")
        _echo(f"  Total deaths : {results.total_deaths:,.0f}")

        out_dir = params["output"]
        _write_outputs(results, out_dir, formats, "mortality", "Mortality Study")

        # Audit record
        checksum = checksum_dataset(dataset)
        audit = StudyRun(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            lifexp_version=lifexp.__version__,
            python_version=sys.version,
            study_type="mortality",
            parameters={
                "data": params["data"],
                "study_start": str(s_start),
                "study_end":   str(s_end),
                "table":       params["table"],
            },
            data_checksum=checksum,
        )
        _write_audit(audit, out_dir, "mortality")
        _echo("Done.")

    except click.UsageError:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        if os.environ.get("LIFEXP_DEBUG"):
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# lifexp lapse
# ---------------------------------------------------------------------------

@cli.command("lapse")
@click.option("--data",         default=None, help="Path to policies CSV file.")
@click.option("--study-start",  default=None, callback=_parse_date, is_eager=False,
              expose_value=True, help="Study start date (YYYY-MM-DD).")
@click.option("--study-end",    default=None, callback=_parse_date, is_eager=False,
              expose_value=True, help="Study end date (YYYY-MM-DD).")
@click.option("--output",       default=None, help="Output directory (default: results).")
@click.option("--format",       "fmt", default=None,
              help="Comma-separated output formats: html, excel (default: html,excel).")
@click.option("--config",       default=None, help="YAML config file.")
def cmd_lapse(data, study_start, study_end, output, fmt, config):
    """Run a lapse experience study."""
    try:
        cfg = _load_config(config)
        params = _merge(
            defaults={"output": "results", "fmt": "html,excel"},
            config=cfg,
            overrides={
                "data": data, "study_start": study_start,
                "study_end": study_end, "output": output, "fmt": fmt,
            },
        )

        if not params.get("data"):
            raise click.UsageError("--data is required (path to policies CSV).")
        if not params.get("study_start"):
            raise click.UsageError("--study-start is required (YYYY-MM-DD).")
        if not params.get("study_end"):
            raise click.UsageError("--study-end is required (YYYY-MM-DD).")

        s_start = params["study_start"]
        s_end   = params["study_end"]
        if isinstance(s_start, str):
            s_start = _parse_date(None, _FakeDateParam("study_start"), s_start)
        if isinstance(s_end, str):
            s_end = _parse_date(None, _FakeDateParam("study_end"), s_end)

        formats = [f.strip().lower() for f in str(params["fmt"]).split(",")]

        from lifexp.core.study_period import StudyPeriod
        from lifexp.lapse.study import LapseStudy
        from lifexp.core.audit import checksum_dataset, StudyRun
        import uuid
        from datetime import datetime

        _echo("lifexp lapse study")
        _echo(f"  Study period : {s_start} → {s_end}")

        dataset = _load_policy_data(params["data"])
        study   = StudyPeriod(s_start, s_end)

        _echo("  Computing ETR and lapse rates ...")
        ls = LapseStudy(dataset, study)
        results = ls.run()

        df = results.summary_df
        total_lapses   = float(df["lapses"].sum()) if "lapses" in df.columns else 0.0
        total_exposed  = float(df["policies_exposed"].sum()) if "policies_exposed" in df.columns else 0.0
        overall_rate   = total_lapses / total_exposed if total_exposed > 0 else 0.0
        _echo(f"  Overall lapse rate: {overall_rate:.4f}")

        out_dir = params["output"]
        _write_outputs(results, out_dir, formats, "lapse", "Lapse Study")

        checksum = checksum_dataset(dataset)
        audit = StudyRun(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            lifexp_version=lifexp.__version__,
            python_version=sys.version,
            study_type="lapse",
            parameters={
                "data": params["data"],
                "study_start": str(s_start),
                "study_end":   str(s_end),
            },
            data_checksum=checksum,
        )
        _write_audit(audit, out_dir, "lapse")
        _echo("Done.")

    except click.UsageError:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        if os.environ.get("LIFEXP_DEBUG"):
            traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# lifexp tables
# ---------------------------------------------------------------------------

@cli.command("tables")
@click.option("--list",  "do_list",  is_flag=True, help="List all registered tables.")
@click.option("--register", "reg_file", default=None,
              help="Path to a CSV file to register as a new table.")
@click.option("--name",  default=None, help="Name for the newly registered table.")
def cmd_tables(do_list, reg_file, name):
    """List or register mortality tables."""
    try:
        from lifexp.core.tables import TableRegistry

        reg = TableRegistry()

        if do_list:
            names = reg.list_tables()
            _echo(f"Registered tables ({len(names)}):")
            for n in names:
                _echo(f"  {n}")
            return

        if reg_file:
            if not name:
                raise click.UsageError("--name is required when using --register.")
            table = reg.load_from_csv(reg_file, name=name)
            reg.register(table)
            _echo(f"Registered table '{name}' from {reg_file}")
            _echo(f"  Age range: {table.age_min}–{table.age_max}")
            return

        raise click.UsageError("Specify --list or --register FILE --name NAME.")

    except click.UsageError:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utility: fake param object for config-file date coercion
# ---------------------------------------------------------------------------

class _FakeDateParam:
    """Minimal stand-in for click.core.Parameter used in _parse_date."""
    def __init__(self, name: str):
        self.name = name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
