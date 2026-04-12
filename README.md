# lifexp

**Actuarial Life Experience Analysis Toolkit**

`lifexp` is a Python library and command-line tool for performing rigorous actuarial experience studies on life insurance portfolios. It covers the full analytical pipeline — from raw policy data ingestion through exposure calculation, A/E analysis, graduation, and professional report generation.

Built for Sri Lankan and broader emerging-market life portfolios, with first-class support for the A 1967-70 standard table.

---

## Features

### Experience Studies
| Module | What it does |
|--------|-------------|
| **Mortality** | Central and initial exposed-to-risk (ETR), A/E ratios against standard tables, Garwood Poisson confidence intervals, group-by segmentation |
| **Lapse** | Policy-year lapse and surrender rates, exposure by duration, year-on-year trend analysis |
| **Morbidity** | Incidence and termination rates using the dual-ETR (healthy/sick) method, multi-state model support |
| **Reinsurance** | Ceded vs. retained exposure split, RI-adjusted A/E, treaty-level summaries |
| **Expense** | Per-policy and per-new-business unit costs, year-on-year inflation analysis, A/E vs. expense assumptions |
| **Commission** | Anomaly detection (Z-score, MAD, IQR, ensemble), agent-level commission review |

### Actuarial Methods
- **Exposed-to-risk**: Central ETR (force of mortality basis) and initial ETR (annual probability basis)
- **Age bases**: Last birthday, next birthday, nearest birthday
- **Graduation**: Whittaker-Henderson 1D and 2D, cubic splines, parametric (Makeham, Gompertz)
- **Graduation diagnostics**: Chi-squared, serial correlation, sign and runs tests
- **Credibility**: Classical and Bühlmann-Straub credibility weighting
- **Mortality projection**: Lee-Carter style future projection

### Standard Tables
- **A 1967-70** — built-in (no setup required)
- Custom tables loadable from CSV via `TableRegistry`

### Reporting
- **HTML reports** — self-contained, embedded MathJax for actuarial formulae, styled A/E tables
- **Excel reports** — openpyxl-based, bold headers, conditional formatting (A/E > 1.2 red, < 0.8 green)
- **Charts** — 5 matplotlib/seaborn chart types: A/E by age bar chart, crude vs. graduated rates, A/E heatmap (center-locked at 1.0), survival curve, lapse funnel

### Audit Trail
- `StudyRun` dataclass captures run ID, timestamp, parameters, data checksum (MD5, order-independent), execution time
- `compare_runs()` detects parameter changes and data drift between study runs
- Audit JSON written automatically alongside every CLI output

---

## Installation

### Requirements
- Python 3.9+
- Dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`, `click>=8.0`, `pyyaml`

### Install from source

```bash
git clone https://github.com/dhanushkasisil/lifexp.git
cd lifexp
pip install -e .
```

### Install dependencies only

```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl click pyyaml
```

---

## Quick Start

### Python API

```python
import pandas as pd
from datetime import date
from lifexp import PolicyDataset, StudyPeriod, MortalityStudy, HTMLReport, ExcelReport

# 1. Load your policy data
df = pd.read_csv("policies.csv")
dataset = PolicyDataset.from_dataframe(df)

# 2. Define the study window
study = StudyPeriod(date(2018, 1, 1), date(2022, 12, 31))

# 3. Run a mortality experience study
ms = MortalityStudy(dataset, study, standard_table="A_1967_70")
results = ms.run()

print(f"Overall A/E ratio : {results.overall_ae:.4f}")
print(f"Total deaths      : {results.total_deaths:.0f}")
print(f"Total ETR (years) : {results.total_etr:.1f}")

# 4. View results
print(results.summary_df.head())
print(results.ae_by_age())

# 5. Confidence intervals (Garwood exact Poisson)
print(results.confidence_interval(level=0.95))

# 6. Graduate the crude rates
print(results.graduate(method="whittaker", lam=100.0))

# 7. Export reports
html = HTMLReport("Mortality Study 2018–2022", {})
html.add_ae_table(results.summary_df, caption="A/E by Age")
html.render("output/mortality.html")

xl = ExcelReport()
xl.add_ae_sheet(results.summary_df, sheet_name="Mortality A/E")
xl.render("output/mortality.xlsx")
```

### Segmented analysis (by gender, product, etc.)

```python
ms = MortalityStudy(
    dataset, study,
    standard_table="A_1967_70",
    group_by=["gender", "product_code"]
)
results = ms.run()
print(results.summary_df)
```

### Lapse study

```python
from lifexp import LapseStudy

ls = LapseStudy(dataset, study)
lapse_results = ls.run()
print(lapse_results.summary_df)
```

### Morbidity study

```python
from lifexp import MorbidityStudy, ClaimDataset

claims_df = pd.read_csv("claims.csv")
claims = ClaimDataset.from_dataframe(claims_df)

ms = MorbidityStudy(dataset, claims, study)
morb_results = ms.run()
print(morb_results.incidence_df)
```

### Custom mortality table

```python
from lifexp import TableRegistry

reg = TableRegistry()
table = reg.load_from_csv("my_table.csv", name="MY_TABLE_2020")
reg.register(table)

# Now use it in any study
ms = MortalityStudy(dataset, study, standard_table="MY_TABLE_2020")
```

---

## CLI Usage

After installation, the `lifexp` command is available on your PATH.

### Mortality study

```bash
lifexp mortality \
  --data policies.csv \
  --study-start 2018-01-01 \
  --study-end   2022-12-31 \
  --table A_1967_70 \
  --output results/ \
  --format html,excel
```

### Lapse study

```bash
lifexp lapse \
  --data policies.csv \
  --study-start 2018-01-01 \
  --study-end   2022-12-31 \
  --output results/
```

### List registered tables

```bash
lifexp tables --list
```

### Register a custom table

```bash
lifexp tables --register my_table.csv --name MY_TABLE_2020
```

### Using a YAML config file

```bash
lifexp mortality --config study_config.yaml
```

**study_config.yaml** example:

```yaml
data: data/policies_2022.csv
study_start: "2018-01-01"
study_end:   "2022-12-31"
table: A_1967_70
output: results/
fmt: html,excel
```

CLI options always override config file values. Config file values override built-in defaults.

### Print version

```bash
lifexp version
```

---

## Input Data Format

### Policy data CSV (`policies.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `policy_id` | string | Unique policy identifier |
| `date_of_birth` | YYYY-MM-DD | Date of birth |
| `issue_date` | YYYY-MM-DD | Policy issue date |
| `gender` | M / F | Gender |
| `smoker_status` | S / NS | Smoker status |
| `sum_assured` | float | Sum assured amount |
| `annual_premium` | float | Annual premium |
| `product_code` | string | Product identifier (e.g. ENDOW, TERM, WHOLELIFE) |
| `channel` | string | Distribution channel |
| `status` | IF / LAPSED / DEATH / SURRENDERED | Current policy status |
| `exit_date` | YYYY-MM-DD or blank | Date of exit (blank for in-force) |
| `exit_reason` | string or blank | Reason for exit |

### Claims data CSV (`claims.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `claim_id` | string | Unique claim identifier |
| `policy_id` | string | Linked policy |
| `claim_start_date` | YYYY-MM-DD | Claim commencement date |
| `claim_end_date` | YYYY-MM-DD | Claim end date |
| `claim_status` | string | e.g. CLOSED_RECOVERY, OPEN |
| `benefit_type` | string | LUMP_SUM / PERIODIC |
| `claim_amount` | float | Total claim amount |
| `benefit_period_days` | int | Duration of benefit payment |

---

## Project Structure

```
lifexp/
├── core/
│   ├── data_model.py       # PolicyRecord, PolicyDataset, ClaimDataset
│   ├── exposure.py         # Central and initial ETR computation
│   ├── study_period.py     # StudyPeriod, AgeBasis
│   ├── tables.py           # MortalityTable, TableRegistry, A_1967_70
│   ├── credibility.py      # Classical and Bühlmann-Straub credibility
│   ├── segmentation.py     # Group-by helper utilities
│   ├── date_utils.py       # Date arithmetic helpers
│   └── audit.py            # StudyRun, @audit_run decorator, compare_runs
│
├── mortality/
│   ├── study.py            # MortalityStudy, MortalityResults
│   └── projection.py       # Mortality projection (Lee-Carter style)
│
├── lapse/
│   └── study.py            # LapseStudy, LapseResults
│
├── morbidity/
│   ├── study.py            # MorbidityStudy, MorbidityResults
│   └── multistate.py       # Multi-state model
│
├── reinsurance/
│   └── study.py            # RIStudy, RIResults
│
├── expense/
│   ├── study.py            # ExpenseStudy, ExpenseResults
│   └── commission.py       # CommissionStudy, anomaly detection
│
├── graduation/
│   ├── whittaker.py        # Whittaker-Henderson 1D
│   ├── whittaker_2d.py     # Whittaker-Henderson 2D
│   ├── splines.py          # Cubic spline graduation
│   ├── parametric.py       # Makeham/Gompertz parametric fitting
│   └── diagnostics.py      # Chi-squared, serial correlation, runs tests
│
├── reporting/
│   ├── html_report.py      # HTMLReport (MathJax, CSS)
│   ├── excel_report.py     # ExcelReport (openpyxl)
│   └── charts.py           # 5 matplotlib/seaborn chart functions
│
└── cli.py                  # Click CLI entry point
```

---

## Running the Tests

```bash
# Full test suite
pytest tests/ -v

# E2E test only (synthetic 5,000-policy Sri Lankan portfolio)
pytest tests/test_e2e.py -v

# Specific module
pytest tests/mortality/ -v
```

The test suite includes 439 tests covering unit, integration, and end-to-end scenarios. A synthetic 5,000-policy Sri Lankan life portfolio is used for E2E validation, verifying A/E ratios, lapse rates, audit determinism, and full pipeline runtime.

---

## Generating a Synthetic Test Portfolio

```bash
python tests/fixtures/generate_portfolio.py
```

This writes `tests/fixtures/portfolio_policies.csv` and `portfolio_claims.csv` — a 5,000-policy synthetic Sri Lankan portfolio with realistic status mix, mortality (A 1967-70 × 0.85), lapse rates (15%/8%/5% by year), and 500 morbidity claims. Includes deliberate edge cases: centenarian policy, Feb-29 birthday, sparse age cell, and study-start issue date.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**W M Dhanushka S B Wijekoon**
Actuarial Analyst - Model Development

| | |
|---|---|
| Phone | +94 72 115 4664 / +94 74 322 5717 |
| Email | [dhanushkasisil@outlook.com](mailto:dhanushkasisil@outlook.com) |
| LinkedIn | [linkedin.com/in/dhanushkasisil](https://lk.linkedin.com/in/dhanushkasisil) |

---

*lifexp is an independent open-source project*
