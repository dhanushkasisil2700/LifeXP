"""Generate a synthetic Sri Lankan life portfolio for end-to-end testing.

Run directly to (re-)generate the fixture files:

    python tests/fixtures/generate_portfolio.py

Or import :func:`build_portfolio` from tests to get DataFrames directly.

Design
------
* 5,000 policies spanning issue years 2010–2022
* Status mix  : ~70 % IF, ~15 % LAPSED, ~10 % DEATH, ~5 % SURRENDERED
* Age at issue : 20–65, truncated-normal (mean 40, sd 12)
* Products     : ENDOW 40 %, TERM 35 %, WHOLELIFE 25 %
* Gender       : 60 % M, 40 % F
* Mortality    : A 1967-70 × 0.85  (15 % lighter than standard)
* Lapse rates  : 15 % yr-1, 8 % yr-2, 5 % thereafter
* 500 morbidity claim records linked to IF policies

Edge cases seeded deliberately
--------------------------------
* Policy issued exactly on study start date (2018-01-01)
* Death on second day of study (2018-01-02 exit)
* Policy with Feb-29 birthday (1992-02-29)
* Centenarian policy (DOB 1916-01-01 → age ~101 at study start; triggers
  AgeOutOfRangeError gracefully)
* Sparse age cell: single policy aged 19 at issue
* All-same-product segment for group_by=['product_code'] round-trip
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED        = 42
N_POLICIES  = 5_000
STUDY_START = date(2018, 1, 1)
STUDY_END   = date(2022, 12, 31)

_STATUS_PROBS   = [0.70, 0.15, 0.10, 0.05]
_STATUS_LABELS  = ["IF", "LAPSED", "DEATH", "SURRENDERED"]
_PRODUCT_PROBS  = [0.40, 0.35, 0.25]
_PRODUCT_LABELS = ["ENDOW", "TERM", "WHOLELIFE"]
_GENDER_PROBS   = [0.60, 0.40]
_GENDER_LABELS  = ["M", "F"]
_CHANNEL_LABELS = ["AGENT", "DIRECT", "BANCASSURANCE"]
_CHANNEL_PROBS  = [0.50, 0.30, 0.20]

_MORT_MULTIPLIER = 0.85   # 15 % lighter than standard A 1967-70


# ---------------------------------------------------------------------------
# Load actual A 1967-70 qx values at import time
# (avoids duplicating hard-coded numbers)
# ---------------------------------------------------------------------------

def _load_a1967_70() -> Dict[int, float]:
    # Add project root to path if running as __main__
    root = str(Path(__file__).parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from lifexp.core.tables import TableRegistry
    reg = TableRegistry()
    t   = reg.get("A_1967_70")
    return {age: t.qx(age) for age in range(t.age_min, t.age_max + 1)}


_A1967_70: Dict[int, float] = {}   # filled lazily on first call to build_portfolio


def _qx(age: int) -> float:
    """Return A 1967-70 qx × 0.85 for *age*, clamped to table range."""
    global _A1967_70
    if not _A1967_70:
        _A1967_70 = _load_a1967_70()
    age_c = max(min(age, max(_A1967_70)), min(_A1967_70))
    return _A1967_70[age_c] * _MORT_MULTIPLIER


def _lapse_rate(policy_year: int) -> float:
    if policy_year == 1:
        return 0.15
    if policy_year == 2:
        return 0.08
    return 0.05


def _anniversary(base: date, year_offset: int) -> date:
    """Safe year-offset that handles Feb-29 birthdays."""
    try:
        return base.replace(year=base.year + year_offset)
    except ValueError:
        return base.replace(year=base.year + year_offset, day=28)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def build_portfolio(
    seed: int = SEED,
    n: int = N_POLICIES,
    study_start: date = STUDY_START,
    study_end:   date = STUDY_END,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(policies_df, claims_df)`` DataFrames.

    Parameters
    ----------
    seed  : Random seed for reproducibility.
    n     : Total number of policies to generate.
    study_start, study_end : Observation window for the E2E tests.

    Returns
    -------
    policies_df : DataFrame with one row per policy.
    claims_df   : DataFrame with morbidity claim rows (≤ 500).
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Issue dates: uniform across 2010-01-01 to study_end
    # ------------------------------------------------------------------
    min_issue = date(2010, 1, 1)
    issue_day_range = (study_end - min_issue).days
    issue_offsets = rng.integers(0, issue_day_range, size=n)
    issue_dates = [min_issue + timedelta(days=int(d)) for d in issue_offsets]

    # ------------------------------------------------------------------
    # 2. Ages at issue: truncated-normal mean=40, sd=12, range [20, 65]
    # ------------------------------------------------------------------
    raw_ages = rng.normal(40, 12, size=n)
    ages_at_issue = np.clip(np.round(raw_ages), 20, 65).astype(int)

    # ------------------------------------------------------------------
    # 3. DOB: birth_year = issue_year − age, random month/day
    # ------------------------------------------------------------------
    dobs = []
    _days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(n):
        issue = issue_dates[i]
        age   = int(ages_at_issue[i])
        birth_year  = issue.year - age
        birth_month = int(rng.integers(1, 13))
        birth_day   = int(rng.integers(1, _days_in_month[birth_month] + 1))
        dobs.append(date(birth_year, birth_month, birth_day))

    # ------------------------------------------------------------------
    # 4. Attributes
    # ------------------------------------------------------------------
    products  = rng.choice(_PRODUCT_LABELS, size=n, p=_PRODUCT_PROBS)
    genders   = rng.choice(_GENDER_LABELS,  size=n, p=_GENDER_PROBS)
    channels  = rng.choice(_CHANNEL_LABELS, size=n, p=_CHANNEL_PROBS)

    sa_by_product = {"ENDOW": 500_000.0, "TERM": 1_000_000.0, "WHOLELIFE": 750_000.0}
    sum_assureds   = np.array([
        sa_by_product[p] * (0.5 + rng.random()) for p in products
    ])
    annual_premiums = sum_assureds * 0.02 * (0.8 + 0.4 * rng.random(n))

    # ------------------------------------------------------------------
    # 5. Status simulation: year-by-year from issue to study_end
    # ------------------------------------------------------------------
    statuses     = ["IF"] * n
    exit_dates   = [None] * n
    exit_reasons = [None] * n

    for i in range(n):
        issue = issue_dates[i]
        age0  = int(ages_at_issue[i])

        py = 0
        seg_start = issue

        while seg_start <= study_end:
            py += 1
            seg_end = _anniversary(issue, py)   # next policy anniversary

            # Current age at start of this segment
            current_age = age0 + py - 1

            # Mortality draw (use A_1967_70 × 0.85)
            qx = _qx(min(current_age, 90))
            if rng.random() < qx:
                # Place death uniformly within this segment
                seg_days = max((min(seg_end, study_end) - seg_start).days, 1)
                d_offset = int(rng.integers(1, seg_days + 1))
                ed = seg_start + timedelta(days=d_offset)
                if ed > issue and ed >= date(issue.year, issue.month, issue.day) + timedelta(days=1):
                    statuses[i]     = "DEATH"
                    exit_dates[i]   = ed
                    exit_reasons[i] = "DEATH"
                    break

            # Lapse draw
            lapse_prob = _lapse_rate(py)
            if rng.random() < lapse_prob:
                seg_days = max((min(seg_end, study_end) - seg_start).days, 1)
                l_offset = int(rng.integers(1, seg_days + 1))
                ed = seg_start + timedelta(days=l_offset)
                if ed > issue:
                    lapse_kind      = rng.choice(["LAPSED", "SURRENDERED"], p=[0.75, 0.25])
                    statuses[i]     = lapse_kind
                    exit_dates[i]   = ed
                    exit_reasons[i] = lapse_kind
                    break

            seg_start = seg_end

    # ------------------------------------------------------------------
    # 6. Deterministic edge cases (override indices 0–4)
    # ------------------------------------------------------------------

    # a) Policy issued exactly on study start → full 5-year observation
    issue_dates[0]  = study_start
    dobs[0]         = date(study_start.year - 35, 3, 15)
    statuses[0]     = "IF"
    exit_dates[0]   = None
    exit_reasons[0] = None

    # b) Death on second day of study
    issue_dates[1]  = date(2015, 6, 1)
    dobs[1]         = date(1975, 6, 1)
    statuses[1]     = "DEATH"
    exit_dates[1]   = study_start + timedelta(days=1)   # 2018-01-02
    exit_reasons[1] = "DEATH"

    # c) Feb-29 birthday
    issue_dates[2]  = date(2016, 5, 1)
    dobs[2]         = date(1992, 2, 29)
    statuses[2]     = "IF"
    exit_dates[2]   = None
    exit_reasons[2] = None

    # d) Centenarian (age ~101 at study start) → AgeOutOfRangeError in MortalityStudy
    issue_dates[3]  = date(2017, 1, 1)
    dobs[3]         = date(1916, 1, 1)
    statuses[3]     = "IF"
    exit_dates[3]   = None
    exit_reasons[3] = None

    # e) Sparse age cell: single policy aged 19 at issue (below normal range)
    issue_dates[4]  = date(2018, 6, 1)
    dobs[4]         = date(1999, 6, 1)
    statuses[4]     = "IF"
    exit_dates[4]   = None
    exit_reasons[4] = None

    # ------------------------------------------------------------------
    # 7. Assemble DataFrame
    # ------------------------------------------------------------------
    rows = []
    for i in range(n):
        rows.append({
            "policy_id":      f"LK{i+1:06d}",
            "date_of_birth":  dobs[i].isoformat(),
            "issue_date":     issue_dates[i].isoformat(),
            "gender":         str(genders[i]),
            "smoker_status":  "NS",
            "sum_assured":    round(float(sum_assureds[i]), 2),
            "annual_premium": round(float(annual_premiums[i]), 2),
            "product_code":   str(products[i]),
            "channel":        str(channels[i]),
            "status":         statuses[i],
            "exit_date":      exit_dates[i].isoformat() if exit_dates[i] else None,
            "exit_reason":    exit_reasons[i],
        })

    policies_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 8. Morbidity claims: 500 records on IF policies within study window
    # ------------------------------------------------------------------
    if_pids = policies_df.loc[policies_df["status"] == "IF", "policy_id"].tolist()
    n_claims = min(500, len(if_pids))
    claim_pids = rng.choice(if_pids, size=n_claims, replace=False)

    claim_rows = []
    window_days = (study_end - study_start).days - 30
    for j, pid in enumerate(claim_pids):
        offset      = int(rng.integers(0, max(window_days, 1)))
        c_start     = study_start + timedelta(days=offset)
        dur_days    = int(rng.integers(14, 180))
        c_end       = min(c_start + timedelta(days=dur_days), study_end)
        claim_rows.append({
            "claim_id":           f"CLM{j+1:06d}",
            "policy_id":          pid,
            "claim_start_date":   c_start.isoformat(),
            "claim_end_date":     c_end.isoformat(),
            "claim_status":       "CLOSED_RECOVERY",
            "benefit_type":       "PERIODIC",
            "claim_amount":       round(float(rng.uniform(10_000, 100_000)), 2),
            "benefit_period_days": (c_end - c_start).days,
        })

    claims_df = pd.DataFrame(claim_rows)
    return policies_df, claims_df


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------

def save_portfolio(output_dir: str = None) -> Tuple[str, str]:
    """Generate and save portfolio CSVs; return ``(policies_path, claims_path)``."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent)
    os.makedirs(output_dir, exist_ok=True)
    pol_path = os.path.join(output_dir, "portfolio_policies.csv")
    clm_path = os.path.join(output_dir, "portfolio_claims.csv")
    policies_df, claims_df = build_portfolio()
    policies_df.to_csv(pol_path, index=False)
    claims_df.to_csv(clm_path, index=False)
    print(f"Written {len(policies_df):,} policies  → {pol_path}")
    print(f"Written {len(claims_df):,} claims    → {clm_path}")
    return pol_path, clm_path


if __name__ == "__main__":
    save_portfolio()
