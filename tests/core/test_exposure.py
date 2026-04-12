"""Tests for lifexp.core.exposure: central_etr and etr_summary."""

from datetime import date, timedelta

import pytest

from lifexp.core.data_model import PolicyDataset, PolicyRecord
from lifexp.core.date_utils import OutOfStudyError, days_in_study
from lifexp.core.exposure import central_etr, etr_comparison, etr_summary, initial_etr
from lifexp.core.study_period import AgeBasis, StudyPeriod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_record(**overrides) -> PolicyRecord:
    defaults = dict(
        policy_id="P001",
        date_of_birth=date(1986, 7, 1),
        issue_date=date(2010, 1, 1),
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


# Study window: exactly 365 days, birthday (Jul 1) falls on the first day
# so DOB=1986-07-01 stays age 35 throughout.
STUDY_365 = StudyPeriod(date(2021, 7, 1), date(2022, 6, 30))
DOB_AGE35 = date(1986, 7, 1)


# ---------------------------------------------------------------------------
# Normal: full year in-force, age 35 throughout
# ---------------------------------------------------------------------------

def test_full_year_inforce_single_age():
    """Policy in-force for the entire study; birthday on study start → age 35 throughout.

    The policy-year boundary (issue anniversary) may split the output into multiple
    rows, but all rows must have age=35 and the ETR must sum to a full year.
    """
    record = make_record(date_of_birth=DOB_AGE35, issue_date=date(2010, 1, 1))
    ds = PolicyDataset([record])
    result = central_etr(ds, STUDY_365, AgeBasis.LAST_BIRTHDAY, group_by=[])

    # All rows must be age 35 (birthday falls on study start — no age crossing)
    assert set(result["age"].tolist()) == {35}
    assert result["policy_count"].max() == 1
    # Total ETR for the 365-day study
    assert result["central_etr"].sum() == pytest.approx(365 / 365.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Normal: policy crosses an age boundary mid-study
# ---------------------------------------------------------------------------

def test_age_boundary_crossing():
    """Policy turns 36 on Jul 1; study is Jan–Dec 2021 (365 days)."""
    dob = date(1985, 7, 1)  # birthday 2021-07-01 falls mid-study
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = make_record(date_of_birth=dob)
    ds = PolicyDataset([record])
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert len(result) == 2, "Expected two age cells (35 and 36)"
    by_age = result.set_index("age")["central_etr"]

    # Age 35: Jan 1 → Jun 30 = 181 days (inclusive)
    assert by_age[35] == pytest.approx(181 / 365.25, rel=1e-4)
    # Age 36: Jul 1 → Dec 31 = 184 days (inclusive)
    assert by_age[36] == pytest.approx(184 / 365.25, rel=1e-4)
    # Total ≈ 1.0 (full year)
    assert by_age.sum() == pytest.approx(365 / 365.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Normal: death mid-year
# ---------------------------------------------------------------------------

def test_death_midyear():
    """Policy dies on Jul 1; should contribute fractional ETR up to exit date."""
    # DOB = 1981-01-01 → turns 40 on study start (Jan 1, 2021), so age 40 throughout.
    dob = date(1981, 1, 1)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = make_record(
        date_of_birth=dob,
        issue_date=date(2010, 1, 1),
        status="DEATH",
        exit_date=date(2021, 7, 1),
    )
    ds = PolicyDataset([record])
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["age"] == 40
    # Jan 1 → Jul 1 inclusive = 182 days
    expected_etr = 182 / 365.25
    assert row["central_etr"] == pytest.approx(expected_etr, rel=1e-4)
    # Approximately half a year
    assert 0.48 < row["central_etr"] < 0.52


# ---------------------------------------------------------------------------
# Normal: group-by segmentation (M/F split)
# ---------------------------------------------------------------------------

def test_group_by_gender():
    """ETR split by gender sums to the same total as no grouping."""
    study = STUDY_365
    dob = DOB_AGE35
    males = [
        make_record(policy_id=f"M{i}", date_of_birth=dob, gender="M") for i in range(3)
    ]
    females = [
        make_record(policy_id=f"F{i}", date_of_birth=dob, gender="F") for i in range(2)
    ]
    ds = PolicyDataset(males + females)

    result_grouped = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=["gender"])
    result_total = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    etr_m = result_grouped[result_grouped["gender"] == "M"]["central_etr"].sum()
    etr_f = result_grouped[result_grouped["gender"] == "F"]["central_etr"].sum()

    # 3 males × 365/365.25, 2 females × 365/365.25
    assert etr_m == pytest.approx(3 * 365 / 365.25, rel=1e-4)
    assert etr_f == pytest.approx(2 * 365 / 365.25, rel=1e-4)

    # Grouped total equals ungrouped total
    assert result_grouped["central_etr"].sum() == pytest.approx(
        result_total["central_etr"].sum(), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Edge: policy issued on study start date
# ---------------------------------------------------------------------------

def test_policy_issued_on_study_start():
    """issue_date == study.start_date → no double-counting, full ETR."""
    study = STUDY_365
    record = make_record(date_of_birth=DOB_AGE35, issue_date=study.start_date)
    ds = PolicyDataset([record])
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert len(result) == 1
    assert result.iloc[0]["central_etr"] == pytest.approx(365 / 365.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Edge: single-day observation
# ---------------------------------------------------------------------------

def test_single_day_observation():
    """Policy issued and exits on the same day → ETR = 1/365.25."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    day = date(2021, 6, 15)
    record = make_record(
        date_of_birth=date(1986, 1, 1),  # age 35 on that day
        issue_date=day,
        status="LAPSED",
        exit_date=day,
    )
    ds = PolicyDataset([record])
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert len(result) == 1
    assert result.iloc[0]["central_etr"] == pytest.approx(1 / 365.25, rel=1e-6)
    assert result.iloc[0]["central_etr"] > 0  # never negative or zero


# ---------------------------------------------------------------------------
# Edge: 100-policy aggregate
# ---------------------------------------------------------------------------

def test_100_identical_policies():
    """100 identical in-force policies → total ETR = 100 × (365/365.25).

    Policy-year splits are possible (issue anniversary mid-study), but the total
    ETR and maximum policy count per cell must both be correct.
    """
    study = STUDY_365
    records = [
        make_record(policy_id=f"P{i:04d}", date_of_birth=DOB_AGE35)
        for i in range(100)
    ]
    ds = PolicyDataset(records)
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    # All rows age 35, 100 distinct policies per row
    assert set(result["age"].tolist()) == {35}
    assert (result["policy_count"] == 100).all()
    # Total ETR across all rows = 100 full years
    assert result["central_etr"].sum() == pytest.approx(100 * 365 / 365.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Boundary: ETR sum invariant
#
# For any set of policies, the sum of age-cell ETRs must exactly equal the sum
# of individual observation-window lengths.  This is the invariant that
# guarantees every downstream A/E ratio is correct.
# ---------------------------------------------------------------------------

def test_etr_sum_invariant():
    """Sum of all age-cell ETRs equals the sum of raw observation-year lengths."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))

    # Three policies with different ages, exit types, and mid-study events
    records = [
        # P1: in-force all year, birthday Mar 15 mid-study (two age cells)
        make_record(
            policy_id="P1",
            date_of_birth=date(1985, 3, 15),
            issue_date=date(2015, 1, 1),
            status="IF",
        ),
        # P2: dies Aug 15, birthday Jul 1 (two age cells before death)
        make_record(
            policy_id="P2",
            date_of_birth=date(1980, 7, 1),
            issue_date=date(2018, 6, 1),
            status="DEATH",
            exit_date=date(2021, 8, 15),
        ),
        # P3: lapses Apr 30, birthday Nov 20 outside observation (single cell)
        make_record(
            policy_id="P3",
            date_of_birth=date(1990, 11, 20),
            issue_date=date(2019, 3, 1),
            status="LAPSED",
            exit_date=date(2021, 4, 30),
        ),
    ]
    # Hand-calculated expected values (days inclusive):
    # P1: Jan 1 → Dec 31 = 365 days
    # P2: Jan 1 → Aug 15 = 227 days
    # P3: Jan 1 → Apr 30 = 120 days
    # Total = 712 days / 365.25
    expected_total = 712 / 365.25

    ds = PolicyDataset(records)
    result = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert result["central_etr"].sum() == pytest.approx(expected_total, rel=1e-9)

    # Also verify programmatically by summing observation windows directly
    programmatic_total = 0.0
    for record in records:
        try:
            obs_start, obs_end = days_in_study(record, study)
            programmatic_total += ((obs_end - obs_start).days + 1) / 365.25
        except OutOfStudyError:
            pass

    assert result["central_etr"].sum() == pytest.approx(programmatic_total, rel=1e-9)


# ---------------------------------------------------------------------------
# etr_summary: collapse to age only
# ---------------------------------------------------------------------------

def test_etr_summary_collapses_dimensions():
    """etr_summary collapses policy_year and group_by fields, preserving ETR totals."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    # Two policies with different ages that cross a birthday boundary
    records = [
        make_record(policy_id="A", date_of_birth=date(1985, 7, 1)),  # crosses at Jul 1
        make_record(policy_id="B", date_of_birth=date(1980, 3, 10)),  # crosses at Mar 10
    ]
    ds = PolicyDataset(records)
    full = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    summary = etr_summary(full)

    assert list(summary.columns) == ["age", "central_etr", "policy_count"]
    assert summary["central_etr"].sum() == pytest.approx(full["central_etr"].sum(), rel=1e-9)
    assert summary["age"].is_monotonic_increasing


# ===========================================================================
# Initial ETR tests
# ===========================================================================

# Reuse: DOB_AGE35 = date(1986, 7, 1), STUDY_365 = 2021-07-01 to 2022-06-30

def _death_record(**overrides) -> PolicyRecord:
    """Helper: make a DEATH record with exit_date required."""
    return make_record(status="DEATH", **overrides)


# ---------------------------------------------------------------------------
# Normal: death at exact mid-year → initial_etr = 1.0 exactly
# ---------------------------------------------------------------------------

def test_initial_etr_death_midyear():
    """A mid-year death contributes central ≈ 0.498, but initial = 1.0 exactly."""
    # DOB=1981-01-01 → turns 40 on 2021-01-01 (study start), age 40 throughout
    dob = date(1981, 1, 1)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = _death_record(
        date_of_birth=dob,
        issue_date=date(2010, 1, 1),
        exit_date=date(2021, 7, 1),
    )
    ds = PolicyDataset([record])

    c = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert len(i) == 1
    assert i.iloc[0]["age"] == 40
    assert i.iloc[0]["deaths"] == 1
    # Central ETR: 182 days (Jan 1 → Jul 1 inclusive) / 365.25 ≈ 0.498
    assert c["central_etr"].sum() == pytest.approx(182 / 365.25, rel=1e-4)
    # Initial ETR: exactly 1.0, regardless of when in the year the death occurred
    assert i.iloc[0]["initial_etr"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Normal: death on day 1 of study → initial_etr = 1.0, central ≈ 0.003
# ---------------------------------------------------------------------------

def test_initial_etr_death_day_one():
    """Death on the very first day: central ≈ 1/365.25, initial must still be 1.0."""
    dob = date(1981, 1, 1)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = _death_record(
        date_of_birth=dob,
        issue_date=date(2010, 1, 1),
        exit_date=date(2021, 1, 1),  # dies on day 1
    )
    ds = PolicyDataset([record])

    c = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert c["central_etr"].sum() == pytest.approx(1 / 365.25, rel=1e-4)
    assert i.iloc[0]["initial_etr"] == pytest.approx(1.0, abs=1e-12)
    assert i.iloc[0]["deaths"] == 1


# ---------------------------------------------------------------------------
# Normal: death on last day → initial_etr = 1.0, central ≈ 1.0
# ---------------------------------------------------------------------------

def test_initial_etr_death_last_day():
    """Death on final day of study: central ≈ 0.9993, initial = 1.0; correction ≈ 0."""
    dob = date(1981, 1, 1)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = _death_record(
        date_of_birth=dob,
        issue_date=date(2010, 1, 1),
        exit_date=date(2021, 12, 31),  # dies on last day
    )
    ds = PolicyDataset([record])

    c = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    # central ≈ 365/365.25 — almost 1.0 but not quite
    assert c["central_etr"].sum() == pytest.approx(365 / 365.25, rel=1e-4)
    # initial is exactly 1.0; the correction (≈ 0.0007) is tiny but present
    assert i.iloc[0]["initial_etr"] == pytest.approx(1.0, abs=1e-12)
    assert i.iloc[0]["deaths"] == 1


# ---------------------------------------------------------------------------
# Normal: lapse has no initial ETR correction
# ---------------------------------------------------------------------------

def test_initial_etr_lapse_equals_central():
    """A lapse contributes fractional ETR only; initial_etr == central_etr."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    record = make_record(
        date_of_birth=date(1981, 1, 1),
        issue_date=date(2010, 1, 1),
        status="LAPSED",
        exit_date=date(2021, 7, 1),
    )
    ds = PolicyDataset([record])

    c = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert i["deaths"].sum() == 0
    assert i["initial_etr"].sum() == pytest.approx(c["central_etr"].sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# Edge: multiple deaths in same age cell → initial_etr = deaths × 1.0
# ---------------------------------------------------------------------------

def test_initial_etr_multiple_deaths_same_cell():
    """3 deaths all at age 45 (different dates): initial_etr = 3.0, deaths = 3."""
    # DOB=1976-01-01 → turns 45 on 2021-01-01 (study start), stays age 45 all year
    dob = date(1976, 1, 1)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    issue = date(2010, 1, 1)
    records = [
        _death_record(policy_id="D1", date_of_birth=dob, issue_date=issue,
                      exit_date=date(2021, 3, 1)),
        _death_record(policy_id="D2", date_of_birth=dob, issue_date=issue,
                      exit_date=date(2021, 6, 15)),
        _death_record(policy_id="D3", date_of_birth=dob, issue_date=issue,
                      exit_date=date(2021, 10, 31)),
    ]
    ds = PolicyDataset(records)

    i = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    # All three deaths fall in the age-45 cell
    age45 = i[i["age"] == 45]
    assert len(age45) >= 1
    assert age45["deaths"].sum() == 3
    assert age45["initial_etr"].sum() == pytest.approx(3.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Boundary: initial_etr >= central_etr always (1000-policy invariant)
# ---------------------------------------------------------------------------

def test_initial_etr_invariant_always_ge_central():
    """For any dataset, initial_etr >= central_etr in every (age, policy_year) cell.

    Study is in 2021 (non-leap year) so no age cell exceeds 365 days, guaranteeing
    that the 1.0 per-death initial ETR is never below the fractional central ETR.
    """
    import random as _random

    rng = _random.Random(99)
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))

    statuses_with_exit = ["LAPSED", "DEATH", "SURRENDERED"]
    records = []
    for k in range(1000):
        yr = rng.randint(1950, 1985)
        mo = rng.randint(1, 12)
        # Avoid Feb-29 DOBs to keep the test focused on the invariant
        day = rng.randint(1, 28)
        dob = date(yr, mo, day)

        issue_yr = rng.randint(2010, 2020)
        issue = date(issue_yr, rng.randint(1, 12), rng.randint(1, 28))

        use_exit = rng.random() < 0.5
        status = rng.choice(statuses_with_exit) if use_exit else "IF"
        if use_exit:
            exit_d = study.start_date + timedelta(days=rng.randint(0, 364))
        else:
            exit_d = None

        records.append(make_record(
            policy_id=f"R{k:04d}",
            date_of_birth=dob,
            issue_date=issue,
            status=status,
            exit_date=exit_d,
        ))

    ds = PolicyDataset(records)
    c_df = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i_df = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    # Merge on (age, policy_year) for cell-level comparison
    merged = c_df.merge(i_df, on=["age", "policy_year"], suffixes=("_c", "_i"))
    for _, row in merged.iterrows():
        assert row["initial_etr"] >= row["central_etr"] - 1e-12, (
            f"Invariant violated at age={row['age']} py={row['policy_year']}: "
            f"initial={row['initial_etr']:.6f} < central={row['central_etr']:.6f}"
        )


# ---------------------------------------------------------------------------
# Boundary: zero-death portfolio → initial_etr == central_etr everywhere
# ---------------------------------------------------------------------------

def test_initial_etr_zero_deaths_equals_central():
    """With no death records, initial_etr must equal central_etr in every cell."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    records = [
        make_record(policy_id="P1", date_of_birth=date(1985, 3, 10), status="IF"),
        make_record(policy_id="P2", date_of_birth=date(1980, 9, 20),
                    status="LAPSED", exit_date=date(2021, 6, 30)),
        make_record(policy_id="P3", date_of_birth=date(1990, 11, 1),
                    status="SURRENDERED", exit_date=date(2021, 9, 15)),
    ]
    ds = PolicyDataset(records)

    c_df = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i_df = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])

    assert i_df["deaths"].sum() == 0
    assert i_df["initial_etr"].sum() == pytest.approx(c_df["central_etr"].sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# etr_comparison: side-by-side view
# ---------------------------------------------------------------------------

def test_etr_comparison_structure():
    """etr_comparison produces correctly structured output with difference column."""
    study = StudyPeriod(date(2021, 1, 1), date(2021, 12, 31))
    dob = date(1981, 1, 1)
    records = [
        make_record(policy_id="IF1", date_of_birth=dob, status="IF"),
        _death_record(policy_id="D1", date_of_birth=dob,
                      issue_date=date(2010, 1, 1), exit_date=date(2021, 7, 1)),
    ]
    ds = PolicyDataset(records)

    c_df = central_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    i_df = initial_etr(ds, study, AgeBasis.LAST_BIRTHDAY, group_by=[])
    comp = etr_comparison(c_df, i_df)

    assert list(comp.columns) == ["age", "deaths", "central_etr", "initial_etr", "difference"]
    # difference = initial - central >= 0 everywhere
    assert (comp["difference"] >= -1e-12).all()
    # For age 40: 1 death → initial > central; difference > 0
    row40 = comp[comp["age"] == 40].iloc[0]
    assert row40["deaths"] == 1
    assert row40["difference"] > 0
