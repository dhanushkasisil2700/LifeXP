"""Tests for lifexp.core.tables: MortalityTable and TableRegistry."""

import warnings
from pathlib import Path

import pytest

from lifexp.core.tables import (
    AgeOutOfRangeError,
    DataError,
    MortalityTable,
    TableRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(["age,qx"] + rows))
    return path


def _make_simple_table(tmp_path: Path, name: str, basis: str = "custom") -> MortalityTable:
    """Create and load a minimal 10-age table (ages 40–49)."""
    rows = [f"{a},{0.00179 * (1.09 ** (a - 40)):.5f}" for a in range(40, 50)]
    p = _write_csv(tmp_path / f"{name}.csv", rows)
    return TableRegistry.load_from_csv(p, name, "ultimate", basis)


# ---------------------------------------------------------------------------
# Normal: A 1967-70 qx lookup at exact integer ages
# ---------------------------------------------------------------------------

def test_a1967_70_qx_age_40():
    """Auto-registered table returns qx(40) = 0.00179 (published CMI value)."""
    reg = TableRegistry()
    table = reg.get("A_1967_70")
    assert table.qx(40) == pytest.approx(0.00179, rel=1e-6)


def test_a1967_70_spot_checks():
    """Spot-check A 1967-70 at ages 30, 40, 50, 60, 70 against published CMI values."""
    reg = TableRegistry()
    table = reg.get("A_1967_70")

    expected = {
        30: 0.00097,
        40: 0.00179,
        50: 0.00427,
        60: 0.01075,
        70: 0.02744,
    }
    for age, qx_pub in expected.items():
        assert table.qx(age) == pytest.approx(qx_pub, rel=1e-4), (
            f"A 1967-70 qx({age}): expected {qx_pub}, got {table.qx(age)}"
        )


# ---------------------------------------------------------------------------
# Normal: linear interpolation at fractional age
# ---------------------------------------------------------------------------

def test_linear_interpolation_at_405():
    """Linear interpolation at 40.5 is the exact midpoint of qx(40) and qx(41)."""
    reg = TableRegistry()
    table = reg.get("A_1967_70")

    qx_40 = table.qx(40)
    qx_41 = table.qx(41)
    qx_mid = table.interpolate(40.5, method="linear")

    # Must lie strictly between the two neighbours
    assert qx_40 < qx_mid < qx_41
    # Linear interpolation at the exact midpoint
    assert qx_mid == pytest.approx((qx_40 + qx_41) / 2.0, rel=1e-9)


def test_cubic_interpolation_within_bounds():
    """Cubic interpolation at 40.5 also lies between qx(40) and qx(41)."""
    reg = TableRegistry()
    table = reg.get("A_1967_70")

    qx_40 = table.qx(40)
    qx_41 = table.qx(41)
    qx_cubic = table.interpolate(40.5, method="cubic")

    # Cubic may differ from linear but should stay in a reasonable neighbourhood
    assert qx_40 * 0.9 < qx_cubic < qx_41 * 1.1


# ---------------------------------------------------------------------------
# Normal: custom CSV load and registration
# ---------------------------------------------------------------------------

def test_custom_csv_load_and_get(tmp_path):
    """Load a user-supplied CSV and retrieve it from the registry."""
    rows = [f"{a},{a * 0.0005:.5f}" for a in range(30, 41)]
    csv_path = _write_csv(tmp_path / "my_table.csv", rows)

    reg = TableRegistry()
    table = TableRegistry.load_from_csv(str(csv_path), "my_table", "ultimate", "custom")
    reg.register(table)

    retrieved = reg.get("my_table")
    assert retrieved.name == "my_table"
    assert retrieved.age_min == 30
    assert retrieved.age_max == 40
    assert "my_table" in reg.list_tables()


# ---------------------------------------------------------------------------
# Edge: age out of range
# ---------------------------------------------------------------------------

def test_qx_age_above_max_raises():
    """qx(150) on A 1967-70 (max 90) raises AgeOutOfRangeError mentioning the age."""
    table = TableRegistry().get("A_1967_70")
    with pytest.raises(AgeOutOfRangeError, match="150"):
        table.qx(150)


def test_qx_age_below_min_raises():
    """qx(10) on A 1967-70 (min 17) raises AgeOutOfRangeError."""
    table = TableRegistry().get("A_1967_70")
    with pytest.raises(AgeOutOfRangeError, match="10"):
        table.qx(10)


def test_interpolate_out_of_range_raises():
    """interpolate at a fractional age outside the table range raises AgeOutOfRangeError."""
    table = TableRegistry().get("A_1967_70")
    with pytest.raises(AgeOutOfRangeError):
        table.interpolate(95.5, method="linear")


# ---------------------------------------------------------------------------
# Edge: duplicate registration warns and overwrites
# ---------------------------------------------------------------------------

def test_duplicate_registration_warns_and_overwrites(tmp_path):
    """Registering a second table under the same name: warning issued, value updated."""
    t1 = _make_simple_table(tmp_path, "dup_test", basis="basis_1")
    t2 = _make_simple_table(tmp_path, "dup_test", basis="basis_2")

    reg = TableRegistry()
    reg.register(t1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.register(t2)

    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "Expected a UserWarning when overwriting an existing table"
    )
    # Second registration must have actually overwritten the first
    assert reg.get("dup_test").basis == "basis_2"


# ---------------------------------------------------------------------------
# Edge: CSV with missing ages raises DataError
# ---------------------------------------------------------------------------

def test_csv_missing_ages_raises_data_error(tmp_path):
    """A CSV with an age gap triggers DataError listing the missing ages."""
    # Skips ages 42 and 43
    rows = ["40,0.00179", "41,0.00195", "44,0.00253", "45,0.00276"]
    csv_path = _write_csv(tmp_path / "gapped.csv", rows)

    with pytest.raises(DataError) as exc_info:
        TableRegistry.load_from_csv(str(csv_path), "gapped", "ultimate", "custom")

    error_text = str(exc_info.value)
    assert "42" in error_text and "43" in error_text


# ---------------------------------------------------------------------------
# Extra: mu method is positive and increases with age
# ---------------------------------------------------------------------------

def test_mu_positive_and_increasing():
    """Force of mortality is positive and broadly increasing across the table range."""
    table = TableRegistry().get("A_1967_70")

    mu_values = [table.mu(float(a)) for a in range(30, 71)]
    assert all(m > 0 for m in mu_values)
    # Overall trend must be increasing (last value > first)
    assert mu_values[-1] > mu_values[0]


# ---------------------------------------------------------------------------
# Extra: auto-registration via 'import lifexp'
# ---------------------------------------------------------------------------

def test_auto_registration_via_package_import():
    """import lifexp is sufficient to make A_1967_70 available in a fresh registry."""
    import importlib
    import lifexp  # noqa: F401 — triggers registration

    importlib.reload(lifexp)  # ensure we're testing the live state, not caches
    reg = TableRegistry()
    assert "A_1967_70" in reg.list_tables()
    assert reg.get("A_1967_70").age_max == 90
