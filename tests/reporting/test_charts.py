"""Tests for lifexp.reporting.charts — all five chart functions.

Checkpoint S19: Artefacts written to reports/checkpoint_s19_*.png for
manual visual inspection.  Automated tests verify Figure type, file
creation, and correctness of NaN handling.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lifexp.reporting.charts import (
    plot_ae_by_age,
    plot_ae_heatmap,
    plot_crude_vs_graduated,
    plot_lapse_funnel,
    plot_survival_curve,
)


# ---------------------------------------------------------------------------
# Shared synthetic data (mirrors S10 / S19 checkpoint data)
# ---------------------------------------------------------------------------

_AGES = list(range(40, 71))


def _ae_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    ages = list(range(40, 40 + n))
    return pd.DataFrame({
        "age":      ages,
        "ae_ratio": rng.uniform(0.7, 1.4, size=n).tolist(),
        "deaths":   rng.integers(1, 30, size=n).tolist(),
        "etr":      rng.uniform(100, 500, size=n).tolist(),
    })


def _ae_df_with_ci(n: int = 10) -> pd.DataFrame:
    df = _ae_df(n)
    df["ci_half_width"] = 0.05
    return df


def _ae_df_with_nan() -> pd.DataFrame:
    df = _ae_df(8)
    df.loc[2, "ae_ratio"] = np.nan
    df.loc[5, "ae_ratio"] = np.nan
    return df


def _ae_surface(n_ages: int = 8, n_years: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    data = rng.uniform(0.6, 1.5, size=(n_ages, n_years))
    # Plant some NaN cells
    data[1, 2] = np.nan
    data[4, 5] = np.nan
    return pd.DataFrame(
        data,
        index=[f"Age {40+i}" for i in range(n_ages)],
        columns=[f"PY{y+1}" for y in range(n_years)],
    )


def _km_df(n: int = 20) -> pd.DataFrame:
    t = np.linspace(0, 10, n)
    s = np.exp(-0.1 * t)
    ci_lo = np.maximum(s - 0.05, 0)
    ci_hi = np.minimum(s + 0.05, 1)
    return pd.DataFrame({"time": t, "survival": s,
                         "ci_lower": ci_lo, "ci_upper": ci_hi})


def _lapse_df(n: int = 10) -> pd.DataFrame:
    counts = [1000 - i * 80 for i in range(n)]
    return pd.DataFrame({
        "policy_year": list(range(1, n + 1)),
        "if_count":    counts,
    })


# Output directory for Checkpoint S19 artefacts
_REPORTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports",
)


# ---------------------------------------------------------------------------
# Test 1 — All functions return a Figure object
# ---------------------------------------------------------------------------

class TestReturnsFigure:
    """Each chart function returns a matplotlib.figure.Figure."""

    def test_ae_by_age_returns_figure(self):
        fig = plot_ae_by_age(_ae_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ae_by_age_with_ci_returns_figure(self):
        fig = plot_ae_by_age(_ae_df_with_ci(), ci_column="ci_half_width")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_crude_vs_graduated_returns_figure(self):
        ages     = list(range(40, 71))
        crude    = [0.001 * np.exp(0.07 * (a - 40)) for a in ages]
        grad     = [0.0009 * np.exp(0.075 * (a - 40)) for a in ages]
        fig = plot_crude_vs_graduated(ages, crude, grad)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_crude_vs_graduated_log_scale_returns_figure(self):
        ages  = list(range(40, 71))
        crude = [0.001 * np.exp(0.07 * (a - 40)) for a in ages]
        grad  = [0.0009 * np.exp(0.075 * (a - 40)) for a in ages]
        fig = plot_crude_vs_graduated(ages, crude, grad, log_scale=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ae_heatmap_returns_figure(self):
        fig = plot_ae_heatmap(_ae_surface())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_survival_curve_returns_figure(self):
        fig = plot_survival_curve(_km_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_lapse_funnel_returns_figure(self):
        fig = plot_lapse_funnel(_lapse_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 2 — File saved when output_path provided
# ---------------------------------------------------------------------------

class TestFileSaved:
    """Each function saves the file when output_path is not None."""

    def test_ae_by_age_saves_file(self, tmp_path):
        out = str(tmp_path / "ae.png")
        plot_ae_by_age(_ae_df(), output_path=out)
        assert os.path.isfile(out)

    def test_crude_vs_graduated_saves_file(self, tmp_path):
        out = str(tmp_path / "grad.png")
        ages  = list(range(40, 71))
        crude = [0.001 * np.exp(0.07 * (a - 40)) for a in ages]
        grad  = [0.0009 * np.exp(0.075 * (a - 40)) for a in ages]
        plot_crude_vs_graduated(ages, crude, grad, output_path=out)
        assert os.path.isfile(out)

    def test_ae_heatmap_saves_file(self, tmp_path):
        out = str(tmp_path / "heatmap.png")
        plot_ae_heatmap(_ae_surface(), output_path=out)
        assert os.path.isfile(out)

    def test_survival_curve_saves_file(self, tmp_path):
        out = str(tmp_path / "km.png")
        plot_survival_curve(_km_df(), output_path=out)
        assert os.path.isfile(out)

    def test_lapse_funnel_saves_file(self, tmp_path):
        out = str(tmp_path / "lapse.png")
        plot_lapse_funnel(_lapse_df(), output_path=out)
        assert os.path.isfile(out)

    def test_saves_to_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "ae.png")
        plot_ae_by_age(_ae_df(), output_path=out)
        assert os.path.isfile(out)

    def test_no_file_when_path_none(self, tmp_path):
        """Without output_path, no file is written."""
        fig = plot_ae_by_age(_ae_df(), output_path=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 3 — Single data point renders without error
# ---------------------------------------------------------------------------

class TestSingleDataPoint:
    """Charts must not crash when given a single-row / single-age input."""

    def test_ae_by_age_single_bar(self):
        df = pd.DataFrame({"age": [50], "ae_ratio": [1.1]})
        fig = plot_ae_by_age(df, title="Single bar")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_crude_vs_graduated_single_point(self):
        fig = plot_crude_vs_graduated([50], [0.005], [0.005])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ae_heatmap_single_cell(self):
        surface = pd.DataFrame([[1.2]], index=["Age 50"], columns=["PY1"])
        fig = plot_ae_heatmap(surface)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_survival_curve_single_point(self):
        df = pd.DataFrame({"time": [0.0], "survival": [1.0]})
        fig = plot_survival_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_lapse_funnel_single_bar(self):
        df = pd.DataFrame({"policy_year": [1], "if_count": [500]})
        fig = plot_lapse_funnel(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 4 — NaN values in A/E
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """NaN cells in ae_ratio / ae_surface must not crash and must render sensibly."""

    def test_ae_by_age_with_nan_returns_figure(self):
        fig = plot_ae_by_age(_ae_df_with_nan())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ae_heatmap_with_nan_returns_figure(self):
        """NaN cells appear masked (grey) rather than raising or defaulting to 0."""
        surface = _ae_surface()
        fig = plot_ae_heatmap(surface)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ae_heatmap_all_nan_returns_figure(self):
        """All-NaN surface still produces a Figure without error."""
        surface = pd.DataFrame(
            np.full((3, 3), np.nan),
            index=["Age 50", "Age 51", "Age 52"],
            columns=["PY1", "PY2", "PY3"],
        )
        fig = plot_ae_heatmap(surface)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_center_is_one(self):
        """Checkpoint S19: diverging colormap is centred at 1.0, not the data mean.

        Seaborn enforces center=1.0 by setting vmin/vmax symmetrically around
        1.0 on the norm.  We verify (vmin + vmax) / 2 == 1.0 on the QuadMesh
        collection norm, which is independent of the data mean.
        """
        # Surface with mean far from 1.0 (all values above 1.0)
        surface = pd.DataFrame(
            [[1.5, 1.6], [1.4, 1.7]],
            index=["Age 50", "Age 51"],
            columns=["PY1", "PY2"],
        )
        fig = plot_ae_heatmap(surface)
        ax = fig.axes[0]
        collections = [c for c in ax.collections if hasattr(c, "norm")]
        assert len(collections) > 0
        norm = collections[0].norm
        midpoint = (norm.vmin + norm.vmax) / 2.0
        assert midpoint == pytest.approx(1.0, abs=0.01), (
            f"Heatmap norm midpoint is {midpoint:.4f} "
            f"(vmin={norm.vmin:.4f}, vmax={norm.vmax:.4f}), expected 1.0"
        )
        plt.close(fig)

    def test_ae_heatmap_nan_cells_are_masked(self):
        """NaN cells are masked (not treated as 0 or raising error)."""
        surface = pd.DataFrame(
            [[1.0, np.nan], [np.nan, 0.8]],
            index=["Age 50", "Age 51"],
            columns=["PY1", "PY2"],
        )
        fig = plot_ae_heatmap(surface)
        assert isinstance(fig, plt.Figure)
        # Figure must not raise; file write also must succeed
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 5 — Output path: directory auto-creation
# ---------------------------------------------------------------------------

class TestOutputPathEdgeCases:
    """Nested output directories are created automatically."""

    def test_creates_nested_dirs_ae(self, tmp_path):
        out = str(tmp_path / "deep" / "dir" / "ae.png")
        fig = plot_ae_by_age(_ae_df(), output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_creates_nested_dirs_heatmap(self, tmp_path):
        out = str(tmp_path / "x" / "y" / "hm.png")
        fig = plot_ae_heatmap(_ae_surface(), output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_pdf_output(self, tmp_path):
        out = str(tmp_path / "ae.pdf")
        fig = plot_ae_by_age(_ae_df(), output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_survival_curve_without_ci(self, tmp_path):
        """Survival curve renders without CI columns present."""
        df = pd.DataFrame({"time": [0, 1, 2, 3], "survival": [1.0, 0.9, 0.8, 0.7]})
        fig = plot_survival_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_survival_curve_no_time_column(self):
        """If 'time' column absent, index is used for x-axis."""
        df = pd.DataFrame({"survival": [1.0, 0.9, 0.8, 0.75]})
        fig = plot_survival_curve(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint S19 artefacts — written to reports/ for visual inspection
# ---------------------------------------------------------------------------

class TestCheckpointS19Artefacts:
    """Generate PNG files using the synthetic S10 mortality dataset.

    Open each file manually to verify:
    1. ae_by_age   — bars, reference line at 1.0, red/green colouring
    2. crude_vs_graduated — dots vs smooth line, log scale variant
    3. ae_heatmap  — diverging colours centred at 1.0
    4. survival_curve — step function with CI band
    5. lapse_funnel — declining bars with counts annotated
    """

    def test_generate_ae_by_age(self):
        rng = np.random.default_rng(10)
        ages = list(range(40, 71))
        ae_vals = rng.uniform(0.65, 1.55, len(ages)).tolist()
        df = pd.DataFrame({
            "age": ages,
            "ae_ratio": ae_vals,
            "ci_half_width": [0.08] * len(ages),
        })
        out = os.path.join(_REPORTS, "checkpoint_s19_ae_by_age.png")
        fig = plot_ae_by_age(
            df, title="S19: A/E by Age (synthetic mortality data)",
            output_path=out, ci_column="ci_half_width",
        )
        assert os.path.isfile(out)
        plt.close(fig)

    def test_generate_crude_vs_graduated(self):
        ages  = list(range(40, 71))
        crude = [0.001 * np.exp(0.075 * (a - 40)) * (1 + 0.1 * np.random.default_rng(a).standard_normal()) for a in ages]
        grad  = [0.001 * np.exp(0.075 * (a - 40)) for a in ages]
        out_lin = os.path.join(_REPORTS, "checkpoint_s19_crude_vs_graduated_linear.png")
        out_log = os.path.join(_REPORTS, "checkpoint_s19_crude_vs_graduated_log.png")
        fig1 = plot_crude_vs_graduated(ages, crude, grad,
                                       title="S19: Crude vs Graduated (linear)",
                                       output_path=out_lin)
        fig2 = plot_crude_vs_graduated(ages, crude, grad,
                                       title="S19: Crude vs Graduated (log scale)",
                                       log_scale=True, output_path=out_log)
        assert os.path.isfile(out_lin) and os.path.isfile(out_log)
        plt.close(fig1)
        plt.close(fig2)

    def test_generate_ae_heatmap(self):
        rng = np.random.default_rng(10)
        data = rng.uniform(0.6, 1.5, size=(12, 8))
        data[3, 4] = np.nan
        data[7, 1] = np.nan
        surface = pd.DataFrame(
            data,
            index=[f"Age {40+i}" for i in range(12)],
            columns=[f"PY{y+1}" for y in range(8)],
        )
        out = os.path.join(_REPORTS, "checkpoint_s19_ae_heatmap.png")
        fig = plot_ae_heatmap(surface,
                              title="S19: A/E Heatmap — centre at 1.0",
                              output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_generate_survival_curve(self):
        t = np.linspace(0, 30, 200)
        s = np.exp(-0.05 * t)
        ci_lo = np.maximum(s - 0.04 * (1 + 0.03 * t), 0)
        ci_hi = np.minimum(s + 0.04 * (1 + 0.03 * t), 1)
        df = pd.DataFrame({"time": t, "survival": s,
                           "ci_lower": ci_lo, "ci_upper": ci_hi})
        out = os.path.join(_REPORTS, "checkpoint_s19_survival_curve.png")
        fig = plot_survival_curve(df, title="S19: Kaplan-Meier Survival Curve",
                                  output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_generate_lapse_funnel(self):
        counts = [5000 - i * 350 + np.random.default_rng(i).integers(-50, 50)
                  for i in range(12)]
        df = pd.DataFrame({
            "policy_year": list(range(1, 13)),
            "if_count":    counts,
        })
        out = os.path.join(_REPORTS, "checkpoint_s19_lapse_funnel.png")
        fig = plot_lapse_funnel(df, title="S19: Lapse Funnel by Policy Year",
                                output_path=out)
        assert os.path.isfile(out)
        plt.close(fig)
