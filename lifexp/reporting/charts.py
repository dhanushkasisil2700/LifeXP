"""Chart functions for experience study reporting.

All functions return a :class:`matplotlib.figure.Figure` and optionally
save to *output_path*.  ``plt.show()`` is never called — callers decide
whether to display or close.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe in tests & CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_return(fig: plt.Figure, output_path: Optional[str]) -> plt.Figure:
    """Save *fig* to *output_path* (creating dirs) then return it."""
    if output_path is not None:
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 1. A/E bar chart by age band
# ---------------------------------------------------------------------------

def plot_ae_by_age(
    ae_df: pd.DataFrame,
    title: str = "A/E Ratio by Age Band",
    output_path: Optional[str] = None,
    ci_column: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of Actual/Expected ratio by age band.

    Parameters
    ----------
    ae_df :
        DataFrame with columns ``age`` and ``ae_ratio``.  Optional column
        *ci_column* (e.g. ``'ci_half_width'``) provides ±symmetric
        confidence-interval whiskers.
    title :
        Chart title.
    output_path :
        If given, the figure is saved here (PNG/PDF/SVG auto-detected by
        extension).
    ci_column :
        Column name for symmetric CI half-widths; ignored when absent.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = ae_df.copy().dropna(subset=["age"])
    ages     = df["age"].tolist()
    ae_vals  = df["ae_ratio"].tolist() if "ae_ratio" in df.columns else [np.nan] * len(df)

    fig, ax = plt.subplots(figsize=(max(6, len(ages) * 0.55), 5))

    yerr = None
    if ci_column and ci_column in df.columns:
        yerr = df[ci_column].tolist()

    # Colour bars: red above 1.0, green below, grey for NaN
    colours = []
    for v in ae_vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            colours.append("#cccccc")
        elif v > 1.0:
            colours.append("#d94e4e")
        else:
            colours.append("#4caf77")

    x_pos = range(len(ages))
    bars = ax.bar(
        x_pos,
        [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0
         for v in ae_vals],
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
        zorder=2,
    )

    # Horizontal reference line at A/E = 1.0
    ax.axhline(1.0, color="#222222", linewidth=1.4, linestyle="--",
               label="A/E = 1.0", zorder=3)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(a) for a in ages], rotation=45 if len(ages) > 10 else 0)
    ax.set_xlabel("Age")
    ax.set_ylabel("A/E Ratio")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
    fig.tight_layout()

    return _save_and_return(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Crude vs graduated mortality rates
# ---------------------------------------------------------------------------

def plot_crude_vs_graduated(
    ages: Sequence[Union[int, float]],
    crude: Sequence[float],
    graduated: Sequence[float],
    title: str = "Crude vs Graduated Mortality Rates",
    log_scale: bool = False,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart comparing crude (scatter) and graduated (line) rates.

    Parameters
    ----------
    ages :
        Sequence of integer or fractional ages.
    crude :
        Crude mortality rates (plotted as dots).  May contain ``NaN``.
    graduated :
        Smoothed/graduated rates (plotted as a continuous line).
    title :
        Chart title.
    log_scale :
        If ``True``, the y-axis is log-scaled.
    output_path :
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ages_arr  = np.asarray(ages, dtype=float)
    crude_arr = np.asarray(crude, dtype=float)
    grad_arr  = np.asarray(graduated, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Graduated line
    ax.plot(ages_arr, grad_arr, color="#1a3a5c", linewidth=2.0,
            label="Graduated", zorder=3)

    # Crude dots (skip NaN for scatter)
    valid = ~np.isnan(crude_arr)
    ax.scatter(ages_arr[valid], crude_arr[valid], color="#d94e4e",
               s=40, zorder=4, label="Crude", marker="o")

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax.set_xlabel("Age")
    ax.set_ylabel("Mortality Rate" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle=":", linewidth=0.7, alpha=0.7)
    fig.tight_layout()

    return _save_and_return(fig, output_path)


# ---------------------------------------------------------------------------
# 3. A/E heatmap (age × policy year)
# ---------------------------------------------------------------------------

def plot_ae_heatmap(
    ae_surface: pd.DataFrame,
    title: str = "A/E Ratio Heatmap (Age × Policy Year)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Seaborn heatmap of A/E ratio over age and policy year.

    Parameters
    ----------
    ae_surface :
        DataFrame with rows = age bands, columns = policy years, values =
        A/E ratios.  ``NaN`` cells are rendered as grey (``mask``).
    title :
        Chart title.
    output_path :
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The diverging colormap is centred at **1.0** (not the data mean) so
    values above 1.0 are red and values below 1.0 are green.
    """
    n_rows, n_cols = ae_surface.shape
    fig_w = max(6, n_cols * 0.9 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mask = ae_surface.isna()

    # Determine symmetric vmin/vmax around 1.0
    finite_vals = ae_surface.values[~np.isnan(ae_surface.values)]
    if finite_vals.size > 0:
        max_dev = max(abs(finite_vals.max() - 1.0), abs(finite_vals.min() - 1.0), 0.01)
    else:
        max_dev = 0.5

    vmin = 1.0 - max_dev
    vmax = 1.0 + max_dev

    sns.heatmap(
        ae_surface,
        mask=mask,
        cmap="RdYlGn_r",       # red = high A/E, green = low
        center=1.0,
        vmin=vmin,
        vmax=vmax,
        annot=ae_surface.shape[0] * ae_surface.shape[1] <= 100,
        fmt=".2f",
        linewidths=0.3,
        linecolor="#dddddd",
        ax=ax,
        cbar_kws={"label": "A/E Ratio", "shrink": 0.8},
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Policy Year")
    ax.set_ylabel("Age")
    fig.tight_layout()

    return _save_and_return(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Kaplan-Meier survival curve
# ---------------------------------------------------------------------------

def plot_survival_curve(
    km_df: pd.DataFrame,
    title: str = "Kaplan-Meier Survival Curve",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Step-function Kaplan-Meier survival curve with optional confidence band.

    Parameters
    ----------
    km_df :
        DataFrame with column ``survival`` (required) and optional columns
        ``time`` (x-axis), ``ci_lower``, ``ci_upper``.  If ``time`` is
        absent, the row index is used.
    title :
        Chart title.
    output_path :
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = km_df.copy()
    t  = df["time"].to_numpy(dtype=float) if "time" in df.columns else df.index.to_numpy(dtype=float)
    s  = df["survival"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.step(t, s, where="post", color="#1a3a5c", linewidth=2.0,
            label="S(t)", zorder=3)

    if "ci_lower" in df.columns and "ci_upper" in df.columns:
        ci_lo = df["ci_lower"].to_numpy(dtype=float)
        ci_hi = df["ci_upper"].to_numpy(dtype=float)
        ax.fill_between(t, ci_lo, ci_hi, step="post",
                        alpha=0.2, color="#1a3a5c", label="95% CI")

    ax.set_xlim(left=t[0] if len(t) else 0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle=":", linewidth=0.7, alpha=0.7)
    fig.tight_layout()

    return _save_and_return(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Lapse funnel bar chart
# ---------------------------------------------------------------------------

def plot_lapse_funnel(
    lapse_df: pd.DataFrame,
    title: str = "In-Force Count by Policy Year",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal funnel bar chart of in-force count by policy year.

    Parameters
    ----------
    lapse_df :
        DataFrame with columns ``policy_year`` and ``if_count``.
    title :
        Chart title.
    output_path :
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = lapse_df.sort_values("policy_year").copy()
    years    = df["policy_year"].tolist()
    if_count = df["if_count"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(years) * 0.6), 5))

    # Gradient colours: darker bars for later years (more lapse attrition)
    n = max(len(years), 1)
    colours = [plt.cm.Blues(0.4 + 0.55 * i / (n - 1)) for i in range(n)] if n > 1 else ["#1a3a5c"]

    ax.bar(range(len(years)), if_count, color=colours, edgecolor="white",
           linewidth=0.6, zorder=2)

    # Annotate each bar with the count
    for i, cnt in enumerate(if_count):
        ax.text(i, cnt + max(if_count) * 0.01, f"{int(cnt):,}",
                ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Policy Year")
    ax.set_ylabel("In-Force Count")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
    fig.tight_layout()

    return _save_and_return(fig, output_path)
