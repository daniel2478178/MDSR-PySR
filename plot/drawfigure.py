import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PLOT_CACHE = Path(tempfile.gettempdir()) / "mdsr_plot_cache"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PLOT_CACHE / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, MaxNLocator


plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EXPECTED_TASK_IDS = [f"P{number:02d}" for number in range(1, 60)]
FAILURE_SENTINEL = -np.inf
FIGURE_HEIGHT_SCALE = 1.25


def taller_figsize(width, height):
    return width, height * FIGURE_HEIGHT_SCALE


A4_FIGSIZE = taller_figsize(190 / 25.4, 78 / 25.4)
A4_LANDSCAPE_FIGSIZE = taller_figsize(267 / 25.4, 78 / 25.4)
A4_DPI = 800
STRUCTURE_LEVEL_ALIASES = {
    "High": "高度相似",
    "Relatively High": "较高相似",
    "Medium": "中等相似",
    "Low": "较低相似",
    "Dissimilar": "不相似",
}


def format_a4_figure(fig):
    """Keep the three paired figures readable at portrait A4 print width."""
    for ax in fig.axes:
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.tick_params(labelsize=7)
        for text in ax.texts:
            text.set_fontsize(7)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(7)
        for line in ax.lines:
            line.set_linewidth(1.2)
            line.set_markersize(4)
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PathCollection):
                collection.set_sizes([15])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate the MDSR summary figures.",
    )
    parser.add_argument(
        "--without-parameters",
        type=Path,
        default=BASE_DIR / "不带参数.xlsx",
    )
    parser.add_argument(
        "--with-parameters",
        type=Path,
        default=BASE_DIR / "带参数统计表格(1).xlsx",
    )
    parser.add_argument(
        "--perfect-fit",
        type=Path,
        default=(
            PROJECT_ROOT
            / "equation_verfication"
            / "physicsMDSR_Range_GenerationFormula_noise_metrics.xlsx"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "mdsr_figures",
    )
    return parser


args = build_parser().parse_args()

input_path = args.without_parameters.resolve()
with_parameter_input_path = args.with_parameters.resolve()
perfect_fit_input_path = args.perfect_fit.resolve()
out_dir = args.output_dir.resolve()

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 80)
print("MDSR Figure Generator")
print("=" * 80)
print(f"Input file : {input_path}")
print(f"Comparison file: {with_parameter_input_path}")
print(f"Perfect-fit file: {perfect_fit_input_path}")
print(f"Output dir : {out_dir}")
print("=" * 80)


# ============================================================
# Read Excel
# ============================================================

df = pd.read_excel(
    input_path,
    sheet_name=0,
    engine="openpyxl",
)
if "structure_similarity_level" not in df.columns:
    df = df.rename(
        columns={"structure_similarity_leve": "structure_similarity_level"}
    )

with_parameter_df = pd.read_excel(
    with_parameter_input_path,
    sheet_name="Sheet1",
    engine="openpyxl",
)

for frame in (df, with_parameter_df):
    if "structure_similarity_level" in frame.columns:
        frame["structure_similarity_level"] = frame[
            "structure_similarity_level"
        ].replace(STRUCTURE_LEVEL_ALIASES)

perfect_fit_df = pd.read_excel(
    perfect_fit_input_path,
    sheet_name="Sheet1",
    engine="openpyxl",
)

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")


# ============================================================
# Required columns
# ============================================================

required_columns = [
    "ID",
    "structure_similarity_score",
    "structure_similarity_level",
    "complexity",
    "simp_complexity",
    "shared_fit_r2_0_7",
    "WASS_0_R2",
    "WASS_025_R2",
    "WASS_04_R2",
    "noise_0_R2",
    "noise_001_R2",
    "noise_003_R2",
    "noise_005_R2",
    "noise_01_R2",
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    print("\nMissing columns:")

    for c in missing_columns:
        print(f"  - {c}")

    print("\nAvailable columns:")

    for c in df.columns:
        print(f"  {c}")

    raise ValueError(
        "Some required columns are missing. "
        "Check the column names above."
    )

comparison_columns = [
    "complexity",
    "structure_similarity_score",
    "structure_similarity_level",
    "shared_fit_r2_0_7",
    "WASS_0_R2",
    "WASS_025_R2",
    "WASS_04_R2",
]

missing_comparison_columns = [
    column
    for column in comparison_columns
    if column not in with_parameter_df.columns
]

if missing_comparison_columns:
    raise ValueError(
        "Missing comparison columns in "
        f"{with_parameter_input_path.name}: "
        f"{missing_comparison_columns}"
    )

perfect_fit_noise_keys = [
    "0",
    "001",
    "003",
    "005",
    "01",
]

perfect_fit_r2_columns = [
    f"noise_{key}_R2"
    for key in perfect_fit_noise_keys
]

perfect_fit_nmse_columns = [
    f"noise_{key}_NMSE"
    for key in perfect_fit_noise_keys
]

perfect_fit_mse_columns = [
    f"noise_{key}_MSE"
    for key in perfect_fit_noise_keys
]

missing_perfect_fit_columns = [
    column
    for column in [
        "ID",
        *perfect_fit_r2_columns,
        *perfect_fit_nmse_columns,
        *perfect_fit_mse_columns,
    ]
    if column not in perfect_fit_df.columns
]

if missing_perfect_fit_columns:
    raise ValueError(
        "Missing perfect-fit columns in "
        f"{perfect_fit_input_path.name}: "
        f"{missing_perfect_fit_columns}"
    )


def add_missing_failed_equations(frame, metric_columns, source_name):
    """Add absent P01-P59 tasks as failed equations."""
    normalized = frame.copy()
    normalized["ID"] = normalized["ID"].astype(str).str.strip()

    duplicate_ids = normalized.loc[
        normalized["ID"].duplicated(),
        "ID",
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            f"{source_name}: duplicate task IDs {duplicate_ids}"
        )

    expected_ids = set(EXPECTED_TASK_IDS)
    present_ids = set(normalized["ID"])
    unexpected_ids = sorted(present_ids - expected_ids)
    if unexpected_ids:
        raise ValueError(
            f"{source_name}: unexpected task IDs {unexpected_ids}"
        )

    missing_ids = [
        task_id
        for task_id in EXPECTED_TASK_IDS
        if task_id not in present_ids
    ]

    normalized = (
        normalized
        .set_index("ID")
        .reindex(EXPECTED_TASK_IDS)
    )
    normalized.index.name = "ID"

    if missing_ids:
        normalized.loc[missing_ids, metric_columns] = FAILURE_SENTINEL
        normalized.loc[
            missing_ids,
            "structure_similarity_level",
        ] = "Failed"

    print(
        f"Added failed equations to {source_name}: "
        f"{missing_ids or 'none'}"
    )

    return normalized.reset_index(), missing_ids


without_parameter_metric_columns = [
    column
    for column in required_columns
    if column not in {"ID", "structure_similarity_level"}
]
with_parameter_metric_columns = [
    column
    for column in comparison_columns
    if column not in {"ID", "structure_similarity_level"}
]

df, without_parameter_missing_ids = add_missing_failed_equations(
    df,
    without_parameter_metric_columns,
    input_path.name,
)
with_parameter_df, with_parameter_missing_ids = add_missing_failed_equations(
    with_parameter_df,
    with_parameter_metric_columns,
    with_parameter_input_path.name,
)


# ============================================================
# Convert columns
# ============================================================

ids = (
    df["ID"]
    .astype(str)
    .to_numpy()
)

def numeric_column(name):

    return pd.to_numeric(
        df[name],
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )


def comparison_numeric_column(name):

    return pd.to_numeric(
        with_parameter_df[name],
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )


sim = numeric_column(
    "structure_similarity_score"
)

with_parameter_sim = comparison_numeric_column(
    "structure_similarity_score"
)

complexity = numeric_column(
    "complexity"
)

with_parameter_complexity = comparison_numeric_column(
    "complexity"
)

simp_complexity = numeric_column(
    "simp_complexity"
)

shared_r2 = numeric_column(
    "shared_fit_r2_0_7"
)

with_parameter_shared_r2 = comparison_numeric_column(
    "shared_fit_r2_0_7"
)

wass0 = numeric_column(
    "WASS_0_R2"
)

wass025 = numeric_column(
    "WASS_025_R2"
)

wass04 = numeric_column(
    "WASS_04_R2"
)

noise0 = numeric_column(
    "noise_0_R2"
)

noise001 = numeric_column(
    "noise_001_R2"
)

noise003 = numeric_column(
    "noise_003_R2"
)

noise005 = numeric_column(
    "noise_005_R2"
)

noise01 = numeric_column(
    "noise_01_R2"
)


n_tasks = len(df)

print(
    f"Number of tasks: {n_tasks}"
)


# ============================================================
# Utility functions
# ============================================================

INVALID_R2_LIMIT = -1e20


def valid_r2(arr):
    """
    Remove NaN, inf, and failure sentinel values
    such as -1e100.
    """

    arr = np.asarray(
        arr,
        dtype=float,
    )

    return arr[
        np.isfinite(arr)
        & (arr > INVALID_R2_LIMIT)
    ]


def median_iqr(arr):
    """
    Return median, Q1, Q3 using valid R² values.
    """

    v = valid_r2(arr)

    if len(v) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return (
        float(
            np.median(v)
        ),
        float(
            np.percentile(
                v,
                25,
            )
        ),
        float(
            np.percentile(
                v,
                75,
            )
        ),
    )


def success_rate(
    arr,
    threshold=0.9,
):
    """
    Percentage of ALL tasks satisfying R² >= threshold.

    Invalid values and sentinel values count as failures.
    """

    arr = np.asarray(
        arr,
        dtype=float,
    )

    success = (
        np.isfinite(arr)
        & (arr > INVALID_R2_LIMIT)
        & (arr >= threshold)
    )

    return (
        100.0
        * np.sum(success)
        / len(arr)
    )


def add_bar_labels_inside(
    ax,
    bars,
    values,
    fontsize=10,
):
    """
    Put value labels inside bars near the top.
    """

    for bar, value in zip(
        bars,
        values,
    ):

        height = bar.get_height()

        if height <= 0:
            continue

        offset = max(
            0.35,
            height * 0.06,
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height - offset,

            f"{value}",

            ha="center",
            va="top",

            fontsize=fontsize,
            fontweight="bold",
            color="white",
        )


# ============================================================
# R² transformed axis utilities
#
# Internal position:
#
#     y = -log10(1 - R²)
#
# This spreads values near R² = 1,
# but axis labels will show actual R² values.
# ============================================================


def r2_to_strength(r2):
    """
    Convert R² to transformed plotting coordinate.

    R² = 0.9      -> 1
    R² = 0.99     -> 2
    R² = 0.999    -> 3
    ...
    """

    if not np.isfinite(r2):
        return np.nan

    r2 = min(
        float(r2),
        1.0,
    )

    residual = max(
        1.0 - r2,
        1e-15,
    )

    return (
        -math.log10(residual)
    )


# Ticks shown as actual R² values.
r2_tick_values = [
    0.0,
    0.9,
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.999999,
    0.9999999,
    0.99999999,
    0.999999999,
]

r2_tick_positions = [
    r2_to_strength(r2)
    for r2 in r2_tick_values
]

r2_tick_labels = [
    "0",
    "0.9",
    "0.99",
    "0.999",
    "0.9999",
    "0.99999",
    "0.999999",
    "0.9999999",
    "0.99999999",
    "0.999999999",
]


def set_r2_transformed_axis(ax):
    """
    Apply transformed R² coordinate positions,
    but display actual R² tick labels.
    """

    ax.set_yticks(
        r2_tick_positions,
        r2_tick_labels,
    )

    ax.set_ylabel(
        r"$R^2$"
    )


# ============================================================
# Structural similarity categories
# ============================================================

level_order_en = [
    "High",
    "Relatively High",
    "Medium",
    "Low",
]

level_labels_with_criteria = [
    "Failed\n($S < 0$)",
    "Low\n($0$–$44$)",
    "Medium\n($45$–$74$)",
    "Relatively High\n($75$–$89$)",
    "High\n($90$–$100$)",
]

def structure_distribution_counts(scores):
    valid = scores[np.isfinite(scores) & (scores >= 0)]
    if np.any(valid > 100):
        raise ValueError("Structural similarity score exceeds 100.")
    counts = np.histogram(valid, bins=[0, 45, 75, 90, 100.0000001])[0]
    return [len(scores) - len(valid), *counts.tolist()]


fig1_level_counts = structure_distribution_counts(sim)
fig1_with_parameter_level_counts = structure_distribution_counts(
    with_parameter_sim
)
level_counts = list(reversed(fig1_level_counts[1:]))


# ============================================================
# Figure 1
# Structural similarity distribution
# ============================================================

fig, ax = plt.subplots(
    figsize=taller_figsize(9.2, 4.8)
)

x_levels = np.arange(
    len(level_labels_with_criteria)
)
bar_width = 0.36

without_parameter_bars = ax.bar(
    x_levels - bar_width / 2,
    fig1_level_counts,
    width=bar_width,
    label="Without parameters",
)

with_parameter_bars = ax.bar(
    x_levels + bar_width / 2,
    fig1_with_parameter_level_counts,
    width=bar_width,
    label="With parameters",
)

ax.set_xticks(
    x_levels,
    level_labels_with_criteria,
)

ax.set_ylabel(
    "Number of tasks"
)

ax.set_xlabel(
    "Structural similarity level"
)

ax.set_title(
    "Distribution of structural similarity levels"
)

ax.set_axisbelow(
    True
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.tick_params(
    axis="x",
    rotation=10,
)

add_bar_labels_inside(
    ax,
    without_parameter_bars,
    fig1_level_counts,
    fontsize=10,
)

add_bar_labels_inside(
    ax,
    with_parameter_bars,
    fig1_with_parameter_level_counts,
    fontsize=10,
)

all_level_counts = (
    fig1_level_counts
    + fig1_with_parameter_level_counts
)

if all_level_counts:

    ax.set_ylim(
        0,
        max(all_level_counts) * 1.05,
    )

ax.legend()

fig.tight_layout()

f1 = (
    out_dir
    / "Fig1_structure_similarity_distribution.png"
)

fig.savefig(
    f1,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Noise statistics
# ============================================================

noise_series = [
    ("0%", noise0),
    ("1%", noise001),
    ("3%", noise003),
    ("5%", noise005),
    ("10%", noise01),
]

noise_labels = [
    x[0]
    for x in noise_series
]

perfect_fit_outliers = (
    perfect_fit_df[perfect_fit_r2_columns] < 0
).all(axis=1)

clean_perfect_fit_df = perfect_fit_df.loc[
    ~perfect_fit_outliers
].copy()

perfect_fit_mean_mse = (
    clean_perfect_fit_df[perfect_fit_mse_columns]
    .mean()
    .to_numpy(dtype=float)
)

perfect_fit_mean_nmse = (
    clean_perfect_fit_df[perfect_fit_nmse_columns]
    .mean()
    .to_numpy(dtype=float)
)

perfect_fit_clean_mean_r2 = (
    1.0 - perfect_fit_mean_nmse
)

perfect_fit_clean_success = (
    clean_perfect_fit_df[perfect_fit_r2_columns]
    .ge(0.9)
    .mean()
    .to_numpy(dtype=float)
    * 100.0
)

perfect_fit_summary = pd.DataFrame({
    "noise": noise_labels,
    "mean_MSE": perfect_fit_mean_mse,
    "mean_NMSE": perfect_fit_mean_nmse,
    "mean_R2": perfect_fit_clean_mean_r2,
    "success_rate_percent": perfect_fit_clean_success,
})

print(
    "Perfect-fit outliers retained for median/IQR only: "
    f"{perfect_fit_df.loc[perfect_fit_outliers, 'ID'].tolist()}"
)
print(
    perfect_fit_summary.to_string(
        index=False,
    )
)

perfect_fit_r2_stats = [
    median_iqr(
        perfect_fit_df[column]
    )
    for column in perfect_fit_r2_columns
]

perfect_fit_median_r2 = [
    values[0]
    for values in perfect_fit_r2_stats
]

perfect_fit_q1_r2 = [
    values[1]
    for values in perfect_fit_r2_stats
]

perfect_fit_q3_r2 = [
    values[2]
    for values in perfect_fit_r2_stats
]

noise_stats = [
    median_iqr(
        x[1]
    )
    for x in noise_series
]

noise_median = [
    x[0]
    for x in noise_stats
]

noise_q1 = [
    x[1]
    for x in noise_stats
]

noise_q3 = [
    x[2]
    for x in noise_stats
]

noise_success = [
    success_rate(
        x[1],
        threshold=0.9,
    )
    for x in noise_series
]

xn = np.arange(
    len(noise_labels)
)


# ============================================================
# Figure 2
# Noise vs median R² + IQR
# ============================================================

fig, ax = plt.subplots(
    figsize=taller_figsize(7.6, 4.8)
)

ax.plot(
    xn,
    noise_median,
    marker="o",
    linewidth=2,
    label="Discovered equations",
)

ax.fill_between(
    xn,
    noise_q1,
    noise_q3,
    alpha=0.20,
    color="C0",
    label="Discovered-equation IQR",
)

ax.plot(
    xn,
    perfect_fit_median_r2,
    linestyle="--",
    marker="s",
    linewidth=2,
    color="C1",
    label="Perfect equations",
)

ax.fill_between(
    xn,
    perfect_fit_q1_r2,
    perfect_fit_q3_r2,
    facecolor="none",
    edgecolor="C1",
    linewidth=1.2,
    linestyle=":",
    label="Perfect-equation IQR",
)

ax.set_xticks(
    xn,
    noise_labels,
)

ax.set_xlabel(
    "Relative noise level"
)

ax.set_ylabel(
    r"$R^2$"
)

ax.set_title(
    "Noise robustness: median R² with interquartile range"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="lower left"
)

fig.tight_layout()

f2 = (
    out_dir
    / "Fig2_noise_median_r2_iqr.png"
)

fig.savefig(
    f2,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Figure 3
# Noise vs success rate
# ============================================================

fig, ax = plt.subplots(
    figsize=taller_figsize(7.6, 4.8)
)

ax.plot(
    xn,
    noise_success,
    marker="o",
    linewidth=2,
    label="Discovered equations",
)

ax.plot(
    xn,
    perfect_fit_clean_success,
    linestyle="--",
    marker="s",
    linewidth=2,
    label="Perfect equations (clean)",
)

ax.set_xticks(
    xn,
    noise_labels,
)

ax.set_ylim(
    0,
    110,
)

ax.set_xlabel(
    "Relative noise level"
)

ax.set_ylabel(
    "Success rate (%)"
)

ax.set_title(
    r"Noise robustness: tasks with $R^2 \geq 0.9$"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="lower left"
)

for i, v in enumerate(
    noise_success
):

    ax.annotate(
        f"{v:.1f}%",

        xy=(
            i,
            v,
        ),

        xytext=(
            0,
            -10,
        ),

        textcoords="offset points",

        ha="center",
        va="top",

        fontsize=9,
    )

for i, v in enumerate(
    perfect_fit_clean_success
):

    ax.annotate(
        f"{v:.1f}%",

        xy=(
            i,
            v,
        ),

        xytext=(
            0,
            8,
        ),

        textcoords="offset points",

        ha="center",
        va="bottom",

        fontsize=9,
    )

fig.tight_layout()

f3 = (
    out_dir
    / "Fig3_noise_success_rate.png"
)

fig.savefig(
    f3,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Combined Figure 3 + Figure 2
# Left: success rate; right: median R² + IQR
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=A4_FIGSIZE,
    sharex=True,
)

ax = axes[0]

ax.plot(
    xn,
    noise_success,
    marker="o",
    linewidth=2,
    label="Discovered equations",
)

ax.plot(
    xn,
    perfect_fit_clean_success,
    linestyle="--",
    marker="s",
    linewidth=2,
    label="Perfect equations (clean)",
)

ax.set_ylim(
    0,
    110,
)

ax.set_ylabel(
    "Success rate (%)"
)

ax.set_title(
    r"(a) Tasks with $R^2 \geq 0.9$"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="lower left"
)

for i, v in enumerate(
    noise_success
):

    ax.annotate(
        f"{v:.1f}%",
        xy=(i, v),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
    )

for i, v in enumerate(
    perfect_fit_clean_success
):

    ax.annotate(
        f"{v:.1f}%",
        xy=(i, v),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax = axes[1]

ax.plot(
    xn,
    noise_median,
    marker="o",
    linewidth=2,
    label="Discovered equations",
)

ax.fill_between(
    xn,
    noise_q1,
    noise_q3,
    alpha=0.20,
    color="C0",
    label="Discovered-equation IQR",
)

ax.plot(
    xn,
    perfect_fit_median_r2,
    linestyle="--",
    marker="s",
    linewidth=2,
    color="C1",
    label="Perfect equations",
)

ax.fill_between(
    xn,
    perfect_fit_q1_r2,
    perfect_fit_q3_r2,
    facecolor="none",
    edgecolor="C1",
    linewidth=1.2,
    linestyle=":",
    label="Perfect-equation IQR",
)

ax.set_xticks(
    xn,
    noise_labels,
)

ax.set_xlabel(
    "Relative noise level"
)

ax.set_ylabel(
    r"$R^2$"
)

ax.set_title(
    "(b) Median R² and IQR"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="lower left"
)

axes[0].set_xlabel("Relative noise level")
axes[0].margins(x=0.09)
format_a4_figure(fig)
fig.tight_layout(pad=0.7, w_pad=1.6)

combined_noise_figure = (
    out_dir
    / "Fig2_Fig3_noise_top_bottom.png"
)

fig.savefig(
    combined_noise_figure,
    dpi=A4_DPI,
    bbox_inches=None,
    transparent=False,
    facecolor="white",
)

plt.close(
    fig
)


# ============================================================
# WASS statistics
# ============================================================

wass_series = [
    ("0", wass0),
    ("0.25", wass025),
    ("0.40", wass04),
]

with_parameter_wass_columns = [
    "WASS_0_R2",
    "WASS_025_R2",
    "WASS_04_R2",
]

with_parameter_wass_series = [
    (
        label,
        comparison_numeric_column(column),
    )
    for label, column in zip(
        ("0", "0.25", "0.40"),
        with_parameter_wass_columns,
    )
]

wass_labels = [
    x[0]
    for x in wass_series
]

wass_stats = [
    median_iqr(
        x[1]
    )
    for x in wass_series
]

wass_median = [
    x[0]
    for x in wass_stats
]

wass_q1 = [
    x[1]
    for x in wass_stats
]

wass_q3 = [
    x[2]
    for x in wass_stats
]

wass_success = [
    success_rate(
        x[1],
        threshold=0.9,
    )
    for x in wass_series
]

xw = np.arange(
    len(wass_labels)
)


# ============================================================
# Figure 4
# WASS success rates
# ============================================================

thresholds = [
    0.90,
    0.95,
    0.99,
]

fig, (ax_overview, ax_zoom) = plt.subplots(
    ncols=2,
    figsize=taller_figsize(10.2, 4.8),
    gridspec_kw={
        "width_ratios": [1, 3],
    },
)

threshold_colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

for threshold, color in zip(
    thresholds,
    threshold_colors,
):

    without_parameter_rates = [
        success_rate(
            arr,
            threshold=threshold,
        )
        for _, arr in wass_series
    ]

    with_parameter_rates = [
        success_rate(
            arr,
            threshold=threshold,
        )
        for _, arr in with_parameter_wass_series
    ]

    for ax in [ax_overview, ax_zoom]:

        ax.plot(
            xw,
            without_parameter_rates,
            color=color,
            linestyle="--",
            marker="s",
            markerfacecolor="white",
            markersize=5,
            linewidth=1.8,
            zorder=2,
        )

        ax.plot(
            xw,
            with_parameter_rates,
            color=color,
            marker="o",
            markersize=8,
            linewidth=3,
            zorder=3,
        )

for ax in [ax_overview, ax_zoom]:

    ax.set_xticks(
        xw,
        wass_labels,
    )

    ax.set_xlabel(
        "Wasserstein radius"
    )

    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)

ax_overview.set_ylim(
    0,
    100,
)

ax_overview.set_yticks(
    np.arange(0, 101, 25)
)

ax_overview.set_ylabel(
    "Task success rate (%)"
)

ax_overview.set_title(
    "Full scale"
)

ax_zoom.set_ylim(
    65,
    100,
)

ax_zoom.set_yticks(
    np.arange(65, 101, 5)
)

break_size = 0.008
break_style = dict(
    color="black",
    clip_on=False,
    linewidth=1.2,
    transform=ax_zoom.transAxes,
)
ax_zoom.plot(
    (-break_size, break_size),
    (-break_size, break_size),
    **break_style,
)
ax_zoom.plot(
    (1.0 - break_size, 1.0 + break_size),
    (-break_size, break_size),
    **break_style,
)

ax_zoom.set_ylabel(
    "Task success rate (%)"
)

ax_zoom.set_title(
    "Zoomed scale"
)

fig.suptitle(
    "Generalization under parameter-distribution shift"
)

threshold_handles = [
    Line2D(
        [0],
        [0],
        color=color,
        linewidth=2.5,
        label=rf"$R^2 \geq {threshold:.2f}$",
    )
    for threshold, color in zip(
        thresholds,
        threshold_colors,
    )
]

dataset_handles = [
    Line2D(
        [0],
        [0],
        color="0.35",
        linestyle=(0, (2, 1.5)),
        marker="s",
        markerfacecolor="white",
        linewidth=2.4,
        label="Without parameters",
    ),
    Line2D(
        [0],
        [0],
        color="0.35",
        marker="o",
        linewidth=3,
        label="With parameters",
    ),
]

fig.legend(
    handles=threshold_handles,
    title="Threshold",
    loc="upper center",
    bbox_to_anchor=(0.27, 0.90),
    ncol=3,
)
fig.legend(
    handles=dataset_handles,
    title="Dataset",
    loc="upper center",
    bbox_to_anchor=(0.74, 0.90),
    ncol=2,
    handlelength=4.0,
)

fig.tight_layout(
    rect=(0, 0, 1, 0.84),
)

f4 = (
    out_dir
    / "Fig4_wasserstein_success_rates.png"
)

fig.savefig(
    f4,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Figure 5
# Structural similarity vs numerical fit
#
# Y-axis:
#
#   NMSE = 1 - R²
# ============================================================

fit_strength = np.full(
    n_tasks,
    np.nan,
)

nmse = np.full(
    n_tasks,
    np.nan,
)

for i, v in enumerate(
    shared_r2
):

    if (
        np.isfinite(v)
        and v > INVALID_R2_LIMIT
    ):

        fit_strength[i] = (
            r2_to_strength(v)
        )

        nmse[i] = (
            max(
                1.0 - min(float(v), 1.0),
                1e-15,
            )
        )


mask = (
    np.isfinite(sim)
    & np.isfinite(nmse)
)


fig, ax = plt.subplots(
    figsize=taller_figsize(7.6, 5.1)
)

x_values = sim[mask]
y_values = nmse[mask]

x_distance = np.abs(
    x_values[:, None] - x_values[None, :]
)
log_nmse = np.log10(y_values)
y_distance = np.abs(
    log_nmse[:, None] - log_nmse[None, :]
)
nearby_count = (
    (x_distance <= 5.0)
    & (y_distance <= 1.0)
).sum(axis=1) - 1

draw_order = np.argsort(nearby_count)

points = ax.scatter(
    x_values[draw_order],
    y_values[draw_order],
    c=nearby_count[draw_order],
    cmap="viridis",
    vmin=0,
    vmax=max(1, nearby_count.max()),
    alpha=0.78,
)

colorbar = fig.colorbar(
    points,
    ax=ax,
    fraction=0.025,
    pad=0.02,
    aspect=35,
)
colorbar.locator = MaxNLocator(
    integer=True,
    nbins=6,
)
colorbar.update_ticks()
colorbar.set_label(
    "Number of nearby points"
)

ax.set_xlabel(
    "Structural similarity score"
)

ax.set_yscale(
    "log"
)

ax.yaxis.set_major_formatter(
    LogFormatterSciNotation()
)

ax.set_ylabel(
    "NMSE"
)

ax.set_title(
    "Numerical fit versus structural correctness"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

fig.tight_layout()

f5 = (
    out_dir
    / "Fig5_structure_vs_numerical_fit.png"
)

fig.savefig(
    f5,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Figure 6
# Task-level heatmap
# ============================================================

metrics = [
    (
        "Structure",
        sim / 100.0,
        "structure",
    ),
    (
        "Shared fit",
        shared_r2,
        "r2",
    ),
    (
        "WASS 0",
        wass0,
        "r2",
    ),
    (
        "WASS .25",
        wass025,
        "r2",
    ),
    (
        "WASS .40",
        wass04,
        "r2",
    ),
    (
        "Noise 1%",
        noise001,
        "r2",
    ),
    (
        "Noise 3%",
        noise003,
        "r2",
    ),
    (
        "Noise 5%",
        noise005,
        "r2",
    ),
    (
        "Noise 10%",
        noise01,
        "r2",
    ),
]


H = np.zeros(
    (
        n_tasks,
        len(metrics),
    ),
    dtype=float,
)


for j, (_, arr, kind) in enumerate(
    metrics
):

    arr = np.asarray(
        arr,
        dtype=float,
    )

    if kind == "structure":

        clean = np.where(
            np.isfinite(arr),
            arr,
            0.0,
        )

    else:

        clean = np.where(
            np.isfinite(arr)
            & (
                arr
                > INVALID_R2_LIMIT
            ),
            arr,
            0.0,
        )

    H[:, j] = np.clip(
        clean,
        0,
        1,
    )


fig, ax = plt.subplots(
    figsize=taller_figsize(10.3, 13.0)
)

im = ax.imshow(
    H,
    aspect="auto",
    vmin=0,
    vmax=1,
)

ax.set_xticks(
    np.arange(
        len(metrics)
    ),
    [
        x[0]
        for x in metrics
    ],
    rotation=45,
    ha="right",
)

ax.set_yticks(
    np.arange(
        n_tasks
    ),
    ids,
    fontsize=7,
)

ax.set_title(
    "Task-level structure, generalization, and noise robustness"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)

cbar.set_label(
    r"Normalized score / $R^2$"
)

fig.tight_layout()

f6 = (
    out_dir
    / "Fig6_task_level_heatmap.png"
)

fig.savefig(
    f6,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Figure 7
# Formula complexity vs fit and structural similarity
# ============================================================

without_r2_mask = (
    np.isfinite(complexity)
    & np.isfinite(shared_r2)
    & (shared_r2 > INVALID_R2_LIMIT)
)

with_r2_mask = (
    np.isfinite(with_parameter_complexity)
    & np.isfinite(with_parameter_shared_r2)
    & (with_parameter_shared_r2 > INVALID_R2_LIMIT)
)

without_structure_mask = (
    np.isfinite(complexity)
    & np.isfinite(sim)
)

with_structure_mask = (
    np.isfinite(with_parameter_complexity)
    & np.isfinite(with_parameter_sim)
)

mc = (
    np.isfinite(simp_complexity)
    & np.isfinite(sim)
)

fig = plt.figure(
    figsize=A4_FIGSIZE,
    layout="constrained",
)

outer_grid = fig.add_gridspec(
    nrows=1,
    ncols=2,
    wspace=0.08,
)

r2_grid = outer_grid[0].subgridspec(
    nrows=2,
    ncols=1,
    height_ratios=[4, 1],
    hspace=0.05,
)

ax_r2 = fig.add_subplot(r2_grid[0])
ax_outlier = fig.add_subplot(
    r2_grid[1],
    sharex=ax_r2,
)

for ax in [ax_r2, ax_outlier]:

    ax.scatter(
        complexity[without_r2_mask],
        shared_r2[without_r2_mask],
        alpha=0.78,
        label="Without parameters",
    )

    ax.scatter(
        with_parameter_complexity[with_r2_mask],
        with_parameter_shared_r2[with_r2_mask],
        alpha=0.78,
        marker="^",
        label="With parameters",
    )

    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)

ax_r2.set_ylim(-0.05, 1.05)

minimum_r2 = min(
    np.min(shared_r2[without_r2_mask]),
    np.min(with_parameter_shared_r2[with_r2_mask]),
)

outlier_padding = max(
    abs(minimum_r2) * 0.03,
    1.0,
)

ax_outlier.set_ylim(
    minimum_r2 - outlier_padding,
    minimum_r2 + outlier_padding,
)

ax_r2.spines["bottom"].set_visible(False)
ax_outlier.spines["top"].set_visible(False)
ax_r2.tick_params(
    axis="x",
    which="both",
    bottom=False,
    labelbottom=False,
)
ax_outlier.tick_params(
    axis="x",
    labelbottom=True,
)
ax_outlier.set_xlabel("Formula complexity")
ax_outlier.yaxis.set_major_locator(
    MaxNLocator(2)
)

break_marker = [
    (-1, -0.5),
    (1, 0.5),
]
break_style = dict(
    marker=break_marker,
    markersize=8,
    linestyle="none",
    color="black",
    clip_on=False,
)

ax_r2.plot(
    [0, 1],
    [0, 0],
    transform=ax_r2.transAxes,
    **break_style,
)
ax_outlier.plot(
    [0, 1],
    [1, 1],
    transform=ax_outlier.transAxes,
    **break_style,
)

ax_r2.set_title(
    r"(A) Shared-fit $R^2$"
)

ax_r2.set_ylabel(
    r"$R^2$"
)

ax_r2.legend()

ax = fig.add_subplot(
    outer_grid[1],
    sharex=ax_r2,
)

ax.scatter(
    complexity[without_structure_mask],
    sim[without_structure_mask],
    alpha=0.78,
    label="Without parameters",
)

ax.scatter(
    with_parameter_complexity[with_structure_mask],
    with_parameter_sim[with_structure_mask],
    alpha=0.78,
    marker="^",
    label="With parameters",
)

ax.set_xlabel(
    "Formula complexity"
)

ax.set_ylabel(
    "Structural similarity score"
)

ax.set_title(
    "(B) Structural similarity"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.25,
)

ax.legend()
format_a4_figure(fig)
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04)

f7 = (
    out_dir
    / "Fig7_complexity_vs_structure.png"
)

fig.savefig(
    f7,
    dpi=A4_DPI,
    bbox_inches=None,
    transparent=False,
    facecolor="white",
)

plt.close(
    fig
)


# ============================================================
# Figure 8
# Shared-fit R² distribution
# ============================================================

r2_distribution_labels = [
    "Failed\n($R^2 = -\\infty$)",
    r"$R^2 < 0$",
    "$0 \\leq R^2$\n$< 0.9$",
    "$0.9 \\leq R^2$\n$< 0.99$",
    "$0.99 \\leq R^2$\n$< 0.9999$",
    "$0.9999 \\leq R^2$\n$< 1$",
    r"$R^2 = 1$",
]


def r2_distribution_counts(values):
    all_values = np.asarray(values, dtype=float)
    values = valid_r2(all_values)

    return [
        int(len(all_values) - len(values)),
        int(np.sum(values < 0)),
        int(np.sum((values >= 0) & (values < 0.9))),
        int(np.sum((values >= 0.9) & (values < 0.99))),
        int(np.sum((values >= 0.99) & (values < 0.9999))),
        int(np.sum((values >= 0.9999) & (values < 1.0))),
        int(np.sum(values >= 1.0)),
    ]


without_parameter_r2_counts = r2_distribution_counts(
    shared_r2
)
with_parameter_r2_counts = r2_distribution_counts(
    with_parameter_shared_r2
)

x_r2 = np.arange(
    len(r2_distribution_labels)
)

fig, ax = plt.subplots(
    figsize=taller_figsize(10.2, 4.8)
)

without_parameter_bars = ax.bar(
    x_r2 - bar_width / 2,
    without_parameter_r2_counts,
    width=bar_width,
    label="Without parameters",
)

with_parameter_bars = ax.bar(
    x_r2 + bar_width / 2,
    with_parameter_r2_counts,
    width=bar_width,
    label="With parameters",
)

ax.set_xticks(
    x_r2,
    r2_distribution_labels,
)

ax.set_xlabel(
    r"Shared-fit $R^2$ range"
)

ax.set_ylabel(
    "Number of tasks"
)

ax.set_title(
    r"Distribution of shared-fit $R^2$ values"
)

ax.set_axisbelow(
    True
)

ax.grid(
    axis="y",
    alpha=0.25,
)

add_bar_labels_inside(
    ax,
    without_parameter_bars,
    without_parameter_r2_counts,
    fontsize=10,
)

add_bar_labels_inside(
    ax,
    with_parameter_bars,
    with_parameter_r2_counts,
    fontsize=10,
)

all_r2_counts = (
    without_parameter_r2_counts
    + with_parameter_r2_counts
)

if all_r2_counts:

    ax.set_ylim(
        0,
        max(all_r2_counts) * 1.05,
    )

ax.legend()

fig.tight_layout()

f8 = (
    out_dir
    / "Fig8_shared_fit_r2_distribution.png"
)

fig.savefig(
    f8,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Figures 8 and 1
# Combined distributions
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=A4_LANDSCAPE_FIGSIZE,
)

combined_r2_labels = [
    "Failed", "$R^2 < 0$", "$0 \\leq R^2 < 0.9$",
    "$0.9 \\leq R^2 < 0.99$", "$0.99 \\leq R^2 < 0.9999$",
    "$0.9999 \\leq R^2 < 1$", "$R^2 = 1$",
]
combined_structure_labels = [
    "Failed\n$S < 0$", "Low\n$0$–$44$", "Medium\n$45$–$74$",
    "Relatively High\n$75$–$89$", "High\n$90$–$100$",
]

combined_plots = [
    (
        axes[0],
        x_r2,
        combined_r2_labels,
        without_parameter_r2_counts,
        with_parameter_r2_counts,
        r"Shared-fit $R^2$ interval",
        r"(A) Shared-fit $R^2$ distribution",
        25,
    ),
    (
        axes[1],
        x_levels,
        combined_structure_labels,
        fig1_level_counts,
        fig1_with_parameter_level_counts,
        "Structural similarity level and score interval",
        "(B) Structural similarity distribution",
        0,
    ),
]

for (
    ax,
    x_values,
    tick_labels,
    without_counts,
    with_counts,
    x_label,
    title,
    tick_rotation,
) in combined_plots:
    without_bars = ax.bar(
        x_values - bar_width / 2,
        without_counts,
        width=bar_width,
        label="Without parameters",
    )
    with_bars = ax.bar(
        x_values + bar_width / 2,
        with_counts,
        width=bar_width,
        label="With parameters",
    )
    ax.set_xticks(
        x_values,
        tick_labels,
        rotation=tick_rotation,
    )
    if tick_rotation:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of tasks")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels_inside(
        ax,
        without_bars,
        without_counts,
        fontsize=10,
    )
    add_bar_labels_inside(
        ax,
        with_bars,
        with_counts,
        fontsize=10,
    )
    for text in ax.texts:
        count = int(text.get_text())
        if count < 4:
            text.set_y(count + 0.25)
            text.set_verticalalignment("bottom")
            text.set_color("black")
    all_counts = without_counts + with_counts
    if all_counts:
        ax.set_ylim(0, max(all_counts) * 1.05)
    ax.legend()

format_a4_figure(fig)
axes[0].tick_params(axis="x", labelsize=6.5)
axes[1].tick_params(axis="x", labelsize=7)
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_linespacing(0.9)
fig.tight_layout(pad=0.7, w_pad=1.6)

combined_path = (
    out_dir
    / "Fig1_Fig8_combined.png"
)

fig.savefig(
    combined_path,
    dpi=A4_DPI,
    bbox_inches=None,
    transparent=False,
    facecolor="white",
)

plt.close(
    fig
)


# ============================================================
# Figure 9
# Wasserstein R² dot plot with non-unit means
# ============================================================

R2_PLOT_MAX = 10.0


def plot_wass_r2_dotplot(
    ax,
    series,
    title,
    rng,
):

    values_by_radius = [
        np.maximum(
            valid_r2(values),
            0.0,
        )
        for _, values in series
    ]

    positions_by_radius = [
        np.minimum(
            [r2_to_strength(value) for value in values],
            R2_PLOT_MAX,
        )
        for values in values_by_radius
    ]

    all_positions = np.concatenate(
        positions_by_radius
    )

    y_min = min(
        float(np.min(all_positions)),
        0.0,
    )

    for index, (values, positions) in enumerate(
        zip(
            values_by_radius,
            positions_by_radius,
        )
    ):

        jitter = rng.uniform(
            -0.16,
            0.16,
            size=len(positions),
        )

        ax.scatter(
            index + jitter,
            positions,
            s=18,
            facecolor="black",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            zorder=3,
        )

        mean_position = min(
            r2_to_strength(np.mean(values)),
            R2_PLOT_MAX,
        )

        ax.hlines(
            mean_position,
            index - 0.24,
            index + 0.24,
            color="black",
            linewidth=3.0,
            zorder=4,
        )

    ax.set_xticks(
        np.arange(len(series)),
        [label for label, _ in series],
    )
    ax.set_xlabel(
        "Wasserstein radius"
    )
    ax.set_title(
        title
    )
    ax.set_xlim(
        -0.5,
        len(series) - 0.5,
    )
    ax.set_ylim(
        y_min,
        R2_PLOT_MAX,
    )
    ax.set_axisbelow(
        True
    )
    ax.grid(
        axis="y",
        alpha=0.2,
    )


fig, axes = plt.subplots(
    1,
    2,
    figsize=taller_figsize(11.4, 5.2),
    sharey=True,
    layout="constrained",
)

rng = np.random.default_rng(
    7
)

plot_wass_r2_dotplot(
    axes[0],
    wass_series,
    "Without parameters",
    rng,
)
plot_wass_r2_dotplot(
    axes[1],
    with_parameter_wass_series,
    "With parameters",
    rng,
)

density_tick_values = [
    0.0,
    0.9,
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.999999,
]
density_tick_positions = [
    r2_to_strength(value)
    for value in density_tick_values
] + [R2_PLOT_MAX]
density_tick_labels = [
    "0",
    r"$1-10^{-1}$",
    r"$1-10^{-2}$",
    r"$1-10^{-3}$",
    r"$1-10^{-4}$",
    r"$1-10^{-5}$",
    r"$1-10^{-6}$",
    "1.0",
]

axes[0].set_yticks(
    density_tick_positions,
    density_tick_labels,
)
axes[0].set_ylabel(
    r"$R^2$"
)

axes[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=3.0,
            label="Mean",
        )
    ],
    loc="lower right",
)

fig.suptitle(
    r"$R^2$ distribution across Wasserstein radii"
)
fig.text(
    0.5,
    0.005,
    r"Note: Negative $R^2$ values are clipped to 0.",
    ha="center",
    fontsize=8,
    color="0.35",
)

f9 = (
    out_dir
    / "Fig9_wasserstein_r2_density.png"
)

fig.savefig(
    f9,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Combined 2 x 3 overview
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=taller_figsize(16, 9),
)


# ============================================================
# (a) Structural similarity
# ============================================================

ax = axes[0, 0]

bars = ax.bar(
    ["Failed", *reversed(level_order_en)],
    fig1_level_counts,
)

add_bar_labels_inside(
    ax,
    bars,
    fig1_level_counts,
    fontsize=9,
)

if fig1_level_counts:

    ax.set_ylim(
        0,
        max(fig1_level_counts) * 1.05,
    )

ax.set_title(
    "(a) Structural similarity"
)

ax.set_ylabel(
    "Tasks"
)

ax.tick_params(
    axis="x",
    rotation=15,
)

ax.set_axisbelow(
    True
)

ax.grid(
    axis="y",
    alpha=0.2,
)


# ============================================================
# (b) Noise median R²
# ============================================================

ax = axes[0, 1]

ax.plot(
    xn,
    noise_median,
    marker="o",
)

ax.fill_between(
    xn,
    noise_q1,
    noise_q3,
    alpha=0.2,
)

ax.set_xticks(
    xn,
    noise_labels,
)

ax.set_title(
    "(b) Median R² vs noise"
)

ax.set_xlabel(
    "Noise"
)

ax.set_ylabel(
    r"$R^2$"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.2,
)


# ============================================================
# (c) Noise success rate
# ============================================================

ax = axes[0, 2]

ax.plot(
    xn,
    noise_success,
    marker="o",
)

ax.set_xticks(
    xn,
    noise_labels,
)

ax.set_ylim(
    0,
    105,
)

ax.set_title(
    "(c) Success rate vs noise"
)

ax.set_xlabel(
    "Noise"
)

ax.set_ylabel(
    r"$R^2 \geq 0.9$ (%)"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.2,
)

for i, v in enumerate(
    noise_success
):

    ax.annotate(
        f"{v:.1f}%",

        xy=(
            i,
            v,
        ),

        xytext=(
            0,
            -8,
        ),

        textcoords="offset points",

        ha="center",
        va="top",

        fontsize=8,
    )


# ============================================================
# (d) WASS generalization
# ============================================================

ax = axes[1, 0]

for threshold in thresholds:

    rates = [
        success_rate(
            arr,
            threshold=threshold,
        )
        for _, arr in wass_series
    ]

    ax.plot(
        xw,
        rates,
        marker="o",
        label=(
            rf"$R^2 \geq {threshold:.2f}$"
        ),
    )

ax.set_xticks(
    xw,
    wass_labels,
)

ax.set_ylim(
    0,
    105,
)

ax.set_title(
    "(d) WASS generalization"
)

ax.set_xlabel(
    "Wasserstein radius"
)

ax.set_ylabel(
    "Success rate (%)"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.2,
)

ax.legend(
    fontsize=8,
)


# ============================================================
# (e) Structure vs numerical fit
#
# Y-axis shows actual R² values.
# ============================================================

ax = axes[1, 1]

ax.scatter(
    sim[mask],
    fit_strength[mask],
    alpha=0.75,
)

ax.set_title(
    "(e) Structure vs fit"
)

ax.set_xlabel(
    "Structural similarity"
)

set_r2_transformed_axis(
    ax
)

if np.any(mask):

    ymax = np.nanmax(
        fit_strength[mask]
    )

    ax.set_ylim(
        0,
        min(
            15.5,
            max(
                3.0,
                ymax + 0.5,
            ),
        ),
    )

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.2,
)


# ============================================================
# (f) Complexity vs structure
# ============================================================

ax = axes[1, 2]

ax.scatter(
    simp_complexity[mc],
    sim[mc],
    alpha=0.75,
)

ax.set_title(
    "(f) Complexity vs structure"
)

ax.set_xlabel(
    "Simplified complexity"
)

ax.set_ylabel(
    "Structural similarity"
)

ax.set_axisbelow(
    True
)

ax.grid(
    alpha=0.2,
)


fig.tight_layout()

f_over = (
    out_dir
    / "MDSR_experiment_overview_2x3.png"
)

fig.savefig(
    f_over,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Print statistics
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Generated figures"
)

print(
    "=" * 80
)

for f in [
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    combined_path,
    f9,
    f_over,
]:

    print(
        f
    )


print(
    "\n"
    + "=" * 80
)

print(
    "Key statistics"
)

print(
    "=" * 80
)


print(
    "Tasks:",
    n_tasks,
)


print(
    "Structural levels:",
    dict(
        zip(
            level_order_en,
            level_counts,
        )
    ),
)


print(
    "Similarity >= 90:",
    int(
        np.sum(
            sim >= 90
        )
    ),
    "/",
    n_tasks,
)


print(
    "Similarity >= 80:",
    int(
        np.sum(
            sim >= 80
        )
    ),
    "/",
    n_tasks,
)


print(
    "Median structural similarity:",
    float(
        np.median(
            sim[np.isfinite(sim)]
        )
    ),
)


print(
    "\nNoise median R²:"
)

for label, value in zip(
    noise_labels,
    noise_median,
):

    print(
        f"  {label:>4}: "
        f"{value:.6f}"
    )


print(
    "\nNoise R² >= 0.9 success rate:"
)

for label, value in zip(
    noise_labels,
    noise_success,
):

    print(
        f"  {label:>4}: "
        f"{value:.1f}%"
    )


print(
    "\nWASS median R²:"
)

for label, value in zip(
    wass_labels,
    wass_median,
):

    print(
        f"  {label:>4}: "
        f"{value:.6f}"
    )


print(
    "\nWASS R² >= 0.9 success rate:"
)

for label, value in zip(
    wass_labels,
    wass_success,
):

    print(
        f"  {label:>4}: "
        f"{value:.1f}%"
    )


print(
    "\nDone."
)
