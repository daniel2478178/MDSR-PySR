import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze robustness across training-noise levels.",
    )
    parser.add_argument(
        "--clean-training",
        type=Path,
        default=BASE_DIR / "不带参数.xlsx",
    )
    parser.add_argument(
        "--noise-001-training",
        type=Path,
        default=BASE_DIR / "噪声001统计表(1).xlsx",
    )
    parser.add_argument(
        "--noise-003-training",
        type=Path,
        default=BASE_DIR / "噪声003统计表.xlsx",
    )
    parser.add_argument(
        "--perfect-fit",
        type=Path,
        default=(
            BASE_DIR.parent
            / "equation_verfication"
            / "physicsMDSR_Range_GenerationFormula_noise_metrics.xlsx"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "noise_robustness_figures",
    )
    return parser


args = build_parser().parse_args()
OUT_DIR = args.output_dir.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
from matplotlib.patches import Patch


TRAINING_FILES = {
    0.00: args.clean_training.resolve(),
    0.01: args.noise_001_training.resolve(),
    0.03: args.noise_003_training.resolve(),
}
TEST_COLUMNS = {
    0.00: "noise_0_R2",
    0.01: "noise_001_R2",
    0.03: "noise_003_R2",
    0.05: "noise_005_R2",
    0.10: "noise_01_R2",
}
LABELS = {
    0.00: "Train noise 0",
    0.01: "Train noise 0.01",
    0.03: "Train noise 0.03",
}
COLORS = {
    0.00: "#1F4E79",
    0.01: "#E69F00",
    0.03: "#C44E52",
}
PERFECT_COLOR = "#7A5195"
PERFECT_LABEL = "Perfect equations (clean)"
FAILURE_SENTINEL = -1e50
N_TASKS = 59


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titleweight": "bold",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 130,
        "savefig.dpi": 300,
    }
)


def load_workbook(path):
    frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    frame = frame[frame["ID"].astype(str).str.fullmatch(r"P\d+")].copy()
    frame = frame.set_index("ID", verify_integrity=True)

    for column in TEST_COLUMNS.values():
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.mask(values <= FAILURE_SENTINEL)

    return frame


def save_figure(fig, filename):
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


data = {noise: load_workbook(path) for noise, path in TRAINING_FILES.items()}
perfect_fit = load_workbook(args.perfect_fit.resolve())
perfect_outliers = perfect_fit[list(TEST_COLUMNS.values())].lt(0).all(axis=1)
clean_perfect_fit = perfect_fit.loc[~perfect_outliers]
common_ids = sorted(
    set.intersection(*(set(frame.index) for frame in data.values())),
    key=lambda value: int(value[1:]),
)

records = []
for train_noise, frame in data.items():
    for test_noise, column in TEST_COLUMNS.items():
        values = frame.loc[common_ids, column]
        for equation_id, r2 in values.items():
            records.append(
                {
                    "ID": equation_id,
                    "train_noise": train_noise,
                    "test_noise": test_noise,
                    "r2": r2,
                }
            )

long = pd.DataFrame(records)
summary_rows = []
for (train_noise, test_noise), group in long.groupby(
    ["train_noise", "test_noise"], sort=True
):
    values = group["r2"]
    valid = values.dropna()
    common_id_count = len(values)
    high_accuracy_count = values.ge(0.9).sum()
    positive_r2_count = values.gt(0).sum()
    summary_rows.append(
        {
            "train_noise": train_noise,
            "test_noise": test_noise,
            "common_id_count": common_id_count,
            "valid_count": len(valid),
            "invalid_count": values.isna().sum(),
            "median_r2": valid.median(),
            "q10_r2": valid.quantile(0.10),
            "q25_r2": valid.quantile(0.25),
            "q75_r2": valid.quantile(0.75),
            "high_accuracy_count": high_accuracy_count,
            "high_accuracy_rate": high_accuracy_count / N_TASKS,
            "positive_r2_count": positive_r2_count,
            "positive_r2_rate": positive_r2_count / N_TASKS,
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "robustness_summary.csv", index=False)

perfect_summary_rows = []
for test_noise, column in TEST_COLUMNS.items():
    values = clean_perfect_fit[column]
    valid = values.dropna()
    perfect_summary_rows.append(
        {
            "test_noise": test_noise,
            "median_r2": valid.median(),
            "q10_r2": valid.quantile(0.10),
            "q25_r2": valid.quantile(0.25),
            "q75_r2": valid.quantile(0.75),
            "high_accuracy_rate": values.ge(0.9).mean(),
            "positive_r2_rate": values.gt(0).mean(),
        }
    )
perfect_summary = pd.DataFrame(perfect_summary_rows)

paired_rows = []
for train_noise in (0.01, 0.03):
    for test_noise, column in TEST_COLUMNS.items():
        comparison = pd.DataFrame(
            {
                "clean": data[0.00].loc[common_ids, column],
                "noisy": data[train_noise].loc[common_ids, column],
            }
        )
        valid_pairs = comparison.dropna()
        delta = valid_pairs["noisy"] - valid_pairs["clean"]
        paired_rows.append(
            {
                "train_noise": train_noise,
                "test_noise": test_noise,
                "valid_pair_count": len(valid_pairs),
                "noisy_invalid_count": comparison["noisy"].isna().sum(),
                "median_delta_r2": delta.median(),
                "q10_delta_r2": delta.quantile(0.10),
                "q90_delta_r2": delta.quantile(0.90),
                "fraction_delta_positive": delta.gt(0).mean(),
            }
        )

paired_summary = pd.DataFrame(paired_rows)
paired_summary.to_csv(OUT_DIR / "paired_delta_summary.csv", index=False)


# 1. Robustness profile and threshold-based reliability.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
axes = axes.ravel()
for train_noise in TRAINING_FILES:
    selected = summary[summary["train_noise"] == train_noise]
    x = selected["test_noise"].to_numpy()
    color = COLORS[train_noise]
    axes[0].plot(
        x,
        selected["median_r2"],
        color=color,
        marker="o",
        linewidth=2,
        label=LABELS[train_noise],
    )
    axes[0].fill_between(
        x,
        selected["q25_r2"],
        selected["q75_r2"],
        color=color,
        alpha=0.14,
    )

perfect_x = perfect_summary["test_noise"].to_numpy()
axes[0].plot(
    perfect_x,
    perfect_summary["median_r2"],
    color=PERFECT_COLOR,
    marker="s",
    markerfacecolor="white",
    markeredgewidth=1.8,
    linestyle="--",
    linewidth=2.4,
    label=PERFECT_LABEL,
)
axes[0].set_title("(A) Typical performance")
axes[0].set_ylabel("R² (median; band = IQR)")
axes[0].set_ylim(0.82, 1.01)
axes[0].set_xlabel("Testing noise")
axes[0].set_xticks(list(TEST_COLUMNS))
axes[0].set_xticklabels(["0", "0.01", "0.03", "0.05", "0.10"])
axes[0].grid(axis="y", linestyle=":", alpha=0.35)

column, title, ylabel = (
    "high_accuracy_rate",
    "High-accuracy reliability",
    "Fraction with R² ≥ 0.9",
)
axis = axes[1]
for train_noise in TRAINING_FILES:
    selected = summary[summary["train_noise"] == train_noise]
    rates = selected[column].to_numpy()
    axis.plot(
        selected["test_noise"],
        rates,
        color=COLORS[train_noise],
        marker="o",
        linewidth=2,
        label=LABELS[train_noise],
    )
axis.plot(
    perfect_x,
    perfect_summary[column],
    color=PERFECT_COLOR,
    marker="s",
    markerfacecolor="white",
    markeredgewidth=1.8,
    linestyle="--",
    linewidth=2.4,
    label=PERFECT_LABEL,
)
axis.set_title(f"(B) {title}")
axis.set_ylabel(ylabel)
axis.set_xlabel("Testing noise")
axis.set_xticks(list(TEST_COLUMNS))
axis.set_xticklabels(["0", "0.01", "0.03", "0.05", "0.10"])
axis.set_ylim(0.55, 1.02)
axis.grid(axis="y", linestyle=":", alpha=0.35)

color_handles, color_labels = axes[0].get_legend_handles_labels()

line_handle = Line2D(
    [0],
    [0],
    color="#333333",
    marker="o",
    linewidth=2,
)
axes[0].legend(
    handles=[
        line_handle,
        Patch(facecolor="#777777", alpha=0.20),
    ],
    labels=[
        "Line/markers: median R²",
        "Shaded band: IQR",
    ],
    loc="lower left",
    fontsize=8,
)
axes[1].legend(
    handles=[line_handle],
    labels=["Line/markers: fraction with R² ≥ 0.9"],
    loc="lower left",
    fontsize=8,
)

fig.suptitle(
    "R² robustness and reliability",
    y=0.99,
)
fig.legend(
    color_handles,
    color_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
    ncol=4,
    frameon=False,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
save_figure(fig, "01_r2_robustness_profile.png")

legacy_rate_figure = OUT_DIR / "02_reliability_rates.png"
if legacy_rate_figure.exists():
    legacy_rate_figure.unlink()


# 3. Paired changes reveal small typical gains and occasional large losses.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
rng = np.random.default_rng(7)
positions = np.arange(len(TEST_COLUMNS))
for panel, axis, train_noise in zip(("A", "B"), axes, (0.01, 0.03)):
    for position, (test_noise, column) in zip(positions, TEST_COLUMNS.items()):
        comparison = pd.DataFrame(
            {
                "clean": data[0.00].loc[common_ids, column],
                "noisy": data[train_noise].loc[common_ids, column],
            }
        )
        valid = comparison.dropna()
        delta = (valid["noisy"] - valid["clean"]).to_numpy()
        jitter = rng.uniform(-0.14, 0.14, len(delta))
        axis.scatter(
            position + jitter,
            delta,
            s=18,
            color=COLORS[train_noise],
            alpha=0.55,
            edgecolors="none",
        )
        axis.scatter(
            position,
            np.median(delta),
            s=58,
            marker="D",
            color="#111111",
            zorder=4,
        )
        invalid_count = comparison["noisy"].isna().sum()
        if invalid_count:
            invalid_x = position + np.linspace(
                -0.045,
                0.045,
                invalid_count,
            )
            axis.scatter(
                invalid_x,
                [-250] * invalid_count,
                marker="v",
                s=55,
                color="#111111",
                zorder=5,
            )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_yscale("symlog", linthresh=1e-4)
    axis.set_ylim(-300, 0.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(["0", "0.01", "0.03", "0.05", "0.10"])
    axis.set_xlabel("Testing noise")
    axis.set_title(f"({panel}) Train noise {train_noise:.2f} vs train noise 0")
    axis.grid(axis="y", linestyle=":", alpha=0.3)
axes[0].set_ylabel("Paired ΔR² (noisy training − clean training)")
axes[0].set_yticks([-100, -10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01])
axes[0].set_yticklabels(
    ["−100", "−10", "−1", "−0.1", "−0.01", "−0.001", "0", "0.001", "0.01"]
)
fig.suptitle("Paired effect of adding training noise", y=1.02)
fig.text(
    0.5,
    -0.02,
    "Diamonds mark valid-pair medians; triangles at the floor mark failed evaluations.",
    ha="center",
    color="#555555",
    fontsize=9,
)
fig.tight_layout()
save_figure(fig, "03_paired_training_noise_effect.png")


# 4. Worst-case score across all testing-noise levels for each equation.
fig, axis = plt.subplots(figsize=(7.4, 4.8))
thresholds = np.linspace(0, 1, 401)
for train_noise, frame in data.items():
    matrix = frame.loc[common_ids, list(TEST_COLUMNS.values())]
    worst_case = matrix.fillna(-np.inf).min(axis=1).to_numpy()
    survival = np.array([(worst_case >= threshold).mean() \
                         for threshold in thresholds])
    axis.plot(
        thresholds,
        survival,
        color=COLORS[train_noise],
        linewidth=2.2,
        label=LABELS[train_noise],
    )

axis.axvline(0.9, color="#777777", linestyle="--", linewidth=1)
axis.set_xlim(0, 1)
axis.set_ylim(0, 1.02)
axis.set_xlabel("Required R² at every testing-noise level")
axis.set_ylabel("Fraction of equations meeting the requirement")
axis.set_title("Worst-case robustness across all five testing-noise levels")
axis.grid(linestyle=":", alpha=0.35)
axis.legend(frameon=False, loc="lower left")
fig.text(
    0.5,
    -0.01,
    "Each equation is scored by its lowest R²; invalid evaluations count as failures.",
    ha="center",
    color="#555555",
    fontsize=9,
)
fig.tight_layout()
save_figure(fig, "04_worst_case_robustness.png")


print(f"Common equations: {len(common_ids)}")
print(f"Outputs written to: {OUT_DIR}")
print(summary.to_string(index=False))
