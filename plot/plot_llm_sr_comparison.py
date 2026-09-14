"""Plot LLM-MDSR noise robustness and compare clean-training distributions with MDSR."""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llm_sr_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TRAINING_FILES = {
    0.00: "merged_statistics_没有噪声.xlsx",
    0.01: "merged_statistics_noise_001.xlsx",
    0.03: "merged_statistics_noise_003.xlsx",
}
TEST_COLUMNS = {
    0.00: "noise_0_R2",
    0.01: "noise_001_R2",
    0.03: "noise_003_R2",
    0.05: "noise_005_R2",
    0.10: "noise_01_R2",
}
COLORS = ["#1F4E79", "#D58A00", "#C44E52"]
SUCCESS_COLORS = ["#1F4E79", "#D58A00", "#2E8B57", "#C44E52"]
PERFECT_COLOR = "#7A5195"
TASK_IDS = [f"P{number:02d}" for number in range(1, 60)]
N_TASKS = len(TASK_IDS)
FAILURE_LIMIT = -1e50
R2_LABELS = [
    "Failed", r"$R^2 < 0$", "$0 \\leq R^2 < 0.9$",
    "$0.9 \\leq R^2 < 0.99$", "$0.99 \\leq R^2 < 0.9999$",
    "$0.9999 \\leq R^2 < 1$", r"$R^2 = 1$",
]
STRUCTURE_LABELS = [
    "Failed\n$S < 0$", "Low\n$0$–$44$", "Medium\n$45$–$74$",
    "Relatively High\n$75$–$89$", "High\n$90$–$100$",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-sr-dir", type=Path, default=BASE_DIR / "LLM-SR")
    parser.add_argument("--mdsr-input", type=Path, default=BASE_DIR / "不带参数.xlsx")
    parser.add_argument(
        "--perfect-fit",
        type=Path,
        default=(
            BASE_DIR.parent
            / "equation_verfication"
            / "physicsMDSR_Range_GenerationFormula_noise_metrics.xlsx"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "llm_sr_comparison_figures")
    return parser.parse_args()


def load_data(path, required_columns):
    frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    missing = {"ID", *required_columns} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame = frame.set_index("ID", verify_integrity=True)
    unexpected = set(frame.index) - set(TASK_IDS)
    if unexpected:
        raise ValueError(f"{path.name}: unexpected task IDs {sorted(unexpected)}")
    # Missing benchmark tasks remain visible in distribution counts.
    return frame.reindex(TASK_IDS)


def clean_r2(values):
    values = pd.to_numeric(values, errors="coerce")
    return values.where(np.isfinite(values) & values.gt(FAILURE_LIMIT))


def distribution_counts(values, edges):
    values = pd.to_numeric(values, errors="coerce")
    valid = values[np.isfinite(values)]
    counts = np.histogram(valid, bins=edges)[0]
    if counts.sum() != len(valid):
        raise ValueError("Distribution values fall outside the specified bins.")
    return np.r_[len(values) - len(valid), counts]


def r2_counts(values):
    values = clean_r2(values)
    if values.gt(1).any():
        raise ValueError("R² exceeds 1; check the input metric.")
    return distribution_counts(values, [-np.inf, 0, .9, .99, .9999, 1, np.inf])


def structure_counts(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = values[np.isfinite(values) & (values >= 0)]
    if (valid > 100).any():
        raise ValueError("Structural similarity score exceeds 100.")
    counts = np.histogram(valid, bins=[0, 45, 75, 90, 100.0000001])[0]
    return np.r_[len(values) - len(valid), counts]


def style_axis(ax):
    ax.grid(axis="y", linestyle=":", alpha=.3)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def save_figure(fig, output_dir, name):
    path = output_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def plot_noise_profile(data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    coverage = []
    for (noise, frame), color in zip(data.items(), COLORS):
        values = frame[list(TEST_COLUMNS.values())].apply(clean_r2)
        quantiles = values.quantile([.25, .5, .75])
        ax.plot(list(TEST_COLUMNS), quantiles.loc[.5], "o-", color=color,
                linewidth=2, label=f"Train noise {noise:g}")
        ax.fill_between(list(TEST_COLUMNS), quantiles.loc[.25], quantiles.loc[.75],
                        color=color, alpha=.13)
        counts = values.notna().sum()
        count_label = str(counts.min()) if counts.min() == counts.max() else f"{counts.min()}–{counts.max()}"
        coverage.append(f"{noise:g}: {count_label}/{len(frame)}")
    ax.set_xticks(list(TEST_COLUMNS), ["0", "0.01", "0.03", "0.05", "0.10"])
    ax.set_xlabel("Testing noise")
    ax.set_ylabel("Median R² (band: IQR)")
    ax.set_title("LLM-MDSR performance under testing noise")
    ax.legend()
    style_axis(ax)
    fig.text(.5, .015, "Valid tasks by training noise — " + "; ".join(coverage)
             + ".\nFailed or missing evaluations are excluded from quantiles.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .1, 1, 1))
    save_figure(fig, output_dir, "Fig2_noise_median_r2_iqr.png")


def plot_success_rates(mdsr, llm_sr, perfect_fit, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [("MSSR without parameters", mdsr, "--", "o")]
    series.extend(
        (f"LLM-MDSR, train noise {noise:g}", frame, linestyle, marker)
        for (noise, frame), linestyle, marker in zip(
            llm_sr.items(), ["-", (0, (6, 2)), ":"], ["s", "^", "D"]
        )
    )
    line_widths = [3.0, 7.0, 4.5, 3.5]
    marker_sizes = [7, 11, 9, 6]
    zorders = [4, 2, 3, 5]
    alphas = [1.0, 0.55, 1.0, 1.0]

    for (label, frame, linestyle, marker), color, linewidth, markersize, zorder, alpha in zip(
        series, SUCCESS_COLORS, line_widths, marker_sizes, zorders, alphas
    ):
        values = frame[list(TEST_COLUMNS.values())].apply(clean_r2)
        counts = values.ge(0.9).sum()
        rates = 100 * counts / N_TASKS
        ax.plot(
            list(TEST_COLUMNS), rates, marker=marker, linestyle=linestyle,
            color=color, linewidth=linewidth, markersize=markersize,
            label=label, zorder=zorder, alpha=alpha,
        )
        print(f"{label}: {counts.tolist()} successes out of {N_TASKS} tasks")

    perfect_values = perfect_fit[list(TEST_COLUMNS.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    perfect_outliers = perfect_values.lt(0).all(axis=1)
    clean_perfect_values = perfect_values.loc[~perfect_outliers]
    perfect_counts = clean_perfect_values.ge(0.9).sum()
    perfect_rates = 100 * perfect_counts / len(clean_perfect_values)
    ax.plot(
        list(TEST_COLUMNS), perfect_rates, marker="s", linestyle="--",
        color=PERFECT_COLOR, linewidth=3, markersize=8,
        markerfacecolor="white", markeredgewidth=2,
        label="Perfect equations (clean)", zorder=6,
    )
    excluded_ids = perfect_fit.index[perfect_outliers].tolist()
    print(
        "Perfect equations (clean): "
        f"{perfect_counts.tolist()} successes out of {len(clean_perfect_values)} tasks; "
        f"excluded {excluded_ids}"
    )

    ax.set_xticks(list(TEST_COLUMNS), ["0", "0.01", "0.03", "0.05", "0.10"])
    ax.set_xlabel("Testing noise")
    ax.set_ylabel("Task success rate (%)")
    ax.set_ylim(60, 105)
    ax.set_title(r"Success rate under testing noise ($R^2 \geq 0.9$)")
    ax.legend(loc="lower left")
    style_axis(ax)
    fig.text(
        .5, .015,
        "MSSR/LLM denominator = 59 tasks; missing evaluations count as failures. "
        "Perfect-equation curve excludes P13 and P20.",
        ha="center", va="bottom", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, output_dir, "Fig3_success_rate_comparison.png")


def plot_bars(ax, counts, offset, color, label):
    bars = ax.bar(np.arange(len(counts)) + offset, counts, width=.36,
                  color=color, label=label)
    for bar, count in zip(bars, counts):
        x = bar.get_x() + bar.get_width()/2
        if 0 < count < 4:
            ax.annotate(str(count), (x, count), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", color="#333333", weight="bold", fontsize=10)
        elif count:
            ax.text(x, count/2, str(count),
                    ha="center", va="center", color="white", weight="bold", fontsize=11)


def plot_comparison(mdsr, llm_sr, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = [
        (r2_counts(mdsr["shared_fit_r2_0_7"]), r2_counts(llm_sr["noise_0_R2"]),
         R2_LABELS, "Shared-fit $R^2$ interval", "(A) Distribution of R² values", 25),
        (structure_counts(mdsr["structure_similarity_score"]), structure_counts(llm_sr["SimilarityScore"]),
         STRUCTURE_LABELS, "Structural similarity level and score interval",
         "(B) Distribution of structural similarity levels", 0),
    ]
    for ax, (mdsr_counts, llm_counts, labels, xlabel, title, rotation) in zip(
        axes, panels
    ):
        plot_bars(ax, mdsr_counts, -.18, COLORS[0], "MSSR without parameters")
        plot_bars(ax, llm_counts, .18, COLORS[1], "LLM-MDSR, train noise 0")
        ax.set_xticks(
            np.arange(len(labels)),
            labels,
            rotation=rotation,
            fontsize=6.5 if rotation else 7,
        )
        if rotation:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
                label.set_rotation_mode("anchor")
        for label in ax.get_xticklabels():
            label.set_linespacing(0.9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of tasks")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(1, max(mdsr_counts), max(llm_counts)) * 1.12)
        ax.legend(loc="upper left")
        style_axis(ax)
        print(f"{xlabel}: MDSR {mdsr_counts.tolist()}; LLM-MDSR {llm_counts.tolist()}")
    fig.text(.5, .018,
             "59 tasks per method. R²: MDSR shared fit; LLM-MDSR clean test, training noise 0.\n"
             "Structure uses common score bins; a failed fit can still have a valid structure score.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .085, 1, 1), h_pad=2.8)
    save_figure(fig, output_dir, "Fig1_Fig8_mdsr_llm_sr_comparison.png")


def main():
    args = parse_args()
    data = {
        noise: load_data(args.llm_sr_dir / filename, [*TEST_COLUMNS.values(), "SimilarityScore"])
        for noise, filename in TRAINING_FILES.items()
    }
    mdsr = load_data(
        args.mdsr_input,
        ["shared_fit_r2_0_7", "structure_similarity_score", *TEST_COLUMNS.values()],
    )
    perfect_fit = load_data(args.perfect_fit, list(TEST_COLUMNS.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_noise_profile(data, args.output_dir)
    plot_comparison(mdsr, data[0.0], args.output_dir)
    plot_success_rates(mdsr, data, perfect_fit, args.output_dir)


if __name__ == "__main__":
    main()
