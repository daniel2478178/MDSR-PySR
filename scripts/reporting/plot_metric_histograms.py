import argparse
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mssr-pysr-mpl"))

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parents[1] / "physics_formula_structural_similarity_evaluation_withooutparam.xlsx"
SHEET = "公式评价"
ID_COLUMN = "ID"
SIMILARITY_COLUMN = "structure_similarity_score"
R2_COLUMN = "shared_fit_r2_0_7"

BLUE = "#2A6F97"
ORANGE = "#E76F51"
TEXT = "#243447"
MUTED = "#66788A"
GRID = "#D8E0E8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot structural-similarity and shared-fit metric distributions")
    parser.add_argument(
        "input", nargs="?", type=Path, default=INPUT,
        help=f"Evaluation workbook (default: {INPUT.name})",
    )
    parser.add_argument("--output-dir", type=Path, help="Destination directory; defaults beside input")
    parser.add_argument("--sheet", default=SHEET)
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--similarity-column", default=SIMILARITY_COLUMN)
    parser.add_argument("--r2-column", default=R2_COLUMN)
    return parser


def configure(args) -> Path:
    global INPUT, SHEET, ID_COLUMN, SIMILARITY_COLUMN, R2_COLUMN
    INPUT = args.input.expanduser().resolve()
    SHEET = args.sheet
    ID_COLUMN = args.id_column
    SIMILARITY_COLUMN = args.similarity_column
    R2_COLUMN = args.r2_column
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else INPUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric values found in {name!r}.")
    return values


def style_axis(axis) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(1.6)
    axis.tick_params(
        which="both", direction="in", top=True, right=True,
        width=1.3, colors=TEXT
    )
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)


def plot_histograms(frame: pd.DataFrame, unique_by: str, output: Path) -> None:
    columns = [SIMILARITY_COLUMN, R2_COLUMN]
    negative_count = int((frame[R2_COLUMN] < 0).sum())
    frame = frame[frame[R2_COLUMN] >= 0].copy()
    plot_total = len(frame)
    print(f"Excluded {negative_count} rows with R² < 0; plotting {plot_total} rows")
    similarity = numeric_column(frame, columns[0])
    shared_r2 = numeric_column(frame, columns[1])
    print(frame.iloc[frame[columns[1]].argmin()])

    paired = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    
    pearson_r, pearson_p = pearsonr(paired[columns[0]], paired[columns[1]])
    spearman_rho, spearman_p = spearmanr(paired[columns[0]], paired[columns[1]])
    print("\nAssociation between StructureSimilarity and shared_fit_r2_0_7")
    print(f"Rows analyzed: {len(paired)}")
    print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.3e})")
    print(f"Spearman rho: {spearman_rho:.4f} (p={spearman_p:.3e})")
    thresh_r2_low = paired[columns[1]].mean()
    thresh_r2_high = 0.9#thresh_r2_low
    thresh_similarity_low = paired[columns[0]].mean()
    thresh_similarity_high = 80#thresh_similarity_low

    mask_low_r2 = paired[columns[1]] < thresh_r2_low
    mask_high_r2 = paired[columns[1]] >= thresh_r2_high
    mask_low_similarity = paired[columns[0]] < thresh_similarity_low
    mask_high_similarity = paired[columns[0]] >= thresh_similarity_high

    mask_low_r2_high_similarity = paired[mask_low_r2 & mask_high_similarity]
    mask_high_r2_low_similarity = paired[mask_high_r2 & mask_low_similarity]
    print(f"\nLow R² & High similarity: {len(mask_low_r2_high_similarity)}")
    print(f"High R² & Low similarity: {len(mask_high_r2_low_similarity)}")

    print(f"percentage of High R² & Low similarity in High R²: {len(mask_high_r2_low_similarity) \
                                                              / len(mask_high_r2) * 100:.2f}%")
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1], width_ratios=[0.65, 0.35]
    )
    similarity_axis = fig.add_subplot(grid[0, :])
    zoom_axis = fig.add_subplot(grid[1, 0])
    r2_axis = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(
        top=0.90, bottom=0.10, left=0.08, right=0.97, hspace=0.52, wspace=0.32
    )
    parameter_label = (
        "with parameters" if INPUT.stem.endswith("_with_param")
        else "without parameters" if INPUT.stem.endswith("_withooutparam")
        else INPUT.stem
    )
    method_label = r"max $r^2$" if unique_by == R2_COLUMN else "max structural similarity"
    title = f"Result Summary ({parameter_label}; method = {method_label})"
    fig.suptitle(title, fontsize=18, fontweight="bold", color=TEXT)

    similarity_counts, _, similarity_bars = similarity_axis.hist(
        similarity, bins=np.linspace(0, 100, 21), color=BLUE,
        edgecolor="white", linewidth=1.1
    )
    similarity_axis.set(
        title="A   Structural similarity",
        xlabel="StructureSimilarity score (0–100)",
        ylabel="Formula count",
        xlim=(0, 100),
    )
    similarity_axis.set_xticks(np.arange(0, 101, 10))
    similarity_axis.set_ylim(0, max(similarity_counts) * 1.1)
    similarity_axis.axvline(
        similarity.median(), color=TEXT, linestyle="--", linewidth=1.4
    )
    similarity_axis.text(
        similarity.median() - 1.5, similarity_axis.get_ylim()[1] * 0.82,
        f"Median = {similarity.median():.1f}", ha="right", color=TEXT
    )
    last_bar = similarity_bars[-1]
    similarity_axis.annotate(
        f"{int(similarity_counts[-1])}/{plot_total}",
        xy=(last_bar.get_x() + last_bar.get_width() / 2, similarity_counts[-1]),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", color=BLUE, fontweight="bold"
    )
    style_axis(similarity_axis)

    main_r2 = shared_r2
    r2_counts, _, r2_bars = r2_axis.hist(
        main_r2, bins=np.linspace(0, 1, 21), color=ORANGE,
        edgecolor="white", linewidth=1.1
    )
    r2_axis.set(
        title="C   Shared-fit R²",
        xlabel="shared_fit_r2_0_7",
        ylabel="Formula count (log scale)",
        xlim=(0, 1),
    )
    r2_axis.set_yscale("log")
    r2_axis.set_ylim(0.8, max(r2_counts) * 2.2)
    r2_axis.yaxis.set_major_locator(LogLocator(base=10))
    r2_axis.yaxis.set_major_formatter(ScalarFormatter())
    style_axis(r2_axis)
    r2_axis.text(
        1, 1.10,
        f"{len(main_r2)}/{plot_total} plotted"
        f"  •  median = {main_r2.median():.6f}",
        transform=r2_axis.transAxes, ha="right", color=MUTED
    )
    last_r2_bar = r2_bars[-1]
    r2_axis.annotate(
        f"{int(r2_counts[-1])}/{plot_total}",
        xy=(
            last_r2_bar.get_x() + last_r2_bar.get_width() / 2,
            r2_counts[-1],
        ),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", color=ORANGE, fontweight="bold"
    )

    inset_r2 = shared_r2[(shared_r2 > 0.9) & (shared_r2 <= 1)]
    transformed_r2 = np.log(inset_r2 - 0.9)
    zoom_axis.hist(
        transformed_r2, bins=10, color=ORANGE, edgecolor="white"
    )
    zoom_axis.set(
        title=f"D   R² detail\n0.9 < R² ≤ 1 "
        f"({len(inset_r2)}/{plot_total} experiments)",
        xlabel="ln(R² − 0.9)", ylabel="Count",
    )
    style_axis(zoom_axis)

    fig.text(
        0.5, 0.035,
        f"Source: {INPUT.name}  •  Sheet: Formula evaluation  •  "
        f"n = {plot_total} after excluding R² < 0",
        ha="center", color=MUTED, fontsize=9
    )
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def main(argv=None) -> None:
    output_dir = configure(build_parser().parse_args(argv))
    columns = [SIMILARITY_COLUMN, R2_COLUMN]
    frame = pd.read_excel(
        INPUT, sheet_name=SHEET, usecols=[ID_COLUMN, *columns], engine="openpyxl"
    )
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")

    source_label = (
        "with_param" if INPUT.stem.endswith("_with_param")
        else "without_param" if INPUT.stem.endswith("_withooutparam")
        else INPUT.stem
    )
    outputs = {
        R2_COLUMN: output_dir / f"metric_histograms_{source_label}_unique_by_max_r2.png",
        SIMILARITY_COLUMN: (
            output_dir / f"metric_histograms_{source_label}_unique_by_max_similarity.png"
        ),
    }
    for unique_by, output in outputs.items():
        valid = frame.dropna(subset=[unique_by])
        print(f"\nUnique by {unique_by}: {len(valid)} valid rows")
        unique_frame = valid.loc[valid.groupby(ID_COLUMN)[unique_by].idxmax()]

        plot_histograms(unique_frame, unique_by, output)


if __name__ == "__main__":
    main()
