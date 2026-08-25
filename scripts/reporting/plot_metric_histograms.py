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
INPUT = ROOT / "physics_formula_structural_similarity_evaluation_with_param.xlsx"
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
    parser.add_argument("input", type=Path, help="Evaluation workbook")
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
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[0.45, 1])
    similarity_axis = fig.add_subplot(grid[0, :])
    zoom_axis = fig.add_subplot(grid[1, 0])
    negative_r2 = shared_r2[shared_r2 < -1]
    if negative_r2.empty:
        negative_axis = None
        r2_axis = fig.add_subplot(grid[1, 1])
    else:
        r2_grid = grid[1, 1].subgridspec(
            1, 2, width_ratios=[0.18, 0.82], wspace=0.05
        )
        negative_axis = fig.add_subplot(r2_grid[0, 0])
        r2_axis = fig.add_subplot(r2_grid[0, 1], sharey=negative_axis)
    fig.subplots_adjust(
        top=0.90, bottom=0.10, left=0.08, right=0.97, hspace=0.52, wspace=0.32
    )
    fig.suptitle(
        f"Metric distributions: one formula per ID (maximum {unique_by})",
        fontsize=18, fontweight="bold", color=TEXT
    )

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
        f"{int(similarity_counts[-1])}/{len(frame)}",
        xy=(last_bar.get_x() + last_bar.get_width() / 2, similarity_counts[-1]),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", color=BLUE, fontweight="bold"
    )
    style_axis(similarity_axis)

    main_r2 = shared_r2 if negative_axis is None else shared_r2[shared_r2 >= -1]
    r2_counts, _, r2_bars = r2_axis.hist(
        main_r2, bins=20, color=ORANGE,
        edgecolor="white", linewidth=1.1
    )
    r2_axis.set(
        title="B   Shared-fit R²",
        xlabel="shared_fit_r2_0_7",
    )
    if negative_axis is None:
        r2_axis.set_ylabel("Formula count (log scale)")
        r2_axes = [r2_axis]
    else:
        negative_axis.hist(
            negative_r2, bins="auto", color=ORANGE,
            edgecolor="white", linewidth=1.1
        )
        negative_axis.set_ylabel("Formula count (log scale)")
        r2_axes = [negative_axis, r2_axis]

    for axis in r2_axes:
        axis.set_yscale("log")
        axis.set_ylim(0.8, max(r2_counts) * 2.2)
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.yaxis.set_major_formatter(ScalarFormatter())
        style_axis(axis)

    if negative_axis is not None:
        negative_axis.spines["right"].set_visible(False)
        r2_axis.spines["left"].set_visible(False)
        negative_axis.tick_params(which="both", right=False)
        r2_axis.tick_params(which="both", left=False, labelleft=False)
        break_size = 0.018
        break_style = dict(color=MUTED, clip_on=False, linewidth=1.5)
        negative_axis.plot(
            (1 - break_size, 1 + break_size), (-break_size, break_size),
            transform=negative_axis.transAxes, **break_style
        )
        negative_axis.plot(
            (1 - break_size, 1 + break_size),
            (1 - break_size, 1 + break_size),
            transform=negative_axis.transAxes, **break_style
        )
        r2_axis.plot(
            (-break_size, break_size), (-break_size, break_size),
            transform=r2_axis.transAxes, **break_style
        )
        r2_axis.plot(
            (-break_size, break_size), (1 - break_size, 1 + break_size),
            transform=r2_axis.transAxes, **break_style
        )

    r2_axis.axvline(0, color=TEXT, linewidth=1.2)
    r2_axis.text(
        1, 1.10,
        f"{(shared_r2 < 0).sum()}/{len(frame)} below 0"
        f"  •  median = {shared_r2.median():.6f}",
        transform=r2_axis.transAxes, ha="right", color=MUTED
    )
    last_r2_bar = r2_bars[-1]
    r2_axis.annotate(
        f"{int(r2_counts[-1])}/{len(frame)}",
        xy=(
            last_r2_bar.get_x() + last_r2_bar.get_width() / 2,
            r2_counts[-1],
        ),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", color=ORANGE, fontweight="bold"
    )

    inset_r2 = shared_r2[(shared_r2 >= 0.8) & (shared_r2 <= 1)]
    zoom_axis.hist(
        inset_r2, bins=np.linspace(0.8, 1, 11), color=ORANGE, edgecolor="white"
    )
    zoom_axis.set(
        title=f"C   R² detail\n0.8 ≤ R² ≤ 1 "
        f"({len(inset_r2)}/{len(frame)} experiments)",
        xlabel="R²", ylabel="Count"
    )
    zoom_axis.set_xticks([0.8, 0.9, 1.0])
    style_axis(zoom_axis)

    fig.text(
        0.5, 0.035,
        f"Source: {INPUT.name}  •  Sheet: Formula evaluation  •  n = {len(frame)}",
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

    outputs = {
        R2_COLUMN: output_dir / "metric_histograms_unique_by_max_r2.png",
        SIMILARITY_COLUMN: output_dir / "metric_histograms_unique_by_max_similarity.png",
    }
    for unique_by, output in outputs.items():
        valid = frame.dropna(subset=[unique_by])
        print(f"\nUnique by {unique_by}: {len(valid)} valid rows")
        unique_frame = valid.loc[valid.groupby(ID_COLUMN)[unique_by].idxmax()]

        plot_histograms(unique_frame, unique_by, output)


if __name__ == "__main__":
    main()
