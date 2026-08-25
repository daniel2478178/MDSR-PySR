
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import sympy as sp


# ============================================================
# CONFIG
# ============================================================

# Root containing one subfolder per benchmark ID.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "physicsMDSR_Range_CSV"

# Master benchmark metadata file.
# This is an Excel workbook, not a CSV text file.
MASTER_FILE = PROJECT_ROOT / "physicsMDSR_Range_with_simpFormula.xlsx"
#MASTER_FILE = Path("physics_datset_MDSR.xlsx")

# Excel sheet to read:
#   0 -> first worksheet
#   "Sheet1" -> named worksheet
MASTER_SHEET = "Sampling design"

# File to find recursively in every benchmark folder.
SUMMARY_NAME = "pysr_warm_0_7_summary.csv"

# ------------------------------------------------------------
# Number of best GROUP-LEADER formulas selected for EACH summary file / ID.
# Change only this value if you want top 5 / top 20 / etc.
# ------------------------------------------------------------
TOP_N = 10

# Output file.
OUTPUT_CSV = ROOT / f"physics_datset_MDSR_top{TOP_N}_shared_fit_formulas.csv"

# Columns copied from each pysr_warm_0_7_summary.csv.
SUMMARY_COLUMNS = [
    "para_eq",
    "orgpara_list",
    "para_list",
    "shared_fit_mse_0_7",
    "shared_fit_mse_hmean_0_7",
    "shared_fit_r2_0_7",
    "complexity",
    "loss",
    "score",
    "dataset",
    "fromdataset",
    "ID",
]

# ``redundant == 0`` identifies the representative/leader of one equivalent
# formula group. Group-member rows are never merged into the final output.
GROUP_LEADER_COLUMN = "redundant"

# Optional metadata columns used to restore PySR-safe feature names.
FORMULA_METADATA_COLUMNS = [
    "IndependentVars",
    "ParameterNames",
    "FixedConstantValues",
    "feature_name_map",
]

# Columns copied from physicsMDSR_Range.xlsx.
MASTER_COLUMNS = [
    "ID",
    "ModelName",
    "OriginalFormula",
    "GenerationFormula",
    "Target",
    "IndependentVars",
    "ParameterRange",
    "FixedConstantValues",
    "simpFormula",
]

# Sorting column.
SORT_COL = "shared_fit_mse_0_7"

# True:
#   rows whose shared_fit_mse_0_7 is blank / NaN / inf are ignored.
# False:
#   they are kept and sorted at the end.
DROP_INVALID_MSE = True

# Optional:
# The previous fitting script used 1e100 to mark a failed common fit.
# If True, those failed candidates are excluded before taking TOP_N.
# If an ID then has fewer than TOP_N valid candidates, fewer rows are output.
EXCLUDE_FAILED_FITS = True
FAILED_MSE_THRESHOLD = 1.0e99


def build_parser():
    parser = argparse.ArgumentParser(description="Merge the best shared PySR formulas with benchmark metadata")
    parser.add_argument("data_root", type=Path, help="Root containing one PySR summary per benchmark ID")
    parser.add_argument("master_workbook", type=Path, help="Workbook containing benchmark metadata and simpFormula")
    parser.add_argument("output", type=Path, help="Destination CSV")
    parser.add_argument("--sheet", default=MASTER_SHEET)
    parser.add_argument("--summary-name", default=SUMMARY_NAME)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--include-failed", action="store_true")
    return parser


def configure(args):
    global ROOT, MASTER_FILE, OUTPUT_CSV, MASTER_SHEET, SUMMARY_NAME, TOP_N
    global EXCLUDE_FAILED_FITS
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    ROOT = args.data_root.expanduser().resolve()
    MASTER_FILE = args.master_workbook.expanduser().resolve()
    OUTPUT_CSV = args.output.expanduser().resolve()
    MASTER_SHEET = args.sheet
    SUMMARY_NAME = args.summary_name
    TOP_N = args.top_n
    EXCLUDE_FAILED_FITS = not args.include_failed


# ============================================================
# ID helpers
# ============================================================

def normalize_id(value):
    """
    Normalize IDs for reliable joining.

    Examples:
        3       -> "3"
        3.0     -> "3"
        "3"     -> "3"
        " 003 " -> "003"

    Non-numeric IDs such as P01 remain strings.

    Note:
    Leading zeros in string IDs are preserved.
    """
    if pd.isna(value):
        return ""

    # Preserve an explicitly textual ID.
    if isinstance(value, str):
        return value.strip()

    # Normalize integer-like numeric values.
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return str(value).strip()


def folder_id_from_summary(summary_file: Path) -> str:
    """
    Fallback ID = immediate parent folder name.
    """
    return summary_file.parent.name.strip()


def alpha_index(index: int) -> str:
    """Return the same alphabetic suffix used by the PySR runner."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    result = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = letters[remainder] + result
    return result


def parse_json_value(value, default):
    if pd.isna(value):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def restore_para_eq(row: pd.Series) -> str:
    """Restore xv/pv/cv names in a formula using the runner's mapping."""
    formula = row.get("para_eq", "")
    if pd.isna(formula):
        return formula

    feature_map = parse_json_value(row.get("feature_name_map"), {})
    if not isinstance(feature_map, dict):
        feature_map = {}

    independent_vars = parse_json_value(row.get("IndependentVars"), [])
    parameter_names = parse_json_value(row.get("ParameterNames"), [])
    fixed_constants = parse_json_value(row.get("FixedConstantValues"), {})

    if not feature_map:
        feature_map = {
            f"xv_{alpha_index(i)}": name
            for i, name in enumerate(independent_vars)
        }
        feature_map.update({
            f"pv_{alpha_index(i)}": name
            for i, name in enumerate(parameter_names)
        })
        feature_map.update({
            f"cv_{alpha_index(i)}": name
            for i, name in enumerate(fixed_constants.keys())
        })

    try:
        expression = sp.sympify(str(formula))
        substitutions = {
            sp.Symbol(str(safe)): sp.Symbol(str(original))
            for safe, original in feature_map.items()
        }
        return str(expression.xreplace(substitutions))
    except (TypeError, ValueError, sp.SympifyError):
        return str(formula)


# ============================================================
# Master metadata
# ============================================================

def load_master(master_file: Path) -> pd.DataFrame:
    """
    Load the benchmark metadata table.

    Supported:
        .xlsx / .xls / .xlsm -> pandas.read_excel()
        .csv                  -> pandas.read_csv()

    physicsMDSR_Range.xlsx is the expected/default input.
    """
    if not master_file.is_file():
        raise FileNotFoundError(
            f"Master file not found: {master_file.resolve()}"
        )

    suffix = master_file.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            df = pd.read_excel(
                master_file,
                sheet_name=MASTER_SHEET,
            )
        except ImportError as e:
            raise ImportError(
                "Reading .xlsx requires an Excel engine. "
                "Install openpyxl with:\n"
                "    pip install openpyxl\n"
                f"Original error: {e}"
            ) from e

    elif suffix == ".csv":
        # Kept only for optional backward compatibility.
        # utf-8-sig works for normal UTF-8 CSVs exported by Excel/WPS.
        df = pd.read_csv(
            master_file,
            encoding="utf-8-sig",
        )

    else:
        raise ValueError(
            f"Unsupported master file type: {suffix!r}. "
            "Use .xlsx, .xls, .xlsm, or .csv."
        )

    # Remove accidental whitespace around Excel header names.
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # physics_datset_MDSR names this field TransformedFormula; keep the
    # existing output contract used by the merge script.
    if "GenerationFormula" not in df.columns and "TransformedFormula" in df.columns:
        df = df.rename(
            columns={"TransformedFormula": "GenerationFormula"}
        )

    missing = [
        c for c in MASTER_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{master_file} is missing required columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )

    # Keep only requested columns.
    df = df[MASTER_COLUMNS].copy()

    # Internal join key.
    df["_merge_ID"] = df["ID"].map(normalize_id)

    # Detect ambiguous duplicate IDs in master.
    duplicate_mask = df["_merge_ID"].duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_ids = (
            df.loc[duplicate_mask, "_merge_ID"]
            .drop_duplicates()
            .tolist()
        )

        print(
            "[WARNING] Duplicate IDs found in physicsMDSR_Range.xlsx:",
            duplicate_ids,
        )
        print(
            "Each matching master row will produce additional merged rows."
        )

    return df


# ============================================================
# One summary file
# ============================================================

def select_top_n_from_summary(
    summary_file: Path,
    top_n: int,
) -> pd.DataFrame:
    """
    Read one pysr_warm_0_7_summary.csv, keep only redundancy-group leaders,
    and select the top N leaders with the smallest shared_fit_mse_0_7.
    """
    df = pd.read_csv(summary_file)

    required_without_id = [
        c for c in SUMMARY_COLUMNS
        if c != "ID"
    ]
    required_without_id.append(GROUP_LEADER_COLUMN)

    missing = [
        c for c in required_without_id
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{summary_file} is missing required columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )

    # If ID is absent, infer it from folder name.
    if "ID" not in df.columns:
        inferred_id = folder_id_from_summary(summary_file)
        df["ID"] = inferred_id
        print(
            f"[INFO] {summary_file}: no ID column; "
            f"using folder name ID={inferred_id!r}"
        )

    # Filter group leaders BEFORE validating MSE and taking TOP_N.  This is
    # important because a redundant member can have a smaller MSE than its
    # leader, but only the leader represents that equivalent-formula group in
    # the merged output.
    redundant_numeric = pd.to_numeric(
        df[GROUP_LEADER_COLUMN],
        errors="coerce",
    )
    redundant_array = redundant_numeric.to_numpy(dtype=float)
    invalid_redundant = (
        ~np.isfinite(redundant_array)
        | (redundant_array < 0)
        | (redundant_array != np.floor(redundant_array))
    )
    if invalid_redundant.any():
        invalid_rows = (np.flatnonzero(invalid_redundant) + 2).tolist()
        raise ValueError(
            f"{summary_file}: {GROUP_LEADER_COLUMN!r} must contain "
            "non-negative integers; invalid CSV row(s): "
            f"{invalid_rows[:20]}"
        )

    df[GROUP_LEADER_COLUMN] = redundant_numeric.astype(np.int64)
    df = df.loc[df[GROUP_LEADER_COLUMN] == 0].copy()
    leader_candidate_count = len(df)

    # Convert metric to numeric so sorting is truly numerical.
    df[SORT_COL] = pd.to_numeric(
        df[SORT_COL],
        errors="coerce",
    )

    if DROP_INVALID_MSE:
        valid = np.isfinite(
            df[SORT_COL].to_numpy(dtype=float)
        )
        df = df.loc[valid].copy()

    if EXCLUDE_FAILED_FITS:
        df = df.loc[
            df[SORT_COL] < FAILED_MSE_THRESHOLD
        ].copy()

    # Existing summaries may contain PySR-safe names such as xv_a and pv_a.
    # Restore them before selecting and exporting candidate formulas.
    df["para_eq"] = df.apply(restore_para_eq, axis=1)

    # Stable sort means ties preserve their original CSV order.
    df = df.sort_values(
        by=SORT_COL,
        ascending=True,
        kind="mergesort",
        na_position="last",
    )

    top = df.head(top_n).copy()

    # Keep only the requested summary fields.
    top = top[SUMMARY_COLUMNS].copy()

    # Normalize ID for joining.
    top["_merge_ID"] = top["ID"].map(normalize_id)

    # Helpful audit fields.
    top["TopRank"] = np.arange(
        1,
        len(top) + 1,
        dtype=int,
    )
    top["SourceFolder"] = str(summary_file.parent)
    top.attrs["leader_candidate_count"] = leader_candidate_count

    return top


# ============================================================
# Collect all summaries
# ============================================================

def collect_top_formulas(
    root: Path,
    top_n: int,
) -> tuple[pd.DataFrame, int, int]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"ROOT directory not found: {root.resolve()}"
        )

    summary_files = sorted(
        root.rglob(SUMMARY_NAME)
    )

    print(
        f"Found {len(summary_files)} {SUMMARY_NAME} files."
    )

    if not summary_files:
        return pd.DataFrame(), 0, 0

    pieces = []
    ok_count = 0
    fail_count = 0

    for i, summary_file in enumerate(
        summary_files,
        start=1,
    ):
        try:
            top = select_top_n_from_summary(
                summary_file,
                top_n,
            )

            pieces.append(top)
            ok_count += 1

            print(
                f"[{i:>4}/{len(summary_files)}] "
                f"ID={folder_id_from_summary(summary_file)} "
                f"leaders={top.attrs.get('leader_candidate_count', 0)} "
                f"selected={len(top)}"
            )

        except Exception as e:
            fail_count += 1
            print(
                f"[ERROR] {summary_file}: {e}",
                file=sys.stderr,
            )

    if not pieces:
        return pd.DataFrame(), ok_count, fail_count

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        ),
        ok_count,
        fail_count,
    )


# ============================================================
# Merge
# ============================================================

def merge_with_master(
    candidates: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    One master ID can match many candidate rows:
        physicsMDSR_Range.xlsx: 1 row per ID
        top group leaders:     up to TOP_N rows per summary file / ID
    """
    if candidates.empty:
        return candidates

    # Rename the summary's ID before merging so the final result has one
    # clean ID column originating from the master when available.
    candidates = candidates.rename(
        columns={"ID": "SummaryID"}
    )

    merged = master.merge(
        candidates,
        on="_merge_ID",
        how="right",
        validate="one_to_many",
        suffixes=("", "_summary"),
    )

    # If no master row matched an ID, keep the summary ID rather than losing it.
    merged["ID"] = merged["ID"].where(
        merged["ID"].notna(),
        merged["SummaryID"],
    )

    unmatched = merged["ModelName"].isna()

    if unmatched.any():
        bad_ids = (
            merged.loc[unmatched, "SummaryID"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        print(
            "[WARNING] IDs present in summary files but not found "
            f"in physicsMDSR_Range.xlsx: {bad_ids}"
        )

    # Final requested layout.
    final_columns = [
        # Link key + rank
        "ID",
        "TopRank",

        # Master metadata
        "ModelName",
        "OriginalFormula",
        "GenerationFormula",
        "Target",
        "IndependentVars",
        "ParameterRange",
        "FixedConstantValues",
        "simpFormula",

        # Selected PySR formula information
        "para_eq",
        "orgpara_list",
        "para_list",
        "shared_fit_mse_0_7",
        "shared_fit_mse_hmean_0_7",
        "shared_fit_r2_0_7",
        "complexity",
        "loss",
        "score",
        "dataset",
        "fromdataset",

        # Useful provenance
        "SourceFolder",
    ]

    merged = merged[final_columns].copy()

    # Sort output by master ID, then rank within ID.
    # Use string helper because IDs may be numeric or symbolic.
    merged["_sort_id"] = merged["ID"].map(normalize_id)

    merged = (
        merged
        .sort_values(
            ["_sort_id", "TopRank"],
            kind="mergesort",
        )
        .drop(columns="_sort_id")
        .reset_index(drop=True)
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    configure(build_parser().parse_args(argv))
    print("=" * 90)
    print("Top group-leader formula merge")
    print("ROOT       :", ROOT.resolve())
    print("MASTER FILE:", MASTER_FILE.resolve())
    print("TOP_N      :", TOP_N)
    print("ROW FILTER : redundant == 0")
    print("OUTPUT     :", OUTPUT_CSV.resolve())
    print("=" * 90)

    master = load_master(
        MASTER_FILE
    )

    candidates, ok_count, fail_count = collect_top_formulas(
        ROOT,
        TOP_N,
    )

    if candidates.empty:
        print(
            "No candidate rows were collected. "
            "No output file was generated."
        )
        return

    result = merge_with_master(
        candidates,
        master,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",  # convenient for WPS/Excel on Windows
    )

    print("\n" + "=" * 90)
    print("Finished")
    print("Processed summary files :", ok_count)
    print("Failed summary files    :", fail_count)
    print("Output rows             :", len(result))
    print("Output file             :", OUTPUT_CSV.resolve())
    print("=" * 90)


if __name__ == "__main__":
    raise SystemExit(main())
