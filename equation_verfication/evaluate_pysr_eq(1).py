
from __future__ import annotations

# ============================================================
# IMPORTANT:
# Limit numerical-library threads BEFORE importing numpy/scipy.
# We parallelize by folder/process, so each worker should use
# one BLAS/OpenMP thread to avoid CPU oversubscription.
# ============================================================
import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import ast
import re
import hashlib
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("problem")
SUMMARY_NAME = "pysr_warm_0_7_summary.csv"

GROUP_IDS = tuple(range(8))       # 0.csv ... 7.csv

# Candidate-selection settings.
# Example: (7, 35) keeps formulas whose complexity is between 7 and 35,
# including both endpoints.
COMPLEXITY_RANGE = (1, 35)

# Evaluate only the N formulas with the lowest loss inside COMPLEXITY_RANGE.
# Set to None to evaluate every formula in the selected complexity range.
LOSS_TOP_N = 30

# Column names used to select candidate formulas. These can be changed when a
# summary file uses different names, without touching the selection logic.
LOSS_COLUMN = "loss"
COMPLEXITY_COLUMN = "complexity"

MAXFEV = 200_000

# ============================================================
# Multi-start retry configuration
# ============================================================
# First try always uses orgpara_list exactly.
# Only if that fit fails do we try RETRY_COUNT perturbed initial guesses.
RETRY_COUNT = 3

# Perturbation:
#   new_p0 = p0 + Normal(0, RETRY_REL_SCALE) * max(|p0|, RETRY_ABS_FLOOR)
#
# RETRY_ABS_FLOOR is important when an original parameter is 0;
# purely multiplicative perturbation would otherwise keep it at 0 forever.
RETRY_REL_SCALE = 0.50
RETRY_ABS_FLOOR = 1.0

# Deterministic multi-start guesses: same file/formula -> same retry guesses.
RETRY_BASE_SEED = 20260820

ERROR_MSE = 1.0e100
ERROR_R2 = -1.0e100

# Two main output columns requested.
MSE_COL = "shared_fit_mse_0_7"
R2_COL = "shared_fit_r2_0_7"
MSE_HMEAN_COL = "shared_fit_mse_hmean_0_7"
MSE_RMS_COL = "shared_fit_mse_rms_0_7"
R2_HMEAN_COL = "shared_fit_r2_hmean_0_7"
R2_RMS_COL = "shared_fit_r2_rms_0_7"
RESULT_COLS = (
    MSE_COL,
    R2_COL,
    MSE_HMEAN_COL,
    MSE_RMS_COL,
    R2_HMEAN_COL,
    R2_RMS_COL,
)

# Diagnostic columns.
SUCCESS_COL = "shared_fit_success_groups"
FAILED_COL = "shared_fit_failed_groups"
RETRY_COL = "shared_fit_retry_groups"

# None -> automatically use all available logical CPUs,
# but never more workers than folders.
#
# Examples:
# MAX_WORKERS = 8
# MAX_WORKERS = 16
MAX_WORKERS = None

# If the raw CSV has an explicit target column, set it here.
# None -> auto-detect y/target; otherwise use last numeric column.
TARGET_COLUMN = None

# If X columns must be forced, e.g. ["x0", "x1"], set them here.
# None -> match formula symbols first, then x0,x1,...,
# then first n numeric non-target columns.
FEATURE_COLUMNS = None

# Repeated formulas are marked with redundant=1 and are not evaluated.
SKIP_REDUNDANT = True

# Master redo switch:
#   0 = resume mode: skip formulas/files that already have complete results.
#   1 = redo mode: recompute selected formulas even if they were processed before.
#
# Normally this is the only variable you need to change when deciding whether
# previously processed results should be recalculated.
REDO_PROCESSED = 1

# Internal flags derived from REDO_PROCESSED.
# Keep these derived instead of editing them independently.
FORCE_RECOMPUTE = bool(REDO_PROCESSED)
PROCESS_ONLY_INCOMPLETE = not bool(REDO_PROCESSED)

# If WPS/Excel locks the original CSV, never lose the calculated result.
# The worker writes a .pending file and tries to atomically replace the original.
# If replacement is blocked, .pending is kept for recovery on the next run.
KEEP_PENDING_ON_LOCK = True

# Folder workers normally should not print every equation because
# output from multiple processes becomes unreadable.
VERBOSE_EQUATIONS_IN_WORKER = False


# ============================================================
# Parsing helpers
# ============================================================

def natural_symbol_key(sym: sp.Symbol):
    """Sort x0,x1,... numerically."""
    name = str(sym)
    m = re.fullmatch(r"x(\d+)", name)
    return (0, int(m.group(1))) if m else (1, name)


def parse_param_names(value) -> list[str]:
    """
    Accept:
        [p0,p1]
        ['p0','p1']
        p0,p1
        []
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    s = str(value).strip()
    if not s or s.lower() == "nan" or s == "[]":
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [
                str(v).strip().strip("'\"")
                for v in obj
                if str(v).strip()
            ]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    return [
        p.strip().strip("'\"")
        for p in s.split(",")
        if p.strip()
    ]


def parse_initial_values(value) -> list[float]:
    """
    Accept:
        [1.0,2.0]
        ['1/2','pi']
        1.0,2.0
        []
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    s = str(value).strip()
    if not s or s.lower() == "nan" or s == "[]":
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [float(sp.N(sp.sympify(v))) for v in obj]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    ans = []
    for p in s.split(","):
        p = p.strip().strip("'\"")
        if p:
            ans.append(float(sp.N(sp.sympify(p))))
    return ans


def parse_equation(text) -> sp.Expr:
    s = str(text).strip().replace("^", "**")

    local_dict = {
        "pi": sp.pi,
        "E": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "sign": sp.sign,
    }

    return sp.sympify(s, locals=local_dict)


# ============================================================
# Data discovery and cache
# ============================================================

def find_group_file(problem_dir: Path, gid: int) -> Path:
    """
    Prefer:
        problem_dir/0.csv
        ...
        problem_dir/7.csv

    If not directly present, recursively search below the folder.
    Exactly one file must be found.
    """
    direct = problem_dir / f"{gid}.csv"
    if direct.is_file():
        return direct

    matches = [
        p for p in problem_dir.rglob(f"{gid}.csv")
        if p.name == f"{gid}.csv"
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Cannot find {gid}.csv below {problem_dir}"
        )

    raise RuntimeError(
        f"More than one {gid}.csv found below {problem_dir}:\n"
        + "\n".join(str(p) for p in matches)
    )


def choose_target_column(df: pd.DataFrame) -> str:
    if TARGET_COLUMN is not None:
        if TARGET_COLUMN not in df.columns:
            raise KeyError(
                f"TARGET_COLUMN={TARGET_COLUMN!r} not in {list(df.columns)}"
            )
        return TARGET_COLUMN

    for c in ("y", "Y", "target", "Target", "TARGET"):
        if c in df.columns:
            return c

    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric:
        raise ValueError("No numeric columns in data file")

    return numeric[-1]


def load_folder_data(folder: Path):
    """
    Read 0.csv ... 7.csv ONCE for one folder.

    Each worker owns exactly one folder at a time, so this cache is local
    to that worker and needs no inter-process synchronization.
    """
    cache = {}

    for gid in GROUP_IDS:
        file = find_group_file(folder, gid)
        cache[gid] = {
            "file": file,
            "df": pd.read_csv(file),
        }

    return cache


def read_data_from_cache(
    cached_item: dict,
    x_symbols: list[sp.Symbol],
):
    """
    Build X,y from an already-loaded DataFrame.

    This avoids rereading 0.csv ... 7.csv for every candidate formula.
    """
    file = cached_item["file"]
    df = cached_item["df"]

    target_col = choose_target_column(df)
    y = pd.to_numeric(
        df[target_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    nx = len(x_symbols)

    if nx == 0:
        X = np.empty((len(df), 0), dtype=float)

    else:
        if FEATURE_COLUMNS is not None:
            xcols = list(FEATURE_COLUMNS)

            if len(xcols) != nx:
                raise ValueError(
                    f"{file}: formula needs {nx} X variables, "
                    f"FEATURE_COLUMNS={xcols}"
                )

        else:
            symbol_names = [str(s) for s in x_symbols]

            # 1. Exact formula variable names.
            if all(c in df.columns for c in symbol_names):
                xcols = symbol_names

            else:
                # 2. Standard x0,x1,... columns.
                xstyle = [f"x{i}" for i in range(nx)]

                if all(c in df.columns for c in xstyle):
                    xcols = xstyle

                else:
                    # 3. Fallback: first nx numeric non-target columns.
                    numeric_x = [
                        c for c in df.columns
                        if c != target_col
                        and pd.api.types.is_numeric_dtype(df[c])
                    ]

                    if len(numeric_x) < nx:
                        raise ValueError(
                            f"{file}: formula needs {nx} X variables "
                            f"{symbol_names}, but numeric non-target "
                            f"columns are {numeric_x}"
                        )

                    xcols = numeric_x[:nx]

        X = (
            df[xcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

    good = np.isfinite(y)

    if X.shape[1] > 0:
        good &= np.all(np.isfinite(X), axis=1)

    X = X[good]
    y = y[good]

    if len(y) == 0:
        raise ValueError(f"{file}: no finite data")

    return X, y


# ============================================================
# Curve fitting
# ============================================================

def make_wrapper(f, nx: int):
    def wrapper(Xdata, *params):
        Xdata = np.asarray(Xdata, dtype=float)

        if Xdata.ndim == 1:
            if nx != 1:
                return np.full(
                    Xdata.shape[0],
                    ERROR_MSE,
                    dtype=float,
                )
            Xdata = Xdata.reshape(-1, 1)

        args = list(params) + [
            Xdata[:, i] for i in range(nx)
        ]

        try:
            with np.errstate(all="ignore"):
                pred = f(*args)

            pred = np.asarray(pred, dtype=float)

        except Exception:
            return np.full(
                Xdata.shape[0],
                ERROR_MSE,
                dtype=float,
            )

        # Constant expression -> broadcast scalar prediction.
        if pred.ndim == 0:
            pred = np.full(
                Xdata.shape[0],
                float(pred),
                dtype=float,
            )
        else:
            pred = np.ravel(pred)

        if (
            pred.shape[0] != Xdata.shape[0]
            or np.any(~np.isfinite(pred))
        ):
            return np.full(
                Xdata.shape[0],
                ERROR_MSE,
                dtype=float,
            )

        return pred

    return wrapper


def _fit_metrics(wrapper, X, y, p0):
    """
    Run one curve_fit attempt and validate the result.

    Returns:
        ok, mse, r2, popt

    A fit with a poor (even negative) R^2 is still a VALID fit as long as
    optimization returns finite parameters and finite predictions.
    """
    try:
        if len(p0) > 0:
            popt, _ = curve_fit(
                wrapper,
                X,
                y,
                p0=np.asarray(p0, dtype=float),
                maxfev=MAXFEV,
            )
        else:
            popt = np.array([], dtype=float)

        popt = np.asarray(popt, dtype=float)

        if np.any(~np.isfinite(popt)):
            raise FloatingPointError("non-finite fitted parameters")

        pred = wrapper(X, *popt)

        if (
            np.any(~np.isfinite(pred))
            or np.any(np.abs(pred) >= ERROR_MSE)
        ):
            raise FloatingPointError("invalid prediction")

        residual = y - pred

        if np.any(~np.isfinite(residual)):
            raise FloatingPointError("invalid residual")

        mse = float(np.mean(residual ** 2))
        sse = float(np.sum(residual ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))

        if sst > 0:
            r2 = float(1.0 - sse / sst)
        else:
            r2 = 1.0 if sse <= 1e-24 else ERROR_R2

        if not np.isfinite(mse):
            raise FloatingPointError("non-finite MSE")

        if not np.isfinite(r2):
            r2 = ERROR_R2

        return True, mse, r2, popt

    except Exception:
        return False, ERROR_MSE, ERROR_R2, None


def _deterministic_retry_rng(cached_item, expr):
    """
    Create a reproducible RNG for one (data file, equation) pair.

    Python's built-in hash() is intentionally randomized between processes,
    so use SHA256 instead.
    """
    key = (
        f"{RETRY_BASE_SEED}|"
        f"{cached_item['file']}|"
        f"{str(expr)}"
    ).encode("utf-8", errors="replace")

    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)

    return np.random.default_rng(seed)


def _make_retry_guesses(p0, cached_item, expr):
    """
    Generate perturbed starting points ONLY after the original p0 has failed.

    For each parameter:
        scale_j = max(abs(p0_j), RETRY_ABS_FLOOR)
        guess_j = p0_j + Normal(0, RETRY_REL_SCALE) * scale_j

    This works for p0_j == 0 as well.
    """
    p0 = np.asarray(p0, dtype=float)

    if len(p0) == 0 or RETRY_COUNT <= 0:
        return []

    rng = _deterministic_retry_rng(cached_item, expr)

    scale = np.maximum(
        np.abs(p0),
        RETRY_ABS_FLOOR,
    )

    guesses = []

    for _ in range(RETRY_COUNT):
        jitter = rng.normal(
            loc=0.0,
            scale=RETRY_REL_SCALE,
            size=len(p0),
        )

        guess = p0 + jitter * scale

        if np.all(np.isfinite(guess)):
            guesses.append(guess)

    return guesses


def fit_one_dataset(
    expr,
    param_symbols,
    x_symbols,
    p0,
    cached_item,
):
    """
    Fit one shared symbolic structure to one dataset.

    Strategy:
      1. First try: orgpara_list exactly.
      2. Only if first try fails, perform multiple perturbed-start retries.
      3. If several retries succeed, keep the one with the lowest MSE.
      4. A finite but poor fit (e.g. R^2 < 0) is NOT treated as optimizer
         failure; it is kept as a legitimate bad score.

    Returns:
        ok, mse, r2, popt, retry_used, attempts_used
    """
    X, y = read_data_from_cache(
        cached_item,
        x_symbols,
    )

    f = sp.lambdify(
        list(param_symbols) + list(x_symbols),
        expr,
        "numpy",
    )

    wrapper = make_wrapper(
        f,
        len(x_symbols),
    )

    p0 = np.asarray(p0, dtype=float)

    # --------------------------------------------------------
    # Attempt 1: exact original orgpara_list
    # --------------------------------------------------------
    ok, mse, r2, popt = _fit_metrics(
        wrapper,
        X,
        y,
        p0,
    )

    if ok:
        return (
            True,
            mse,
            r2,
            popt,
            False,   # retry_used
            1,       # attempts_used
        )

    # --------------------------------------------------------
    # Original attempt failed -> multi-start retry
    # --------------------------------------------------------
    retry_guesses = _make_retry_guesses(
        p0,
        cached_item,
        expr,
    )

    best_mse = np.inf
    best_r2 = ERROR_R2
    best_popt = None

    attempts_used = 1

    for guess in retry_guesses:
        attempts_used += 1

        ok_i, mse_i, r2_i, popt_i = _fit_metrics(
            wrapper,
            X,
            y,
            guess,
        )

        if not ok_i:
            continue

        # Among all successful retry starts, use the lowest-MSE solution.
        if mse_i < best_mse:
            best_mse = mse_i
            best_r2 = r2_i
            best_popt = popt_i

    if best_popt is not None:
        return (
            True,
            float(best_mse),
            float(best_r2),
            best_popt,
            True,          # retry_used
            attempts_used,
        )

    # All original + retry starts failed.
    return (
        False,
        ERROR_MSE,
        ERROR_R2,
        None,
        True,
        attempts_used,
    )


def aggregate_group_metrics(values, error_value, allow_harmonic=True):
    """Return arithmetic mean, harmonic mean, and RMS for group metrics."""
    values = np.asarray(values, dtype=float)

    mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(values ** 2)))

    if allow_harmonic and np.all(values > 0):
        harmonic = float(len(values) / np.sum(1.0 / values))
    else:
        harmonic = error_value

    return tuple(
        value if np.isfinite(value) else error_value
        for value in (mean, harmonic, rms)
    )


def evaluate_row(
    row: pd.Series,
    data_cache: dict,
):
    """
    One candidate para_eq is fitted independently to 0...7.

    Parameters are reset to orgpara_list before each group fit.

    Final values follow the semantics of the user's reference code:
        shared_fit_mse_0_7 = mean(MSE_0, ..., MSE_7)
        shared_fit_r2_0_7  = mean(R2_0,  ..., R2_7)
        Additional MSE/R2 columns contain the harmonic mean and RMS.

    If even one of the eight fits fails, the common formula is marked failed.
    """
    expr = parse_equation(
        row["para_eq"]
    )

    param_names = parse_param_names(
        row["para_list"]
    )

    param_symbols = [
        sp.Symbol(s)
        for s in param_names
    ]

    p0 = parse_initial_values(
        row["orgpara_list"]
    )

    if len(param_symbols) != len(p0):
        raise ValueError(
            "para_list/orgpara_list length mismatch: "
            f"{param_names} vs {p0}"
        )

    param_set = set(param_symbols)

    x_symbols = sorted(
        [
            s
            for s in expr.free_symbols
            if s not in param_set
        ],
        key=natural_symbol_key,
    )

    unused = [
        s
        for s in param_symbols
        if s not in expr.free_symbols
    ]

    if unused:
        raise ValueError(
            f"Parameters absent from para_eq: {unused}"
        )

    mse_list = []
    r2_list = []

    success = []
    failed = []
    retried = []

    for gid in GROUP_IDS:
        ok, mse, r2, _, retry_used, attempts_used = fit_one_dataset(
            expr=expr,
            param_symbols=param_symbols,
            x_symbols=x_symbols,
            p0=p0,
            cached_item=data_cache[gid],
        )

        if retry_used:
            retried.append(gid)

        if ok:
            success.append(gid)
            mse_list.append(mse)
            r2_list.append(r2)

        else:
            failed.append(gid)

    # A shared formula must fit all 8 datasets.
    if failed:
        return (
            ERROR_MSE,
            ERROR_R2,
            ERROR_MSE,
            ERROR_MSE,
            ERROR_R2,
            ERROR_R2,
            success,
            failed,
            retried,
        )

    mean_mse, hmean_mse, rms_mse = aggregate_group_metrics(
        mse_list,
        ERROR_MSE,
    )
    mean_r2, hmean_r2, rms_r2 = aggregate_group_metrics(
        r2_list,
        ERROR_R2,
    )

    return (
        mean_mse,
        mean_r2,
        hmean_mse,
        rms_mse,
        hmean_r2,
        rms_r2,
        success,
        failed,
        retried,
    )



# ============================================================
# Resume / WPS-lock-safe helpers
# ============================================================

def result_columns_complete(df: pd.DataFrame) -> bool:
    """
    True when every row already has a numeric finite MSE and R2 result.

    ERROR_MSE=1e100 and ERROR_R2=-1e100 are finite by design, so a formula
    that was evaluated but failed on one of the 0-7 groups still counts as
    'processed'. This prevents pointless recomputation.
    """
    if any(column not in df.columns for column in RESULT_COLS):
        return False

    if len(df) == 0:
        return True

    return all(
        np.all(np.isfinite(
            pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        ))
        for column in RESULT_COLS
    )


def summary_file_complete(summary_file: Path) -> bool:
    """
    Lightweight pre-check used by main() before a folder is submitted to the
    process pool.
    """
    try:
        df = pd.read_csv(summary_file)
        return result_columns_complete(df)
    except Exception:
        return False


def recovery_files(summary_file: Path) -> list[Path]:
    """
    New version uses .pending.
    Old version used .tmp, so support both for recovery.
    """
    return [
        summary_file.with_suffix(summary_file.suffix + ".pending"),
        summary_file.with_suffix(summary_file.suffix + ".tmp"),
    ]


def recovery_file_complete(path: Path) -> bool:
    if not path.is_file():
        return False

    try:
        df = pd.read_csv(path)
        return result_columns_complete(df)
    except Exception:
        return False


def try_recover_saved_result(summary_file: Path):
    """
    Try to promote a complete .pending/.tmp result left by a previous run.

    Returns:
        ("none", None)
            no usable recovery file exists.

        ("recovered", recovery_path)
            recovery file was atomically moved onto summary_file.

        ("locked", recovery_path)
            recovery file is complete, but WPS/Excel still locks summary_file.
            Do NOT recompute; keep the recovery file and retry next run.
    """
    for recovery in recovery_files(summary_file):
        if not recovery_file_complete(recovery):
            continue

        try:
            os.replace(recovery, summary_file)
            return "recovered", recovery

        except PermissionError:
            # A complete calculation already exists in recovery.
            # Recomputing would only waste CPU.
            return "locked", recovery

        except OSError:
            # On Windows some file-lock conditions may surface as OSError
            # subclasses other than PermissionError. Preserve the result.
            if recovery.exists():
                return "locked", recovery
            raise

    return "none", None


def save_dataframe_lock_safe(df: pd.DataFrame, summary_file: Path):
    """
    Save without losing results if WPS/Excel has summary_file open.

    1. Write the complete result to:
           pysr_warm_0_7_summary.csv.pending

    2. Try os.replace(pending, summary).

    3. If Windows refuses because the target is locked:
           - leave .pending untouched;
           - report 'locked';
           - worker can immediately move on to another folder.

    Returns:
        ("saved", pending_path)
        ("locked", pending_path)
    """
    pending = summary_file.with_suffix(
        summary_file.suffix + ".pending"
    )

    # Writing pending does not touch the WPS-opened original file.
    df.to_csv(pending, index=False)

    try:
        os.replace(pending, summary_file)
        return "saved", pending

    except PermissionError:
        if not KEEP_PENDING_ON_LOCK:
            raise
        return "locked", pending

    except OSError:
        # Preserve pending on Windows-like sharing/locking errors.
        if KEEP_PENDING_ON_LOCK and pending.exists():
            return "locked", pending
        raise


def mark_redundant_formulas(df: pd.DataFrame, candidate_indices):
    """Mark mathematically equivalent candidate formulas as redundant=1."""
    if "redundant" not in df.columns:
        df["redundant"] = 0
    else:
        df["redundant"] = pd.to_numeric(
            df["redundant"],
            errors="coerce",
        ).fillna(0).astype(int)

    representatives = []
    parsed = {}

    for idx in candidate_indices:
        if df.at[idx, "redundant"] > 0:
            continue

        try:
            expression = parse_equation(df.at[idx, "para_eq"])
        except Exception:
            continue

        is_redundant = False
        for representative_idx in representatives:
            representative = parsed[representative_idx]
            try:
                equivalent = sp.simplify(expression - representative) == 0
            except Exception:
                equivalent = expression.equals(representative) is True

            if equivalent:
                df.at[idx, "redundant"] = 1
                is_redundant = True
                break

        if not is_redundant:
            df.at[idx, "redundant"] = 0
            representatives.append(idx)
            parsed[idx] = expression

    return df


def select_candidate_indices(df: pd.DataFrame) -> pd.Index:
    """Select lowest-loss formulas inside the configured complexity range."""
    missing = {
        LOSS_COLUMN,
        COMPLEXITY_COLUMN,
    } - set(df.columns)

    if missing:
        raise KeyError(
            "Missing candidate-selection columns: "
            f"{sorted(missing)}"
        )

    try:
        min_complexity, max_complexity = COMPLEXITY_RANGE
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "COMPLEXITY_RANGE must contain exactly two values: "
            "(minimum, maximum)"
        ) from exc

    if min_complexity > max_complexity:
        raise ValueError(
            "COMPLEXITY_RANGE minimum must not exceed its maximum: "
            f"{COMPLEXITY_RANGE}"
        )

    if LOSS_TOP_N is not None:
        if isinstance(LOSS_TOP_N, bool) or not isinstance(LOSS_TOP_N, int):
            raise TypeError("LOSS_TOP_N must be a positive integer or None")
        if LOSS_TOP_N <= 0:
            raise ValueError("LOSS_TOP_N must be greater than 0 or None")

    loss_values = pd.to_numeric(
        df[LOSS_COLUMN],
        errors="coerce",
    )
    complexity_values = pd.to_numeric(
        df[COMPLEXITY_COLUMN],
        errors="coerce",
    )
    in_range = complexity_values.between(
        min_complexity,
        max_complexity,
        inclusive="both",
    )

    candidates = (
        loss_values.where(in_range)
        .dropna()
        .sort_values(kind="stable")
    )

    if LOSS_TOP_N is not None:
        candidates = candidates.head(LOSS_TOP_N)

    return candidates.index


# ============================================================
# One independent folder = one process task
# ============================================================

def process_summary(summary_file: Path):
    """
    Entire work for ONE folder.

    Different folders are independent, so no inter-folder lock is needed.
    The only expected write conflict is an external program such as WPS/Excel.
    """
    folder = summary_file.parent

    # --------------------------------------------------------
    # 0. First recover a complete result left by an earlier run.
    #    This is especially useful after:
    #        df.to_csv(...tmp) succeeded
    #        os.replace(...tmp, summary) failed because WPS locked it.
    # --------------------------------------------------------
    recovery_status, recovery_path = try_recover_saved_result(summary_file)

    if recovery_status == "recovered":
        return {
            "status": "recovered",
            "evaluated": 0,
            "total_rows": len(pd.read_csv(summary_file)),
            "recovery_path": str(recovery_path),
        }

    if recovery_status == "locked":
        return {
            "status": "locked",
            "evaluated": 0,
            "total_rows": 0,
            "recovery_path": str(recovery_path),
        }

    # --------------------------------------------------------
    # 1. If this summary is already completely evaluated,
    #    do nothing.
    # --------------------------------------------------------
    df = pd.read_csv(summary_file)

    if PROCESS_ONLY_INCOMPLETE and result_columns_complete(df):
        return {
            "status": "skipped_complete",
            "evaluated": 0,
            "total_rows": len(df),
            "recovery_path": "",
        }

    # --------------------------------------------------------
    # 2. This folder really needs work. Only now load 0-7 data.
    # --------------------------------------------------------
    data_cache = load_folder_data(folder)

    required = {
        "para_eq",
        "para_list",
        "orgpara_list",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing columns in {summary_file}: "
            f"{sorted(missing)}"
        )

    if MSE_COL not in df.columns:
        df[MSE_COL] = np.nan

    if R2_COL not in df.columns:
        df[R2_COL] = np.nan

    for column in RESULT_COLS[2:]:
        if column not in df.columns:
            df[column] = np.nan

    if SUCCESS_COL not in df.columns:
        df[SUCCESS_COL] = ""

    if FAILED_COL not in df.columns:
        df[FAILED_COL] = ""

    if RETRY_COL not in df.columns:
        df[RETRY_COL] = ""

    # Keep the full dataframe so formulas outside the selected set remain untouched.
    candidate_indices = select_candidate_indices(df)
    mark_redundant_formulas(df, candidate_indices)
    n = len(candidate_indices)
    evaluated = 0

    for pos, idx in enumerate(
        candidate_indices,
        start=1,
    ):
        row = df.loc[idx]

        if SKIP_REDUNDANT and df.at[idx, "redundant"] > 0:
            continue

        if not FORCE_RECOMPUTE:
            old_mse = pd.to_numeric(
                pd.Series(
                    [row.get(MSE_COL)]
                ),
                errors="coerce",
            ).iloc[0]

            old_r2 = pd.to_numeric(
                pd.Series(
                    [row.get(R2_COL)]
                ),
                errors="coerce",
            ).iloc[0]

            old_results = [
                old_mse,
                old_r2,
                *(
                    pd.to_numeric(
                        pd.Series([row.get(column)]),
                        errors="coerce",
                    ).iloc[0]
                    for column in RESULT_COLS[2:]
                ),
            ]

            if all(np.isfinite(value) for value in old_results):
                continue

        if VERBOSE_EQUATIONS_IN_WORKER:
            print(
                f"{folder.name}: "
                f"[{pos}/{n}] "
                f"{str(row['para_eq'])[:100]}",
                flush=True,
            )

        try:
            (
                mse,
                r2,
                hmean_mse,
                rms_mse,
                hmean_r2,
                rms_r2,
                success,
                failed,
                retried,
            ) = evaluate_row(
                row,
                data_cache,
            )

        except Exception as e:
            mse = ERROR_MSE
            r2 = ERROR_R2
            hmean_mse = ERROR_MSE
            rms_mse = ERROR_MSE
            hmean_r2 = ERROR_R2
            rms_r2 = ERROR_R2
            success = []
            failed = list(GROUP_IDS)
            retried = []

            if VERBOSE_EQUATIONS_IN_WORKER:
                print(
                    f"{folder.name}: "
                    f"row={idx} ERROR: {e}",
                    flush=True,
                )

        df.at[idx, MSE_COL] = mse
        df.at[idx, R2_COL] = r2
        df.at[idx, MSE_HMEAN_COL] = hmean_mse
        df.at[idx, MSE_RMS_COL] = rms_mse
        df.at[idx, R2_HMEAN_COL] = hmean_r2
        df.at[idx, R2_RMS_COL] = rms_r2

        df.at[idx, SUCCESS_COL] = ",".join(
            map(str, success)
        )

        df.at[idx, FAILED_COL] = ",".join(
            map(str, failed)
        )

        df.at[idx, RETRY_COL] = ",".join(
            map(str, retried)
        )

        evaluated += 1

    # WPS/Excel-lock-safe save.
    save_status, pending_path = save_dataframe_lock_safe(
        df,
        summary_file,
    )

    return {
        "status": save_status,
        "evaluated": evaluated,
        "total_rows": len(df),
        "recovery_path": str(pending_path) if save_status == "locked" else "",
    }


def process_summary_worker(
    summary_file_str: str,
):
    """
    Top-level worker function: required for Windows multiprocessing pickling.
    """
    summary_file = Path(
        summary_file_str
    )

    start = time.perf_counter()

    try:
        info = process_summary(
            summary_file
        )

        status = info["status"]

        return {
            # "locked" is not a computation failure: its result is safely
            # preserved in .pending/.tmp.
            "ok": status != "error",
            "status": status,
            "summary": str(summary_file),
            "folder": str(summary_file.parent),
            "evaluated": info.get("evaluated", 0),
            "total_rows": info.get("total_rows", 0),
            "recovery_path": info.get("recovery_path", ""),
            "seconds": time.perf_counter() - start,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "status": "error",
            "summary": str(summary_file),
            "folder": str(summary_file.parent),
            "evaluated": 0,
            "total_rows": 0,
            "recovery_path": "",
            "seconds": time.perf_counter() - start,
            "error": (
                f"{type(e).__name__}: {e}\n"
                + traceback.format_exc()
            ),
        }


# ============================================================
# Progress helpers
# ============================================================

def format_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"

    seconds = max(
        0,
        int(seconds),
    )

    h, rem = divmod(
        seconds,
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


# ============================================================
# Parallel main
# ============================================================

def main():
    if not ROOT.exists():
        raise FileNotFoundError(
            f"ROOT does not exist: "
            f"{ROOT.resolve()}"
        )

    all_summaries = sorted(
        ROOT.rglob(SUMMARY_NAME)
    )

    print(
        "ROOT:",
        ROOT.resolve(),
    )

    print(
        f"Found {len(all_summaries)} "
        f"{SUMMARY_NAME} files."
    )

    print(
        "Candidate selection: "
        f"{COMPLEXITY_COLUMN} in {COMPLEXITY_RANGE}, "
        f"lowest {LOSS_TOP_N if LOSS_TOP_N is not None else 'all'} "
        f"by {LOSS_COLUMN}."
    )

    print(
        "Redo processed results: "
        + ("YES" if REDO_PROCESSED else "NO (resume mode)")
    )

    if not all_summaries:
        return

    # --------------------------------------------------------
    # Resume scan:
    #   - already complete -> skip
    #   - incomplete -> submit
    #   - a complete .tmp/.pending may still exist from a WPS lock;
    #     submit it so the worker can promote it without recomputation.
    # --------------------------------------------------------
    summaries = []
    already_complete = 0

    for summary in all_summaries:
        has_complete_recovery = any(
            recovery_file_complete(p)
            for p in recovery_files(summary)
        )

        if has_complete_recovery:
            summaries.append(summary)
            continue

        if PROCESS_ONLY_INCOMPLETE and summary_file_complete(summary):
            already_complete += 1
            continue

        summaries.append(summary)

    total = len(summaries)

    print(
        f"Already complete and skipped: {already_complete}"
    )

    print(
        f"Folders requiring recovery/evaluation: {total}"
    )

    if total == 0:
        print("Nothing to do.")
        return

    cpu_count = (
        os.cpu_count()
        or 1
    )

    if MAX_WORKERS is None:
        workers = min(
            total,
            cpu_count,
        )
    else:
        workers = min(
            total,
            max(1, int(MAX_WORKERS)),
        )

    print(
        f"CPU logical cores: {cpu_count}"
    )

    print(
        f"Parallel folder workers: {workers}"
    )

    print(
        "Parallel unit: one "
        "pysr_warm_0_7_summary.csv folder "
        "per process."
    )

    print(
        "BLAS/OpenMP threads per worker: 1"
    )

    overall_start = time.perf_counter()

    ok_count = 0
    fail_count = 0
    locked_count = 0
    recovered_count = 0
    skipped_count = 0
    completed = 0

    # Windows-safe because this function is called only under
    # if __name__ == "__main__".
    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        future_to_summary = {
            executor.submit(
                process_summary_worker,
                str(summary),
            ): summary
            for summary in summaries
        }

        for future in as_completed(
            future_to_summary
        ):
            completed += 1

            summary = future_to_summary[
                future
            ]

            try:
                result = future.result()

            except Exception as e:
                result = {
                    "ok": False,
                    "status": "error",
                    "folder": str(summary.parent),
                    "summary": str(summary),
                    "evaluated": 0,
                    "total_rows": 0,
                    "recovery_path": "",
                    "seconds": 0.0,
                    "error": (
                        f"Worker process exception: {e}"
                    ),
                }

            elapsed = (
                time.perf_counter()
                - overall_start
            )

            # Throughput-based ETA. It starts correcting itself
            # as more parallel folders finish.
            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                total - completed
            )

            eta = (
                remaining / rate
                if rate > 0
                else float("nan")
            )

            percent = (
                100.0
                * completed
                / total
            )

            status = result.get("status", "error")

            if status == "saved":
                ok_count += 1

                print(
                    f"[{completed:>4}/{total}] "
                    f"{percent:6.2f}%  "
                    f"DONE  "
                    f"{result['folder']}  "
                    f"rows={result['evaluated']}/"
                    f"{result['total_rows']}  "
                    f"task={format_time(result['seconds'])}  "
                    f"ETA≈{format_time(eta)}",
                    flush=True,
                )

            elif status == "recovered":
                ok_count += 1
                recovered_count += 1

                print(
                    f"[{completed:>4}/{total}] "
                    f"{percent:6.2f}%  "
                    f"RECOVERED  "
                    f"{result['folder']}  "
                    f"(used previous .tmp/.pending; no refit)  "
                    f"ETA≈{format_time(eta)}",
                    flush=True,
                )

            elif status == "skipped_complete":
                ok_count += 1
                skipped_count += 1

                print(
                    f"[{completed:>4}/{total}] "
                    f"{percent:6.2f}%  "
                    f"SKIP-COMPLETE  "
                    f"{result['folder']}  "
                    f"ETA≈{format_time(eta)}",
                    flush=True,
                )

            elif status == "locked":
                locked_count += 1

                print(
                    f"[{completed:>4}/{total}] "
                    f"{percent:6.2f}%  "
                    f"LOCKED-BY-WPS  "
                    f"{result['folder']}  "
                    f"result kept at: {result['recovery_path']}  "
                    f"ETA≈{format_time(eta)}",
                    flush=True,
                )

            else:
                fail_count += 1

                print(
                    f"[{completed:>4}/{total}] "
                    f"{percent:6.2f}%  "
                    f"FAILED  "
                    f"{result['folder']}  "
                    f"ETA≈{format_time(eta)}",
                    flush=True,
                )

                print(
                    result["error"],
                    flush=True,
                )

    total_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print(
        "\n" + "=" * 100
    )

    print("Finished.")

    print(
        "Successful folders:",
        ok_count,
    )

    print(
        "Recovered without refit:",
        recovered_count,
    )

    print(
        "Skipped complete:",
        already_complete + skipped_count,
    )

    print(
        "Still locked by WPS/Excel:",
        locked_count,
    )

    print(
        "Failed folders:",
        fail_count,
    )

    print(
        "Total elapsed:",
        format_time(total_elapsed),
    )


if __name__ == "__main__":
    # Required/recommended for Windows multiprocessing,
    # especially if the script is later packaged.
    mp.freeze_support()
    main()
