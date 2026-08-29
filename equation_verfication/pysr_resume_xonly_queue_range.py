from pathlib import Path
import json, re, warnings, os, multiprocessing as mp, time, contextlib
import shutil, zipfile, uuid
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
import os

ROOT = Path(os.getcwd())
EXCEL_FILE = ROOT / "physicsMDSR_Range.xlsx"
DATA_ROOT = ROOT / "problem"
SHEET_NAME = "Sampling design"
DATASET_INDICES = range(8)

# -----------------------------------------------------------------------------
# TASK SELECTION
# -----------------------------------------------------------------------------
# Inclusive subfolder-ID range. Examples:
#   TASK_RANGE = None        -> process all discovered subfolders
#   TASK_RANGE = "P10-P20"   -> process only P10, P11, ..., P20
#   TASK_RANGE = "P10"       -> process only P10
# The filter is applied to the actual discovered folder names, so zero-padding
# such as P01/P001 is preserved automatically.
TASK_RANGE = "P01-P59"

# Optional exact-ID whitelist retained for compatibility. Normally leave None.
# If both TASK_RANGE and TARGET_ROW_IDS are set, BOTH filters must match.
TARGET_ROW_IDS = None

niterations = 400
populations = 20
population_size = 40
nodemaxsize = 50
min_complexity = 7
max_complexity = 40
top_n = 10

OUTER_PROCESSES = 4
PYSR_PROCS_PER_PROCESS = 3

# IMPORTANT: new run id because feature layout changed.
# Old warm_0_7 checkpoints included parameters/constants as input features.
RUN_ID = "warm_0_7_xonly_v2"
STATE_DIR_NAME = f".state_{RUN_ID}"

# WPS/file-lock resilience.
CACHE_DIR = ROOT / ".pysr_cache"
EXCEL_SNAPSHOT = CACHE_DIR / "physicsMDSR_Range_snapshot.xlsx"
FILE_RETRY_COUNT = 15
FILE_RETRY_SECONDS = 2.0

binary_operators = ["+", "-", "*", "/", "^"]
unary_operators = ["cos", "sin", "exp", "sqrt", "log", "tanh", "acos", "atan", "atanh"]

nested_constraints = {
    "sin": {"sin": 0, "cos": 0, "acos": 0, "atanh": 0, "tanh": 0},
    "cos": {"sin": 0, "cos": 0, "acos": 0, "atanh": 0, "tanh": 0},
    "exp": {"exp": 0, "log": 0},
    "log": {"log": 0, "exp": 0},
    "sqrt": {"sqrt": 1},
    "acos": {"sin": 0, "cos": 0, "acos": 0, "atanh": 0, "tanh": 0},
    "tanh": {"sin": 0, "cos": 0, "acos": 0, "atanh": 0, "tanh": 0},
    "atanh": {"sin": 0, "cos": 0, "acos": 0, "atanh": 0, "tanh": 0},
}

REQUIRE_ALL_FILES = True
SAVE_ALL_EQUATIONS = True


def _alpha_index(i):
    letters = "abcdefghijklmnopqrstuvwxyz"
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = letters[r] + s
    return s


# -----------------------------------------------------------------------------
# 1) ONLY independent variables are PySR inputs.
# Parameters/fixed constants are retained only as metadata for result files.
# -----------------------------------------------------------------------------
def make_feature_maps(independent_vars):
    original_to_safe, safe_to_original = {}, {}
    for i, name in enumerate(independent_vars):
        safe = f"xv_{_alpha_index(i)}"
        original_to_safe[name] = safe
        safe_to_original[safe] = name
    return original_to_safe, safe_to_original


def restore_expression_names(expr, safe_to_original):
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)
    return expr.xreplace({
        sp.Symbol(safe): sp.Symbol(original)
        for safe, original in safe_to_original.items()
    })


RANGE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)


def parse_independent_vars(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_parameter_names(value):
    if pd.isna(value):
        return []
    return [name for name, _, _ in RANGE_RE.findall(str(value))]


def parse_fixed_constants(value):
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"(none)", "none", "nan"}:
        return {}
    out = {}
    for part in text.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, val = part.split("=", 1)
        out[name.strip()] = float(val.strip())
    return out


def parse_para_value(value):
    if pd.isna(value):
        raise ValueError("paraValue 为空")
    data = json.loads(str(value))
    if len(data) < 8:
        raise ValueError(f"paraValue 只有 {len(data)} 组，至少需要 8 组")
    return data


# -----------------------------------------------------------------------------
# 2a) WPS-safe Excel loading.
# First make/read a validated snapshot. If the live workbook is temporarily
# locked by WPS, use the last valid snapshot instead of crashing immediately.
# -----------------------------------------------------------------------------
def load_excel_resilient():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last_exc = None

    for attempt in range(1, FILE_RETRY_COUNT + 1):
        tmp = CACHE_DIR / f"excel_snapshot_{os.getpid()}_{uuid.uuid4().hex}.xlsx"
        try:
            shutil.copyfile(EXCEL_FILE, tmp)

            # Detect a half-written/corrupted XLSX while WPS is saving it.
            with zipfile.ZipFile(tmp, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise zipfile.BadZipFile(f"损坏成员: {bad_member}")

            df = pd.read_excel(tmp, sheet_name=SHEET_NAME)

            # Only replace the cache AFTER successful validation/read.
            os.replace(tmp, EXCEL_SNAPSHOT)
            return df

        except (PermissionError, OSError, zipfile.BadZipFile, EOFError, KeyError) as exc:
            last_exc = exc
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

            if attempt < FILE_RETRY_COUNT:
                time.sleep(FILE_RETRY_SECONDS)

    # Live file stayed locked/unreadable. Use last known-good snapshot.
    if EXCEL_SNAPSHOT.exists():
        print(
            f"\n[WARN] Excel 可能被 WPS 锁定/保存中；使用上次可读快照: {EXCEL_SNAPSHOT}",
            flush=True,
        )
        return pd.read_excel(EXCEL_SNAPSHOT, sheet_name=SHEET_NAME)

    raise RuntimeError(
        f"无法读取 Excel，且没有可用快照: {EXCEL_FILE}; last_error={last_exc!r}"
    )


def load_metadata():
    df = load_excel_resilient()
    required = {"ID", "IndependentVars", "ParameterRange", "FixedConstantValues", "paraValue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少列: {sorted(missing)}")

    result = {}
    for _, row in df.iterrows():
        row_id = str(row["ID"]).strip()
        independent_vars = parse_independent_vars(row["IndependentVars"])
        parameter_names = parse_parameter_names(row["ParameterRange"])
        fixed_constants = parse_fixed_constants(row["FixedConstantValues"])
        para_values = parse_para_value(row["paraValue"])

        for i, vec in enumerate(para_values):
            if len(vec) != len(parameter_names):
                raise ValueError(
                    f"{row_id}: paraValue[{i}]={vec} 与 ParameterRange 参数 {parameter_names} 维数不一致"
                )

        result[row_id] = {
            "IndependentVars": independent_vars,
            "ParameterNames": parameter_names,
            "FixedConstants": fixed_constants,
            "ParaValues": para_values,
        }
    return result


# -----------------------------------------------------------------------------
# 2b) WPS-safe output writing.
# Write temp file first; replace target atomically. If WPS keeps the target
# locked, save to a unique sibling file and continue instead of killing worker.
# -----------------------------------------------------------------------------
def _replace_temp_with_lock_fallback(tmp_path, target_path):
    target_path = Path(target_path)
    last_exc = None

    for attempt in range(1, FILE_RETRY_COUNT + 1):
        try:
            os.replace(tmp_path, target_path)
            return target_path
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt < FILE_RETRY_COUNT:
                time.sleep(FILE_RETRY_SECONDS)

    fallback = target_path.with_name(
        f"{target_path.stem}__WPS_LOCKED_{time.strftime('%Y%m%d_%H%M%S')}"
        f"_{os.getpid()}{target_path.suffix}"
    )
    os.replace(tmp_path, fallback)
    print(
        f"\n[WARN] WPS 锁定输出文件 {target_path.name}; 已改存为 {fallback.name}"
        f" | last_error={last_exc!r}",
        flush=True,
    )
    return fallback


def safe_to_csv(df, target_path, index=False):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(
        f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    df.to_csv(tmp, index=index)
    return _replace_temp_with_lock_fallback(tmp, target_path)


def atomic_write_json(target_path, data):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(
        f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # State files live in a hidden/internal folder and should never be opened
    # by WPS. If even this cannot be committed, do NOT mark the dataset done.
    last_exc = None
    for attempt in range(1, FILE_RETRY_COUNT + 1):
        try:
            os.replace(tmp, target_path)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt < FILE_RETRY_COUNT:
                time.sleep(FILE_RETRY_SECONDS)

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    raise RuntimeError(f"无法写入完成标记 {target_path}: {last_exc!r}")


def read_data(csv_file, independent_vars, original_to_safe):
    df = pd.read_csv(csv_file)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_file} 至少需要 2 列")

    csv_x_cols = list(df.columns[:-1])
    target_name = str(df.columns[-1])

    if len(csv_x_cols) != len(independent_vars):
        raise ValueError(
            f"{csv_file}: CSV 自变量列数={len(csv_x_cols)}，Excel IndependentVars={independent_vars}"
        )

    if csv_x_cols != independent_vars:
        warnings.warn(
            f"{csv_file}: CSV 自变量名 {csv_x_cols} 与 Excel {independent_vars} 不一致，按 Excel 名称重命名。"
        )
        df = df.rename(columns=dict(zip(csv_x_cols, independent_vars)))

    X_original = df.iloc[:, :-1].astype(float).copy()
    X_original.columns = independent_vars
    y = df.iloc[:, -1].astype(float).to_numpy()

    # ONLY independent variables are put into X.
    X = pd.DataFrame(index=X_original.index)
    for name in independent_vars:
        X[original_to_safe[name]] = X_original[name].to_numpy(dtype=float)

    mask = np.isfinite(y)
    for col in X.columns:
        mask &= np.isfinite(X[col].to_numpy(dtype=float))

    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]

    if len(y) == 0:
        raise ValueError(f"{csv_file}: 没有有效数据")
    return X, y, target_name


def _model_kwargs(id_folder):
    run_root = id_folder / "pysr_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    return dict(
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        nested_constraints=nested_constraints,
        maxsize=nodemaxsize,
        complexity_of_operators={
            "cos": 0, "sin": 0, "^": 3, "exp": 3, "sqrt": 2, "log": 2,
        },
        model_selection="best",
        constraints={"^": (-1, 1)},
        parsimony=0.001,
        should_optimize_constants=True,
        batching=False,
        warm_start=True,
        procs=PYSR_PROCS_PER_PROCESS,
        verbosity=0,
        output_directory=str(run_root),
        run_id=RUN_ID,
    )


def build_new_model(id_folder):
    return PySRRegressor(**_model_kwargs(id_folder))


def state_dir(id_folder):
    d = id_folder / STATE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_marker_path(id_folder, dataset_index):
    return state_dir(id_folder) / f"dataset_{dataset_index}.done.json"


def load_done_marker(id_folder, dataset_index):
    p = done_marker_path(id_folder, dataset_index)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("run_id") != RUN_ID:
            return None
        top_path = Path(data["top_csv"])
        if not top_path.exists():
            return None
        return data
    except Exception:
        return None


def completed_prefix_indices(id_folder):
    """Only a contiguous prefix is safe to skip with sequential warm-start."""
    done = []
    for dataset_index in DATASET_INDICES:
        if load_done_marker(id_folder, dataset_index) is None:
            break
        done.append(dataset_index)
    return done


def remaining_dataset_count(row_id):
    id_folder = DATA_ROOT / row_id
    if not id_folder.is_dir():
        return len(DATASET_INDICES)
    return len(DATASET_INDICES) - len(completed_prefix_indices(id_folder))


def build_or_resume_model(id_folder, expected_safe_features, completed_prefix):
    run_root = id_folder / "pysr_runs"
    run_dir = run_root / RUN_ID

    # If PySR already created a run directory, try restoring its checkpoint.
    if run_dir.is_dir() and any(run_dir.iterdir()):
        try:
            # Current PySR API: restore from [output_directory]/[run_id].
            model = PySRRegressor.from_file(
                run_directory=str(run_dir),
                **_model_kwargs(id_folder),
            )

            actual_features = list(getattr(model, "feature_names_in_", []))
            if actual_features and actual_features != list(expected_safe_features):
                raise RuntimeError(
                    f"checkpoint features={actual_features}, expected={list(expected_safe_features)}"
                )

            return model, True

        except Exception as exc:
            # If some datasets are already marked done, starting a brand-new
            # model and skipping them would break warm-start continuity.
            if completed_prefix:
                raise RuntimeError(
                    f"{id_folder.name}: 已有完成标记 {completed_prefix}，"
                    f"但无法恢复 PySR checkpoint，因此拒绝重跑/跳过造成状态错位。"
                    f"原错误: {exc!r}"
                ) from exc

            # No committed dataset yet: quarantine broken/incompatible run and
            # safely start over without risking duplicate completed work.
            broken = run_root / f"{RUN_ID}__BROKEN_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            try:
                os.replace(run_dir, broken)
                print(
                    f"\n[WARN][{id_folder.name}] 无法恢复未完成 checkpoint，已移到 {broken.name} 后重新开始。",
                    flush=True,
                )
            except Exception:
                pass

    return build_new_model(id_folder), False


def select_top_pysr_models(equations_df):
    filtered = equations_df[
        (equations_df["complexity"] >= min_complexity)
        & (equations_df["complexity"] <= max_complexity)
    ].copy()
    if filtered.empty:
        filtered = equations_df.copy()
    return filtered.nsmallest(top_n, "loss").reset_index(drop=True)


def replace_floats_toConstant(expr):
    number_to_symbol, param_list, const_list = {}, [], []
    counter = 0

    def rec(node):
        nonlocal counter
        if node.is_Number:
            if node not in number_to_symbol:
                p = sp.Symbol(f"p{counter}")
                number_to_symbol[node] = p
                param_list.append(p)
                const_list.append(node)
                counter += 1
            return number_to_symbol[node]
        if node.args:
            return node.func(*[rec(arg) for arg in node.args])
        return node

    return rec(expr), param_list, const_list


def save_regression_result(
    model, id_folder, dataset_index, csv_file, target_name,
    independent_vars, parameter_names, parameter_values,
    fixed_constants, safe_to_original,
):
    equations = model.equations_.copy()

    all_path = None
    if SAVE_ALL_EQUATIONS:
        all_path = safe_to_csv(
            equations,
            id_folder / f"{dataset_index}_pysr_all.csv",
            index=False,
        )

    top = select_top_pysr_models(equations)
    out_rows = []

    for _, row in top.iterrows():
        sympy_expr = row.get("sympy_format", row["equation"])
        if not isinstance(sympy_expr, sp.Basic):
            sympy_expr = sp.sympify(sympy_expr)

        restored_expr = restore_expression_names(sympy_expr, safe_to_original)
        para_eq, para_list, const_list = replace_floats_toConstant(restored_expr)

        out_rows.append({
            "ID": id_folder.name,
            "dataset": dataset_index,
            "orgfile": csv_file.name,
            "target": target_name,
            "IndependentVars": json.dumps(independent_vars, ensure_ascii=False),
            # These remain metadata only; they are NOT PySR input features.
            "ParameterNames": json.dumps(parameter_names, ensure_ascii=False),
            "ParameterValues": json.dumps([float(x) for x in parameter_values], ensure_ascii=False),
            "FixedConstantValues": json.dumps(fixed_constants, ensure_ascii=False),
            "PySRFeatures": json.dumps(list(model.feature_names_in_), ensure_ascii=False),
            "feature_name_map": json.dumps(safe_to_original, ensure_ascii=False),
            "complexity": row["complexity"],
            "loss": row["loss"],
            "score": row.get("score", np.nan),
            "equation": row["equation"],
            "sympy_format": str(sympy_expr),
            "restored_equation": str(restored_expr),
            "orgpara_list": str(const_list),
            "para_eq": str(para_eq),
            "para_list": str(para_list),
        })

    top_df = pd.DataFrame(out_rows)
    top_path = safe_to_csv(
        top_df,
        id_folder / f"{dataset_index}_pysr_top{top_n}.csv",
        index=False,
    )
    return top_df, top_path, all_path


def quiet_model_fit(model, X, y):
    """Suppress normal PySR/Julia stdout/stderr; exceptions still propagate."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return model.fit(X, y)


def mark_dataset_done(id_folder, dataset_index, top_path, all_path, csv_file):
    marker = {
        "run_id": RUN_ID,
        "dataset": int(dataset_index),
        "source_csv": str(csv_file),
        "top_csv": str(top_path),
        "all_csv": str(all_path) if all_path is not None else None,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(done_marker_path(id_folder, dataset_index), marker)


def rebuild_summary_from_markers(id_folder):
    parts = []
    for dataset_index in DATASET_INDICES:
        marker = load_done_marker(id_folder, dataset_index)
        if marker is None:
            continue
        top_path = Path(marker["top_csv"])
        parts.append(pd.read_csv(top_path))

    if parts:
        summary = pd.concat(parts, ignore_index=True)
        safe_to_csv(
            summary,
            id_folder / "pysr_warm_0_7_summary.csv",
            index=False,
        )


def process_one_id(row_id, info, worker_id=None, progress_queue=None, worker_state=None):
    id_folder = DATA_ROOT / row_id

    if not id_folder.is_dir():
        msg = f"找不到目录: {id_folder}"
        if REQUIRE_ALL_FILES:
            raise FileNotFoundError(msg)
        return

    independent_vars = info["IndependentVars"]
    parameter_names = info["ParameterNames"]
    fixed_constants = info["FixedConstants"]
    para_values = info["ParaValues"]

    original_to_safe, safe_to_original = make_feature_maps(independent_vars)
    expected_safe_features = [original_to_safe[x] for x in independent_vars]

    # Skip only a contiguous, committed prefix: 0,1,...,k.
    completed_prefix = completed_prefix_indices(id_folder)
    completed_set = set(completed_prefix)

    if len(completed_prefix) == len(DATASET_INDICES):
        rebuild_summary_from_markers(id_folder)
        return

    model, resumed = build_or_resume_model(
        id_folder=id_folder,
        expected_safe_features=expected_safe_features,
        completed_prefix=completed_prefix,
    )

    # After loading a checkpoint, feature names should correspond ONLY to xv_*.
    actual_features = list(getattr(model, "feature_names_in_", []))
    if actual_features and actual_features != expected_safe_features:
        raise RuntimeError(
            f"{row_id}: PySR feature mismatch: actual={actual_features}, expected={expected_safe_features}"
        )

    for dataset_index in DATASET_INDICES:
        if dataset_index in completed_set:
            continue

        csv_file = id_folder / f"{dataset_index}.csv"
        if not csv_file.exists():
            msg = f"{row_id}: 缺少 {csv_file.name}"
            if REQUIRE_ALL_FILES:
                raise FileNotFoundError(msg)
            continue

        parameter_values = para_values[dataset_index]

        X, y, target_name = read_data(
            csv_file=csv_file,
            independent_vars=independent_vars,
            original_to_safe=original_to_safe,
        )

        csv_start = time.perf_counter()
        quiet_model_fit(model, X, y)
        csv_elapsed = time.perf_counter() - csv_start

        top_df, top_path, all_path = save_regression_result(
            model=model,
            id_folder=id_folder,
            dataset_index=dataset_index,
            csv_file=csv_file,
            target_name=target_name,
            independent_vars=independent_vars,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            fixed_constants=fixed_constants,
            safe_to_original=safe_to_original,
        )

        # IMPORTANT: only mark done AFTER PySR fit + result files completed.
        # On restart, this marker is the authority for skipping work.
        mark_dataset_done(
            id_folder=id_folder,
            dataset_index=dataset_index,
            top_path=top_path,
            all_path=all_path,
            csv_file=csv_file,
        )

        if progress_queue is not None and worker_state is not None:
            worker_state["completed_units"] += 1

            elapsed_worker = time.perf_counter() - worker_state["worker_start"]
            done_units = worker_state["completed_units"]
            total_units = worker_state["total_units"]

            avg_unit_time = elapsed_worker / done_units if done_units > 0 else None
            remaining_units = max(0, total_units - done_units)
            worker_eta_seconds = (
                avg_unit_time * remaining_units
                if avg_unit_time is not None
                else None
            )

            progress_queue.put({
                "type": "csv_done",
                "worker_id": worker_id,
                "row_id": row_id,
                "dataset_index": dataset_index,
                "csv_elapsed": csv_elapsed,
                "worker_completed_units": done_units,
                "worker_total_units": total_units,
                "worker_eta_seconds": worker_eta_seconds,
            })

    rebuild_summary_from_markers(id_folder)


def format_duration(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "估算中"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_clock_time(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def format_worker_progress(worker_completed_units, worker_total_units):
    parts = []
    for worker_id in sorted(worker_total_units):
        total = worker_total_units[worker_id]
        done = min(worker_completed_units.get(worker_id, 0), total)
        pct = 100.0 * done / total if total > 0 else 100.0
        parts.append(f"W{worker_id} {done}/{total}({pct:.0f}%)")
    return " | ".join(parts)


def process_task_queue(worker_id, task_queue, progress_queue):
    """Continuously pull one ID-folder task at a time from the shared queue."""
    completed, failed = [], []

    while True:
        task = task_queue.get()
        if task is None:
            break

        row_id, info = task
        dir_start = time.perf_counter()
        remaining_units = remaining_dataset_count(row_id)
        worker_state = {
            # Reset for each folder so the per-worker display means
            # "progress inside the current subfolder".
            "worker_start": dir_start,
            "completed_units": 0,
            "total_units": remaining_units,
        }

        progress_queue.put({
            "type": "directory_start",
            "worker_id": worker_id,
            "row_id": row_id,
            "remaining_units": remaining_units,
        })

        try:
            process_one_id(
                row_id,
                info,
                worker_id=worker_id,
                progress_queue=progress_queue,
                worker_state=worker_state,
            )
            completed.append(row_id)
            status = "OK"
            error_text = None

        except Exception as exc:
            failed.append((row_id, repr(exc)))
            status = "FAILED"
            error_text = repr(exc)
            print(f"\n[ERROR][worker {worker_id}][{row_id}] {exc}", flush=True)

        dir_elapsed = time.perf_counter() - dir_start
        progress_queue.put({
            "type": "directory_done",
            "worker_id": worker_id,
            "row_id": row_id,
            "status": status,
            "error": error_text,
            "dir_elapsed": dir_elapsed,
            "folder_completed_units": worker_state["completed_units"],
            "folder_total_units": worker_state["total_units"],
        })

    return {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "completed": completed,
        "failed": failed,
    }


ID_NUMBER_RE = re.compile(r"^([^0-9]*)([0-9]+)$")


def parse_task_range(value):
    """Parse TASK_RANGE into (prefix, start_number, end_number, raw_text)."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if "-" in text:
        left, right = [part.strip() for part in text.split("-", 1)]
    else:
        left = right = text

    m1 = ID_NUMBER_RE.fullmatch(left)
    m2 = ID_NUMBER_RE.fullmatch(right)
    if not m1 or not m2:
        raise ValueError(
            f"TASK_RANGE 格式错误: {value!r}. "
            f"请使用例如 'P10-P20'、'P10'，或 None。"
        )

    prefix1, start_text = m1.groups()
    prefix2, end_text = m2.groups()
    if prefix1 != prefix2:
        raise ValueError(
            f"TASK_RANGE 两端前缀必须相同: {left!r} vs {right!r}"
        )

    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(
            f"TASK_RANGE 起点不能大于终点: {left!r} > {right!r}"
        )

    return prefix1, start, end, text


def row_id_in_task_range(row_id, parsed_range):
    if parsed_range is None:
        return True

    prefix, start, end, _ = parsed_range
    m = ID_NUMBER_RE.fullmatch(str(row_id).strip())
    if not m:
        return False

    row_prefix, number_text = m.groups()
    return row_prefix == prefix and start <= int(number_text) <= end


def filter_input_dirs_by_task_selection(input_dirs, all_metadata):
    """Select actual subfolders first, then validate only the selected tasks."""
    parsed_range = parse_task_range(TASK_RANGE)
    selected_dirs = list(input_dirs)

    if parsed_range is not None:
        selected_dirs = [
            path for path in selected_dirs
            if row_id_in_task_range(path.name, parsed_range)
        ]

        prefix, start, end, raw = parsed_range
        endpoint_numbers = {start, end}
        found_endpoint_numbers = set()
        for path in input_dirs:
            m = ID_NUMBER_RE.fullmatch(path.name)
            if m and m.group(1) == prefix and int(m.group(2)) in endpoint_numbers:
                found_endpoint_numbers.add(int(m.group(2)))
        missing_endpoints = sorted(endpoint_numbers - found_endpoint_numbers)
        if missing_endpoints:
            print(
                f"[WARN] TASK_RANGE={raw!r} 的边界编号中，以下编号没有对应的数据子文件夹: "
                f"{missing_endpoints}",
                flush=True,
            )

    if TARGET_ROW_IDS is not None:
        requested = [str(x).strip() for x in TARGET_ROW_IDS]
        missing_metadata_ids = [row_id for row_id in requested if row_id not in all_metadata]
        if missing_metadata_ids:
            raise KeyError(f"Excel 中缺少目标 ID: {missing_metadata_ids}")

        folder_names = {path.name for path in input_dirs}
        missing_folder_ids = [row_id for row_id in requested if row_id not in folder_names]
        if missing_folder_ids:
            raise FileNotFoundError(
                f"DATA_ROOT 中缺少 TARGET_ROW_IDS 对应子文件夹: {missing_folder_ids}"
            )

        requested_set = set(requested)
        selected_dirs = [path for path in selected_dirs if path.name in requested_set]

    if not selected_dirs:
        raise ValueError(
            f"任务筛选后没有可执行的子文件夹。"
            f" TASK_RANGE={TASK_RANGE!r}, TARGET_ROW_IDS={TARGET_ROW_IDS!r}"
        )

    return selected_dirs


def row_id_sort_key(row_id):
    """Natural sort: P2 before P10, while still supporting non-numeric names."""
    m = ID_NUMBER_RE.fullmatch(str(row_id).strip())
    if m:
        return (m.group(1), int(m.group(2)), str(row_id))
    return (str(row_id), -1, str(row_id))


def print_run_configuration(metadata, worker_count, total_units, skipped_units):
    """Print all important settings before any regression worker is started."""
    selected_ids = list(metadata.keys())
    dataset_indices = list(DATASET_INDICES)

    print("\n" + "=" * 92, flush=True)
    print("PySR RUN CONFIGURATION -- PLEASE CHECK BEFORE EXECUTION", flush=True)
    print("=" * 92, flush=True)
    print(f"ROOT                     : {ROOT}", flush=True)
    print(f"EXCEL_FILE               : {EXCEL_FILE}", flush=True)
    print(f"DATA_ROOT                : {DATA_ROOT}", flush=True)
    print(f"SHEET_NAME               : {SHEET_NAME}", flush=True)
    print(f"TASK_RANGE               : {TASK_RANGE!r}", flush=True)
    print(f"TARGET_ROW_IDS            : {TARGET_ROW_IDS!r}", flush=True)
    print(f"Selected folders ({len(selected_ids):>3})   : {', '.join(selected_ids)}", flush=True)
    print(f"DATASET_INDICES           : {dataset_indices}", flush=True)
    print(f"Remaining CSV fits        : {total_units}", flush=True)
    print(f"Already-done CSV skipped  : {skipped_units}", flush=True)
    print("-" * 92, flush=True)
    print(f"OUTER_PROCESSES           : {OUTER_PROCESSES} (actual workers={worker_count})", flush=True)
    print(f"PYSR_PROCS_PER_PROCESS    : {PYSR_PROCS_PER_PROCESS}", flush=True)
    print(f"Potential process budget  : {worker_count * PYSR_PROCS_PER_PROCESS}", flush=True)
    print(f"niterations               : {niterations}", flush=True)
    print(f"populations               : {populations}", flush=True)
    print(f"population_size           : {population_size}", flush=True)
    print(f"nodemaxsize               : {nodemaxsize}", flush=True)
    print(f"complexity range          : [{min_complexity}, {max_complexity}]", flush=True)
    print(f"top_n                     : {top_n}", flush=True)
    print(f"RUN_ID                    : {RUN_ID}", flush=True)
    print(f"binary_operators          : {binary_operators}", flush=True)
    print(f"unary_operators           : {unary_operators}", flush=True)
    print(f"REQUIRE_ALL_FILES         : {REQUIRE_ALL_FILES}", flush=True)
    print(f"SAVE_ALL_EQUATIONS        : {SAVE_ALL_EQUATIONS}", flush=True)
    print(f"warm_start                : True", flush=True)
    print(f"model_selection           : best", flush=True)
    print(f"parsimony                 : 0.001", flush=True)
    print("=" * 92 + "\n", flush=True)


def discover_input_metadata(all_metadata):
    input_dirs = [
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not input_dirs:
        raise FileNotFoundError(f"{DATA_ROOT} 中没有可处理的数据集目录")

    # IMPORTANT: select the requested folder range BEFORE metadata validation.
    # Therefore folders outside TASK_RANGE cannot break this run.
    input_dirs = filter_input_dirs_by_task_selection(input_dirs, all_metadata)
    input_dirs.sort(key=lambda path: row_id_sort_key(path.name))

    missing_metadata = [
        path.name for path in input_dirs
        if path.name not in all_metadata
    ]
    if missing_metadata:
        raise KeyError(f"Excel 中缺少本次任务对应的 ID: {missing_metadata}")

    return {path.name: all_metadata[path.name] for path in input_dirs}


def main():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(EXCEL_FILE)
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    all_metadata = load_metadata()
    metadata = discover_input_metadata(all_metadata)

    # One subfolder = one queue task.  Put heavier/resumed folders first to
    # reduce the long tail while workers still dynamically pull from one queue.
    tasks = list(metadata.items())
    tasks.sort(key=lambda item: remaining_dataset_count(item[0]), reverse=True)

    worker_count = min(OUTER_PROCESSES, len(tasks))
    if worker_count <= 0:
        raise ValueError("没有可执行的子文件夹任务")

    total_units = sum(remaining_dataset_count(row_id) for row_id, _ in tasks)
    skipped_units = len(metadata) * len(DATASET_INDICES) - total_units

    # IMPORTANT: print configuration BEFORE any regression worker is started.
    print_run_configuration(
        metadata=metadata,
        worker_count=worker_count,
        total_units=total_units,
        skipped_units=skipped_units,
    )

    if skipped_units:
        print(f"已完成并跳过 {skipped_units} 个 CSV 拟合；继续剩余任务。", flush=True)

    print(
        f"队列模式: {len(tasks)} 个子文件夹任务 | {worker_count} 个 worker | "
        f"剩余 CSV 拟合={total_units}",
        flush=True,
    )
    print("剩余时间估算中...", end="", flush=True)

    ctx = mp.get_context("spawn")
    results = []
    overall_start_perf = time.perf_counter()

    completed_dirs = 0
    failed_dirs = 0
    completed_units = 0
    observed_csv_times = []
    worker_current = {wid: None for wid in range(worker_count)}
    worker_folder_progress = {wid: (0, 0) for wid in range(worker_count)}

    PROVISIONAL_ETA_AFTER = 30.0
    last_status_refresh = 0.0

    def render_status():
        elapsed = time.perf_counter() - overall_start_perf
        remaining_units = max(0, total_units - completed_units)

        avg_csv = (
            float(np.mean(observed_csv_times))
            if observed_csv_times else None
        )

        if remaining_units == 0:
            eta_seconds = 0.0
        elif avg_csv is not None:
            # Dynamic queue keeps workers balanced, so divide the remaining
            # work by the number of active workers rather than using fixed chunks.
            active_workers = max(1, sum(v is not None for v in worker_current.values()))
            if active_workers == 1 and completed_dirs < len(tasks):
                # At startup task_start messages may not all have arrived yet.
                active_workers = worker_count
            eta_seconds = avg_csv * remaining_units / active_workers
        elif elapsed >= PROVISIONAL_ETA_AFTER:
            # Before the first CSV finishes, only show a deliberately rough ETA.
            eta_seconds = elapsed * remaining_units / max(1, worker_count)
        else:
            eta_seconds = None

        finish_text = (
            format_clock_time(time.time() + eta_seconds)
            if eta_seconds is not None else "估算中"
        )

        worker_parts = []
        for wid in range(worker_count):
            row_id = worker_current[wid]
            done_u, total_u = worker_folder_progress[wid]
            if row_id is None:
                worker_parts.append(f"W{wid}:idle")
            elif total_u > 0:
                worker_parts.append(f"W{wid}:{row_id} {done_u}/{total_u}")
            else:
                worker_parts.append(f"W{wid}:{row_id}")

        status_line = (
            f"\r剩余≈{format_duration(eta_seconds)}"
            f" | ETA={finish_text}"
            f" | folders={completed_dirs}/{len(tasks)}"
            f" | CSV={completed_units}/{total_units}"
            f" | " + " | ".join(worker_parts)
        )
        if failed_dirs:
            status_line += f" | errors={failed_dirs}"
        print(status_line.ljust(200), end="", flush=True)

    with mp.Manager() as manager:
        task_queue = manager.Queue()
        progress_queue = manager.Queue()

        for task in tasks:
            task_queue.put(task)
        # One sentinel per worker cleanly terminates the worker loops.
        for _ in range(worker_count):
            task_queue.put(None)

        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
        ) as executor:
            futures = {
                executor.submit(
                    process_task_queue,
                    worker_id,
                    task_queue,
                    progress_queue,
                ): worker_id
                for worker_id in range(worker_count)
            }

            pending = set(futures)

            while pending:
                got_message = False
                while True:
                    try:
                        msg = progress_queue.get_nowait()
                    except Exception:
                        break

                    got_message = True
                    msg_type = msg.get("type")
                    wid = msg.get("worker_id")

                    if msg_type == "directory_start":
                        worker_current[wid] = msg["row_id"]
                        worker_folder_progress[wid] = (0, msg.get("remaining_units", 0))

                    elif msg_type == "csv_done":
                        completed_units += 1
                        done_u = msg.get("worker_completed_units", 0)
                        total_u = msg.get("worker_total_units", 0)
                        worker_folder_progress[wid] = (done_u, total_u)

                        csv_t = msg.get("csv_elapsed")
                        if csv_t is not None and np.isfinite(csv_t) and csv_t > 0:
                            observed_csv_times.append(float(csv_t))

                    elif msg_type == "directory_done":
                        completed_dirs += 1
                        if msg["status"] == "FAILED":
                            failed_dirs += 1
                        worker_current[wid] = None
                        worker_folder_progress[wid] = (0, 0)
                    else:
                        continue

                    render_status()

                done, pending = wait(
                    pending,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )

                now_elapsed = time.perf_counter() - overall_start_perf
                if not got_message and now_elapsed - last_status_refresh >= 1.0:
                    render_status()
                    last_status_refresh = now_elapsed

                for future in done:
                    worker_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except BaseException as exc:
                        print(
                            f"\n[PROCESS ERROR][worker {worker_id}] "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    worker_current[worker_id] = None
                    worker_folder_progress[worker_id] = (0, 0)

            # Drain messages that arrived just before the final workers exited.
            while True:
                try:
                    msg = progress_queue.get_nowait()
                except Exception:
                    break

                msg_type = msg.get("type")
                wid = msg.get("worker_id")
                if msg_type == "csv_done":
                    completed_units += 1
                    csv_t = msg.get("csv_elapsed")
                    if csv_t is not None and np.isfinite(csv_t) and csv_t > 0:
                        observed_csv_times.append(float(csv_t))
                elif msg_type == "directory_done":
                    completed_dirs += 1
                    if msg["status"] == "FAILED":
                        failed_dirs += 1
                    if wid is not None:
                        worker_current[wid] = None

    total_elapsed = time.perf_counter() - overall_start_perf
    total_ok = sum(len(result["completed"]) for result in results)
    total_failed = sum(len(result["failed"]) for result in results)

    print(
        (
            f"\r完成 | 总耗时={format_duration(total_elapsed)}"
            f" | 成功={total_ok}"
            f" | 失败={total_failed}"
            f" | 启动时已跳过CSV={skipped_units}"
        ).ljust(200),
        flush=True,
    )

    if total_failed:
        for result in sorted(results, key=lambda x: x["worker_id"]):
            for row_id, err in result["failed"]:
                print(
                    f"[ERROR][worker {result['worker_id']}][{row_id}] {err}",
                    flush=True,
                )


if __name__ == "__main__":
    mp.freeze_support()
    main()
    print("\n所有任务完成。", flush=True)
