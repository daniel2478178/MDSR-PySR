# -*- coding: utf-8 -*-
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import re
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

DEFAULT_SHEET_NAME = "Sampling design"
DEFAULT_DATASET_INDICES = tuple(range(8))
DEFAULT_ITERATIONS = 400
populations = 20
population_size = 40
nodemaxsize = 50
min_complexity = 7
max_complexity = 40
DEFAULT_TOP_N = 10
DEFAULT_OUTER_PROCESSES = 4
DEFAULT_PYSR_PROCESSES = 3


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

DEFAULT_SUMMARY_NAME = "pysr_warm_0_7_summary.csv"
DEFAULT_RUN_ID = "warm_0_7_with_params"


@dataclass(frozen=True)
class RunConfig:
    workbook: Path
    data_root: Path
    sheet_name: str
    dataset_indices: tuple[int, ...]
    target_row_ids: tuple[str, ...] | None
    iterations: int
    outer_processes: int
    pysr_processes: int
    top_n: int
    run_id: str
    summary_name: str
    require_all_files: bool
    save_all_equations: bool


def parse_dataset_indices(value):
    indices = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            indices.extend(range(start, end + 1))
        elif part:
            indices.append(int(part))
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("datasets must be unique non-negative indices, e.g. 0-7")
    return tuple(indices)


def build_parser():
    parser = argparse.ArgumentParser(description="Run PySR discovery with physical parameters and constants as features")
    parser.add_argument("workbook", type=Path, help="Benchmark metadata workbook")
    parser.add_argument("data_root", type=Path, help="Root containing one dataset directory per ID")
    parser.add_argument("--sheet", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--datasets", type=parse_dataset_indices, default=DEFAULT_DATASET_INDICES)
    parser.add_argument("--ids", nargs="+", help="Optional subset of benchmark IDs")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--outer-processes", type=int, default=DEFAULT_OUTER_PROCESSES)
    parser.add_argument("--pysr-processes", type=int, default=DEFAULT_PYSR_PROCESSES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--allow-missing-files", action="store_true")
    parser.add_argument("--no-save-all-equations", action="store_true")
    return parser


def build_config(args):
    if min(args.iterations, args.outer_processes, args.pysr_processes, args.top_n) <= 0:
        raise ValueError("iterations, process counts, and top-n must be positive")
    return RunConfig(
        workbook=args.workbook.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        sheet_name=args.sheet,
        dataset_indices=tuple(args.datasets),
        target_row_ids=tuple(args.ids) if args.ids else None,
        iterations=args.iterations,
        outer_processes=args.outer_processes,
        pysr_processes=args.pysr_processes,
        top_n=args.top_n,
        run_id=args.run_id,
        summary_name=args.summary_name,
        require_all_files=not args.allow_missing_files,
        save_all_equations=not args.no_save_all_equations,
    )

def _alpha_index(i):
    letters = "abcdefghijklmnopqrstuvwxyz"
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = letters[r] + s
    return s

def make_feature_maps(independent_vars, parameter_names, fixed_constants):
    original_to_safe, safe_to_original = {}, {}
    for i, name in enumerate(independent_vars):
        safe = f"xv_{_alpha_index(i)}"
        original_to_safe[name] = safe
        safe_to_original[safe] = name
    for i, name in enumerate(parameter_names):
        safe = f"pv_{_alpha_index(i)}"
        original_to_safe[name] = safe
        safe_to_original[safe] = name
    for i, name in enumerate(fixed_constants.keys()):
        safe = f"cv_{_alpha_index(i)}"
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
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|in)\s*\[\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)
NUMBER_PREFIX_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+.*)?$"
)

def parse_independent_vars(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r"[,;]", str(value)) if x.strip()]

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
        match = NUMBER_PREFIX_RE.fullmatch(val.strip())
        if match is None:
            raise ValueError(f"Cannot parse FixedConstantValues item {part!r}")
        out[name.strip()] = float(match.group(1))
    return out

def parse_para_value(value, dataset_indices):
    if pd.isna(value):
        raise ValueError("paraValue 为空")
    data = json.loads(str(value))
    required_length = max(dataset_indices) + 1
    if len(data) < required_length:
        raise ValueError(
            f"paraValue 只有 {len(data)} 组，数据集索引 "
            f"{max(dataset_indices)} 至少需要 {required_length} 组"
        )
    return data

def load_metadata(config):
    df = pd.read_excel(config.workbook, sheet_name=config.sheet_name)
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
        para_values = parse_para_value(row["paraValue"], config.dataset_indices)

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

def read_data(csv_file, independent_vars, parameter_names, parameter_values,
              fixed_constants, original_to_safe):
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

    X = pd.DataFrame(index=X_original.index)
    for name in independent_vars:
        X[original_to_safe[name]] = X_original[name].to_numpy(dtype=float)
    for name, value in zip(parameter_names, parameter_values):
        X[original_to_safe[name]] = float(value)
    for name, value in fixed_constants.items():
        X[original_to_safe[name]] = float(value)

    mask = np.isfinite(y)
    for col in X.columns:
        mask &= np.isfinite(X[col].to_numpy(dtype=float))
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]

    if len(y) == 0:
        raise ValueError(f"{csv_file}: 没有有效数据")
    return X, y, target_name

def build_model(id_folder, config):
    from pysr import PySRRegressor

    run_root = id_folder / "pysr_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    return PySRRegressor(
        niterations=config.iterations,
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
        procs=config.pysr_processes,
        verbosity=0,
        output_directory=str(run_root),
        run_id=config.run_id,
    )

def select_top_pysr_models(equations_df, top_n):
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

def save_regression_result(model, id_folder, dataset_index, csv_file, target_name,
                           independent_vars, parameter_names, parameter_values,
                           fixed_constants, safe_to_original, config):
    equations = model.equations_.copy()

    if config.save_all_equations:
        equations.to_csv(id_folder / f"{dataset_index}_pysr_all.csv", index=False)

    top = select_top_pysr_models(equations, config.top_n)
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
    top_df.to_csv(id_folder / f"{dataset_index}_pysr_top{config.top_n}.csv", index=False)
    return top_df


def quiet_model_fit(model, X, y):
    """
    Suppress normal PySR/Julia stdout/stderr during fit.
    Exceptions still propagate and will be reported by the worker/main process.
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return model.fit(X, y)


def resolve_id_folder(row_id, config):
    return config.data_root / row_id


def process_one_id(row_id, info, config, worker_id=None, progress_queue=None,
                   worker_state=None):
    id_folder = resolve_id_folder(row_id, config)

    if not id_folder.is_dir():
        msg = f"找不到目录: {id_folder}"
        if config.require_all_files:
            raise FileNotFoundError(msg)
        return

    summary_file = id_folder / config.summary_name
    if summary_file.is_file() and summary_file.stat().st_size > 0:
        skipped_units = len(config.dataset_indices)
        if worker_state is not None:
            worker_state["completed_units"] += skipped_units
        return "SKIPPED"

    independent_vars = info["IndependentVars"]
    parameter_names = info["ParameterNames"]
    fixed_constants = info["FixedConstants"]
    para_values = info["ParaValues"]

    original_to_safe, safe_to_original = make_feature_maps(
        independent_vars,
        parameter_names,
        fixed_constants,
    )

    # One model per folder; warm_start is shared only inside this folder.
    model = build_model(id_folder, config)
    summary_parts = []

    for dataset_index in config.dataset_indices:
        csv_file = id_folder / f"{dataset_index}.csv"

        if not csv_file.exists():
            msg = f"{row_id}: 缺少 {csv_file.name}"
            if config.require_all_files:
                raise FileNotFoundError(msg)
            continue

        parameter_values = para_values[dataset_index]

        X, y, target_name = read_data(
            csv_file=csv_file,
            independent_vars=independent_vars,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            fixed_constants=fixed_constants,
            original_to_safe=original_to_safe,
        )

        # Suppress ordinary PySR output.
        # Exceptions still propagate.
        csv_start = time.perf_counter()
        quiet_model_fit(model, X, y)
        csv_elapsed = time.perf_counter() - csv_start

        # Report progress after EACH CSV, so ETA can appear quickly.
        if progress_queue is not None and worker_state is not None:
            worker_state["completed_units"] += 1
            worker_state["observed_unit_times"].append(csv_elapsed)

            elapsed_worker = (
                time.perf_counter()
                - worker_state["worker_start"]
            )

            done_units = worker_state["completed_units"]
            total_units = worker_state["total_units"]

            # Use actual wall-clock average per completed CSV for this worker.
            avg_unit_time = (
                elapsed_worker / done_units
                if done_units > 0
                else None
            )

            remaining_units = max(
                0,
                total_units - done_units,
            )

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

        top_df = save_regression_result(
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
            config=config,
        )

        summary_parts.append(top_df)

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(
            id_folder / config.summary_name,
            index=False,
        )


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
    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp),
    )


def process_directory_task(row_id, info, config, progress_queue):
    """Process one ID directory as one queue task.

    ProcessPoolExecutor keeps four worker processes alive.  Whenever a worker
    finishes this function, it automatically takes the next unfinished ID from
    the executor's task queue.
    """
    worker_id = os.getpid()
    worker_start = time.perf_counter()
    worker_state = {
        "worker_start": worker_start,
        "completed_units": 0,
        "total_units": len(config.dataset_indices),
        "observed_unit_times": [],
    }
    progress_queue.put({
        "type": "directory_start",
        "worker_id": worker_id,
        "row_id": row_id,
    })
    try:
        process_status = process_one_id(
            row_id,
            info,
            config,
            worker_id=worker_id,
            progress_queue=progress_queue,
            worker_state=worker_state,
        )
        status = process_status or "OK"
        error_text = None
    except Exception as exc:
        status = "FAILED"
        error_text = repr(exc)
        print(
            f"\n[ERROR][process {worker_id}][{row_id}] {exc}",
            flush=True,
        )

    dir_elapsed = time.perf_counter() - worker_start
    progress_queue.put({
        "type": "directory_done",
        "worker_id": worker_id,
        "row_id": row_id,
        "status": status,
        "error": error_text,
        "dir_elapsed": dir_elapsed,
        "task_completed_units": worker_state["completed_units"],
    })
    return {
        "worker_id": worker_id,
        "pid": worker_id,
        "row_id": row_id,
        "status": status,
        "error": error_text,
    }

def is_directory_complete(row_id, config):
    """Use the existing non-empty summary as the directory completion marker."""
    summary_file = resolve_id_folder(row_id, config) / config.summary_name
    return summary_file.is_file() and summary_file.stat().st_size > 0


def collect_unfinished_tasks(metadata, config):
    """Return one queue item per ID whose summary has not been completed."""
    unfinished = []
    completed_ids = []
    for row_id, info in metadata.items():
        if is_directory_complete(row_id, config):
            completed_ids.append(row_id)
        else:
            unfinished.append((row_id, info))
    return unfinished, completed_ids

def main(argv=None):
    """Run unfinished ID directories through one shared four-process queue."""
    config = build_config(build_parser().parse_args(argv))
    if not config.workbook.exists():
        raise FileNotFoundError(config.workbook)
    if not config.data_root.is_dir():
        raise FileNotFoundError(config.data_root)

    print(
        f"配置 | workbook={config.workbook} | data_root={config.data_root} "
        f"| datasets={config.dataset_indices} | iterations={config.iterations} "
        f"| outer_processes={config.outer_processes} "
        f"| pysr_processes={config.pysr_processes} | run_id={config.run_id}",
        flush=True,
    )

    metadata = load_metadata(config)
    if config.target_row_ids is not None:
        missing = [row_id for row_id in config.target_row_ids if row_id not in metadata]
        if missing:
            raise KeyError(f"Excel 中缺少目标 ID: {missing}")
        metadata = {row_id: metadata[row_id] for row_id in config.target_row_ids}
    tasks, already_completed = collect_unfinished_tasks(metadata, config)
    total_dirs = len(tasks)
    active_process_count = min(config.outer_processes, total_dirs)

    print(
        f"全部目录={len(metadata)} | 已完成并跳过={len(already_completed)} "
        f"| 当前未完成={total_dirs} | 动态进程数={active_process_count}",
        flush=True,
    )
    if not tasks:
        print("所有目录均已完成，无需运行。", flush=True)
        return

    ctx = mp.get_context("spawn")
    overall_start = time.perf_counter()
    total_units = total_dirs * len(config.dataset_indices)
    completed_units = 0
    completed_dirs = 0
    failed_dirs = 0
    observed_csv_times = []
    task_reported_units = {}
    active_tasks = {}
    seen_done_tasks = set()
    results = []

    def consume_message(msg):
        nonlocal completed_units, completed_dirs, failed_dirs
        msg_type = msg.get("type")
        row_id = msg.get("row_id")
        worker_id = msg.get("worker_id")

        if msg_type == "directory_start":
            active_tasks[worker_id] = (row_id, None)
            return

        if msg_type == "csv_done":
            current = task_reported_units.get(row_id, 0)
            if current < len(config.dataset_indices):
                task_reported_units[row_id] = current + 1
                completed_units += 1
            active_tasks[worker_id] = (row_id, msg.get("dataset_index"))
            csv_elapsed = msg.get("csv_elapsed")
            if (
                csv_elapsed is not None
                and np.isfinite(csv_elapsed)
                and csv_elapsed > 0
            ):
                observed_csv_times.append(float(csv_elapsed))
            return

        if msg_type == "directory_done":
            if row_id not in seen_done_tasks:
                seen_done_tasks.add(row_id)
                completed_dirs += 1
                if msg.get("status") == "FAILED":
                    failed_dirs += 1

                # A skipped or failed directory may emit fewer than eight
                # csv_done messages. Mark the unattempted units as resolved so
                # the progress counter still reaches the total.
                reported = task_reported_units.get(row_id, 0)
                completed_units += max(
                    0,
                    len(config.dataset_indices) - reported,
                )
                task_reported_units[row_id] = len(config.dataset_indices)

            if active_tasks.get(worker_id, (None, None))[0] == row_id:
                active_tasks.pop(worker_id, None)

    def drain_progress(progress_queue):
        while True:
            try:
                message = progress_queue.get_nowait()
            except Exception:
                return
            consume_message(message)

    def show_status():
        elapsed = time.perf_counter() - overall_start
        remaining_units = max(0, total_units - completed_units)
        if observed_csv_times:
            average_csv_time = float(np.mean(observed_csv_times))
            eta_seconds = (
                average_csv_time * remaining_units / active_process_count
            )
        elif elapsed >= 30.0:
            # Until the first CSV completes, elapsed time is a rough estimate
            # of one CSV duration for each active process.
            eta_seconds = elapsed * remaining_units / active_process_count
        else:
            eta_seconds = None

        finish_text = (
            format_clock_time(time.time() + eta_seconds)
            if eta_seconds is not None
            else "估算中"
        )
        active_text = ", ".join(
            (
                f"PID{pid}:{row_id}"
                if dataset_index is None
                else f"PID{pid}:{row_id}/{dataset_index}.csv"
            )
            for pid, (row_id, dataset_index) in sorted(active_tasks.items())
        )
        if not active_text:
            active_text = "进程启动/领取任务中"

        line = (
            f"\r剩余≈{format_duration(eta_seconds)}"
            f" | ETA={finish_text}"
            f" | 目录={completed_dirs}/{total_dirs}"
            f" | CSV={min(completed_units, total_units)}/{total_units}"
            f" | {active_text}"
        )
        if failed_dirs:
            line += f" | errors={failed_dirs}"
        print(line.ljust(190), end="", flush=True)

    print("剩余时间估算中...", end="", flush=True)
    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(
            max_workers=config.outer_processes,
            mp_context=ctx,
        ) as executor:
            # One future equals one ID subdirectory. ProcessPoolExecutor keeps
            # these futures in a shared queue. Four persistent processes pull
            # the next directory immediately after completing the current one.
            futures = {
                executor.submit(
                    process_directory_task,
                    row_id,
                    info,
                    config,
                    progress_queue,
                ): row_id
                for row_id, info in tasks
            }
            pending = set(futures)

            while pending:
                done, pending = wait(
                    pending,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                drain_progress(progress_queue)
                show_status()

                for future in done:
                    row_id = futures[future]
                    try:
                        results.append(future.result())
                    except BaseException as exc:
                        # A hard worker/process failure is recorded without
                        # deliberately cancelling the other queued tasks.
                        results.append({
                            "worker_id": None,
                            "pid": None,
                            "row_id": row_id,
                            "status": "FAILED",
                            "error": repr(exc),
                        })
                        consume_message({
                            "type": "directory_done",
                            "worker_id": None,
                            "row_id": row_id,
                            "status": "FAILED",
                            "task_completed_units": 0,
                        })
                        print(
                            f"\n[PROCESS ERROR][{row_id}] "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )

            # Manager queues can lag slightly behind completed futures.
            time.sleep(0.1)
            drain_progress(progress_queue)
            show_status()

    total_elapsed = time.perf_counter() - overall_start
    total_ok = sum(result["status"] != "FAILED" for result in results)
    total_failed = sum(result["status"] == "FAILED" for result in results)

    print(
        (
            f"\r完成 | 总耗时={format_duration(total_elapsed)}"
            f" | 本次任务={total_dirs}"
            f" | 成功={total_ok}"
            f" | 失败={total_failed}"
        ).ljust(190),
        flush=True,
    )

    for result in sorted(results, key=lambda item: item["row_id"]):
        if result["status"] == "FAILED":
            print(
                f"[ERROR][process {result['worker_id']}]"
                f"[{result['row_id']}] {result['error']}",
                flush=True,
            )

    if total_failed:
        raise RuntimeError(f"{total_failed} 个目录处理失败")


if __name__ == "__main__":
    mp.freeze_support()
    main()
