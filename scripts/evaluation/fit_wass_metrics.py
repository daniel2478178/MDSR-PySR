#!/usr/bin/env python3
"""Fit para_eq independently to eight WASS datasets and write mean metrics.

For every row whose TopRank equals the requested value, this program:
1. locates <data-root>/<WASS_K>/<ID>/0.csv ... 7.csv;
2. fits para_eq to each file independently, using para_list and orgpara_list;
3. computes the arithmetic mean of the eight per-file MSE, NMSE, and R2 values;
4. writes them to <WASS_K>_MSE, <WASS_K>_NMSE, and <WASS_K>_R2; and
5. marks a failed row with large finite metrics instead of stopping the run.

NMSE is defined as MSE / Var(y), equivalently SSE / SST.  If any of an ID's
eight files fails, that ID receives the configured failure value for MSE and NMSE and
the negative of that value for R2.  The remaining IDs continue normally.

By default, three independent child processes run WASS_0, WASS_025, and
WASS_04 respectively.  Each WASS child uses ``--workers`` parallel ID-fitting
workers, so the maximum number of fitting workers is the number of WASS labels
times ``--workers``.  Results are saved as
<input-stem>_WASS_0.csv, <input-stem>_WASS_025.csv, and
<input-stem>_WASS_04.csv (or with the input Excel extension).

Example:
    python scripts/evaluation/fit_wass_metrics.py results.csv generated/wass \
        --top-rank 1 --wass WASS_0 WASS_025 WASS_04 --workers 2
"""

from __future__ import annotations

import argparse
import ast
import copy
import keyword
import math
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


DATASET_INDICES = tuple(range(8))
DEFAULT_MAX_NFEV = 10_000
DEFAULT_SEED = 20260824
DEFAULT_FAILURE_METRIC_VALUE = 1.0e100

FUNCTIONS: dict[str, Any] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "abs": np.abs,
    "Mod": np.mod,
}
BUILTIN_CONSTANTS = {"pi": math.pi, "e": math.e}
NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
RANGE_RE = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*\[\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)

_WORKER_PROGRESS_QUEUE: Any = None


class FitDataError(ValueError):
    """Raised for invalid table metadata, datasets, or formulas."""


def initialize_worker_progress(progress_queue: Any) -> None:
    """Attach the inherited progress queue inside each worker process."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


@dataclass(frozen=True)
class PreparedFormula:
    source: str
    compiled: Any
    aliases: Mapping[str, str]
    variable_names: tuple[str, ...]
    fitted_names: tuple[str, ...]
    initial_values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    constants: Mapping[str, float]
    inferred_parameters: tuple[str, ...]


class FormulaValidator(ast.NodeVisitor):
    _binary_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    _unary_ops = (ast.UAdd, ast.USub)

    def __init__(self, allowed_symbols: set[str]) -> None:
        self.allowed_symbols = allowed_symbols

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self._binary_ops):
            raise FitDataError(f"Unsupported operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self._unary_ops):
            raise FitDataError(f"Unsupported unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise FitDataError("The formula calls an unsupported function")
        if node.keywords:
            raise FitDataError("Keyword arguments are not allowed in formula function calls")
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_symbols and node.id not in FUNCTIONS:
            raise FitDataError(f"The formula contains an undefined symbol: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FitDataError("Only numeric constants are allowed in formulas")

    def generic_visit(self, node: ast.AST) -> None:
        raise FitDataError(f"The formula contains unsupported syntax: {type(node).__name__}")


class ProtectedPowerTransformer(ast.NodeTransformer):
    """Replace ``base ** exponent`` with a real-valued protected power call."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="__protected_power", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node


def protected_power(base: Any, exponent: Any) -> Any:
    """Evaluate powers in the real domain during coefficient optimization.

    Positive bases use the ordinary power.  Negative bases preserve the exact
    real sign for integer and one-third-grid exponents; other real exponents
    use ``abs(base) ** exponent``.  This prevents optimizer trial steps from
    turning an otherwise usable candidate into NaN solely because an exponent
    temporarily leaves an integer/rational value.
    """
    base_array = np.asarray(base)
    exponent_array = np.asarray(exponent)
    with np.errstate(all="ignore"):
        magnitude = np.power(np.abs(base_array), exponent_array)
        nearest_integer = np.rint(exponent_array)
        is_integer = np.isclose(
            exponent_array, nearest_integer, rtol=0.0, atol=1.0e-10
        )
        integer_sign = np.where(np.mod(nearest_integer, 2.0) == 0.0, 1.0, -1.0)

        nearest_third_numerator = np.rint(exponent_array * 3.0)
        nearest_third = nearest_third_numerator / 3.0
        is_third = np.isclose(
            exponent_array, nearest_third, rtol=0.0, atol=1.0e-10
        )
        third_sign = np.where(
            np.mod(nearest_third_numerator, 2.0) == 0.0, 1.0, -1.0
        )
        negative_base_sign = np.where(
            is_integer,
            integer_sign,
            np.where(is_third, third_sign, 1.0),
        )
        return np.where(base_array < 0.0, negative_base_sign * magnitude, magnitude)


def normalize_wass_label(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("WASS_K cannot be empty")
    if value.upper().startswith("WASS_"):
        return "WASS_" + value[5:]
    return "WASS_" + value


def split_names(cell: Any, field: str) -> tuple[str, ...]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        raise FitDataError(f"{field} is empty")
    result = tuple(x.strip() for x in re.split(r"[,;]", str(cell)) if x.strip())
    if not result or len(result) != len(set(result)):
        raise FitDataError(f"{field} is empty or contains duplicate names")
    if any(not NAME_RE.fullmatch(x) for x in result):
        raise FitDataError(f"{field} contains invalid names: {result}")
    return result


def parse_para_list(cell: Any) -> tuple[str, ...]:
    text = "" if cell is None else str(cell).strip()
    if text in {"", "[]", "nan"}:
        return ()
    if not (text.startswith("[") and text.endswith("]")):
        raise FitDataError(f"para_list is not a list: {text!r}")
    body = text[1:-1].strip()
    names = tuple(x.strip().strip("'\"") for x in body.split(",") if x.strip())
    if len(names) != len(set(names)) or any(not NAME_RE.fullmatch(x) for x in names):
        raise FitDataError(f"para_list contains duplicate or invalid names: {names}")
    return names


def _eval_numeric_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_numeric_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _eval_numeric_node(node.left), _eval_numeric_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise FitDataError("Initial values may contain only real numbers and +, -, *, /, **")


def parse_initial_values(cell: Any) -> np.ndarray:
    text = "" if cell is None else str(cell).strip()
    if text in {"", "[]", "nan"}:
        return np.empty(0, dtype=float)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FitDataError(f"Invalid orgpara_list syntax: {text!r}") from exc
    if not isinstance(tree.body, (ast.List, ast.Tuple)):
        raise FitDataError("orgpara_list must be a list")
    values = np.asarray([_eval_numeric_node(x) for x in tree.body.elts], dtype=float)
    if not np.all(np.isfinite(values)):
        raise FitDataError("orgpara_list contains non-finite initial values")
    return values


def parse_constants(cell: Any) -> dict[str, float]:
    if cell is None or str(cell).strip().lower() in {"", "nan", "none", "(none)"}:
        return {}
    result: dict[str, float] = {}
    for item in str(cell).split(";"):
        if "=" not in item:
            raise FitDataError(f"Cannot parse FixedConstantValues item: {item!r}")
        name, value = (x.strip() for x in item.split("=", 1))
        if not NAME_RE.fullmatch(name) or name in result:
            raise FitDataError(f"Invalid or duplicate constant name: {name!r}")
        try:
            number = float(value)
        except ValueError as exc:
            raise FitDataError(f"Constant {name} is not numeric") from exc
        if not math.isfinite(number):
            raise FitDataError(f"Constant {name} is not finite")
        result[name] = number
    return result


def parse_parameter_ranges(cell: Any) -> dict[str, tuple[float, float]]:
    if cell is None:
        return {}
    result: dict[str, tuple[float, float]] = {}
    for name, low_text, high_text in RANGE_RE.findall(str(cell)):
        low, high = float(low_text), float(high_text)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise FitDataError(f"Invalid range for {name} in ParameterRange")
        result[name] = (low, high)
    return result


def formula_names(tree: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS
    }


def prepare_formula(row: Mapping[str, Any], strict_symbols: bool) -> PreparedFormula:
    variables = split_names(row.get("IndependentVars"), "IndependentVars")
    listed_parameters = parse_para_list(row.get("para_list"))
    listed_initials = parse_initial_values(row.get("orgpara_list"))
    if len(listed_parameters) != len(listed_initials):
        raise FitDataError(
            f"para_list has {len(listed_parameters)} items, but orgpara_list has {len(listed_initials)}"
        )
    constants = parse_constants(row.get("FixedConstantValues"))
    parameter_ranges = parse_parameter_ranges(row.get("ParameterRange"))

    source = str(row.get("para_eq", "")).strip().replace("^", "**")
    if not source or source.lower() == "nan":
        raise FitDataError("para_eq is empty")
    aliases = {
        name: f"_symbol_{name}"
        for name in (*variables, *listed_parameters, *constants, *parameter_ranges)
        if keyword.iskeyword(name)
    }
    parse_source = source
    for original, alias in aliases.items():
        parse_source = re.sub(rf"\b{re.escape(original)}\b", alias, parse_source)
    try:
        preliminary_tree = ast.parse(parse_source, mode="eval")
    except SyntaxError as exc:
        raise FitDataError(f"Invalid para_eq syntax: {source}") from exc

    aliased_variables = {aliases.get(x, x) for x in variables}
    aliased_listed = {aliases.get(x, x) for x in listed_parameters}
    aliased_constants = {aliases.get(x, x) for x in constants}
    unresolved_aliases = formula_names(preliminary_tree) - aliased_variables - aliased_listed - aliased_constants
    unresolved = tuple(
        original
        for alias in sorted(unresolved_aliases)
        for original in [next((k for k, v in aliases.items() if v == alias), alias)]
        if original not in BUILTIN_CONSTANTS
    )
    builtin_used = {
        name: value
        for name, value in BUILTIN_CONSTANTS.items()
        if aliases.get(name, name) in unresolved_aliases
    }
    constants = {**builtin_used, **constants}
    unresolved = tuple(x for x in unresolved if x not in constants)

    if unresolved and strict_symbols:
        raise FitDataError(f"para_eq contains unresolved symbols outside para_list: {unresolved}")
    missing_ranges = [name for name in unresolved if name not in parameter_ranges]
    if missing_ranges:
        raise FitDataError(
            f"para_eq contains unresolved symbols with no ParameterRange entry: {missing_ranges}"
        )

    inferred_initials = np.asarray(
        [(parameter_ranges[x][0] + parameter_ranges[x][1]) / 2 for x in unresolved], dtype=float
    )
    initial_values = np.concatenate([listed_initials, inferred_initials])
    fitted_names = (*listed_parameters, *unresolved)
    lower_bounds = np.full(len(fitted_names), -np.inf, dtype=float)
    upper_bounds = np.full(len(fitted_names), np.inf, dtype=float)
    for index, name in enumerate(fitted_names[len(listed_parameters):], start=len(listed_parameters)):
        lower_bounds[index], upper_bounds[index] = parameter_ranges[name]

    allowed = {
        aliases.get(name, name)
        for name in (*variables, *fitted_names, *constants)
    }
    FormulaValidator(allowed).visit(preliminary_tree)
    protected_tree = ProtectedPowerTransformer().visit(preliminary_tree)
    ast.fix_missing_locations(protected_tree)
    return PreparedFormula(
        source=source,
        compiled=compile(protected_tree, "<para_eq>", "eval"),
        aliases=aliases,
        variable_names=variables,
        fitted_names=fitted_names,
        initial_values=initial_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constants=constants,
        inferred_parameters=unresolved,
    )


def evaluate_formula(
    formula: PreparedFormula,
    x_columns: Mapping[str, np.ndarray],
    parameter_values: Sequence[float],
) -> np.ndarray:
    environment: dict[str, Any] = dict(FUNCTIONS)
    environment["__protected_power"] = protected_power
    environment.update(formula.constants)
    environment.update(zip(formula.fitted_names, parameter_values))
    environment.update(x_columns)
    for original, alias in formula.aliases.items():
        if original in environment:
            environment[alias] = environment[original]
    with np.errstate(all="ignore"):
        result = eval(formula.compiled, {"__builtins__": {}}, environment)
    values = np.asarray(result)
    sample_count = len(next(iter(x_columns.values())))
    if values.ndim == 0:
        values = np.full(sample_count, values.item())
    else:
        try:
            values = np.broadcast_to(values, (sample_count,))
        except ValueError as exc:
            raise FitDataError("para_eq output cannot be converted to a one-dimensional vector") from exc
    if np.iscomplexobj(values):
        values = np.where(np.abs(values.imag) <= 1e-12, values.real, np.nan)
    try:
        return values.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise FitDataError("para_eq output is not real-valued") from exc


def read_dataset(path: Path, variables: Sequence[str], target: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = [*variables, target]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise FitDataError(f"{path} is missing columns {missing}; available columns: {list(frame.columns)}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(numeric.to_numpy(dtype=float)), axis=1)
    if not np.any(valid):
        raise FitDataError(f"{path} contains no finite numeric samples")
    numeric = numeric.loc[valid]
    x_columns = {name: numeric[name].to_numpy(dtype=float) for name in variables}
    y = numeric[target].to_numpy(dtype=float)
    return x_columns, y


def initial_candidates(
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    restarts: int,
    seed: int,
) -> list[np.ndarray]:
    if len(initial) == 0:
        return [initial]
    candidates = [np.clip(initial, lower, upper)]
    rng = np.random.default_rng(seed)
    for _ in range(1, restarts):
        scale = 0.15 * (np.abs(initial) + 1.0)
        candidate = initial + rng.normal(size=len(initial)) * scale
        candidates.append(np.clip(candidate, lower, upper))
    return candidates


def fit_one_dataset(
    formula: PreparedFormula,
    x_columns: Mapping[str, np.ndarray],
    y: np.ndarray,
    max_nfev: int,
    restarts: int,
    seed: int,
) -> tuple[float, float, float]:
    """Fit one dataset while retaining the best fully finite trial point.

    The optimizer is allowed to explore invalid regions, but invalid residuals
    are penalized and are never accepted as the final solution.  If the final
    optimizer step is non-finite, the best finite point visited earlier is used.
    """
    if len(formula.fitted_names) == 0:
        prediction = evaluate_formula(formula, x_columns, ())
        if not np.all(np.isfinite(prediction)):
            raise FitDataError(
                "The formula produces non-finite values and has no fitted parameters"
            )
    else:
        penalty = np.maximum(1.0, np.abs(y)) * 1.0e6
        best_parameters: np.ndarray | None = None
        best_sse = math.inf

        def evaluate_and_record(parameters: np.ndarray) -> np.ndarray:
            """Return finite residuals and remember the best valid parameters."""
            nonlocal best_parameters, best_sse

            prediction_local = evaluate_formula(formula, x_columns, parameters)
            residual = prediction_local - y
            finite = np.isfinite(residual)

            if np.all(finite):
                sse = float(np.dot(residual, residual))
                if math.isfinite(sse) and sse < best_sse:
                    best_sse = sse
                    best_parameters = np.asarray(parameters, dtype=float).copy()

            # scipy.optimize.least_squares requires finite residual values.
            return np.where(finite, residual, penalty)

        candidates = initial_candidates(
            formula.initial_values,
            formula.lower_bounds,
            formula.upper_bounds,
            restarts,
            seed,
        )
        fitting_errors: list[str] = []

        for candidate in candidates:
            # Preserve the initial point in case a later step crosses the
            # formula's real-valued domain boundary.
            evaluate_and_record(candidate)
            try:
                result = least_squares(
                    evaluate_and_record,
                    candidate,
                    bounds=(formula.lower_bounds, formula.upper_bounds),
                    method="trf",
                    x_scale="jac",
                    max_nfev=max_nfev,
                )
                # Explicitly inspect the point returned by the optimizer.
                evaluate_and_record(result.x)
            except (ValueError, FloatingPointError) as exc:
                fitting_errors.append(str(exc))

        if best_parameters is None:
            details = (
                f"; optimizer errors: {' | '.join(fitting_errors[:3])}"
                if fitting_errors
                else ""
            )
            raise FitDataError(
                "No parameter vector produced finite predictions"
                f"{details}. Try increasing --restarts."
            )

        prediction = evaluate_formula(formula, x_columns, best_parameters)

    if not np.all(np.isfinite(prediction)):
        raise FitDataError(
            "Internal error: selected parameters produce non-finite values"
        )

    residual = y - prediction
    residual_sum = float(np.dot(residual, residual))
    if not math.isfinite(residual_sum):
        raise FitDataError("The residual sum of squares is non-finite")
    mse = residual_sum / len(y)
    total = float(np.sum((y - np.mean(y)) ** 2))

    if total == 0.0:
        if residual_sum == 0.0:
            nmse, r2 = 0.0, 1.0
        else:
            raise FitDataError("NMSE is undefined because the target variance is zero")
    else:
        nmse = residual_sum / total
        r2 = 1.0 - nmse
    return mse, nmse, r2


def resolve_wass_root(data_root: Path, wass_label: str) -> Path:
    data_root = data_root.expanduser().resolve()
    return data_root if data_root.name.lower() == wass_label.lower() else data_root / wass_label


def fit_row_task(
    row_position: int,
    row: Mapping[str, Any],
    wass_root_text: str,
    max_nfev: int,
    restarts: int,
    seed: int,
    strict_symbols: bool,
    failure_value: float,
    progress_queue: Any = None,
) -> tuple[int, str, float, float, float, tuple[str, ...], tuple[str, ...]]:
    model_id = str(row.get("ID", "")).strip()
    if not model_id:
        model_id = f"table-row-{row_position + 2}"

    active_progress_queue = (
        progress_queue if progress_queue is not None else _WORKER_PROGRESS_QUEUE
    )

    try:
        formula = prepare_formula(row, strict_symbols)
        target = str(row.get("Target", "")).strip()
        if not NAME_RE.fullmatch(target):
            raise FitDataError(f"Invalid Target name: {target!r}")
    except Exception as exc:
        message = f"Metadata/formula failed for ID={model_id}: {exc}"
        print(f"Warning: {message}; writing failure markers and continuing.", flush=True)
        if active_progress_queue is not None:
            for dataset_index in DATASET_INDICES:
                active_progress_queue.put(
                    (
                        row_position, model_id, dataset_index,
                        failure_value, failure_value, -failure_value, True,
                    )
                )
        return (
            row_position, model_id,
            failure_value, failure_value, -failure_value,
            (), (message,),
        )

    id_root = Path(wass_root_text) / model_id
    metrics: list[tuple[float, float, float]] = []
    failures: list[str] = []
    for dataset_index in DATASET_INDICES:
        try:
            x_columns, y = read_dataset(
                id_root / f"{dataset_index}.csv", formula.variable_names, target
            )
            mse, nmse, r2 = fit_one_dataset(
                formula,
                x_columns,
                y,
                max_nfev,
                restarts,
                seed + row_position * 1009 + dataset_index,
            )
        except Exception as exc:
            message = f"Fit failed at {model_id}/{dataset_index}.csv: {exc}"
            failures.append(message)
            print(f"Warning: {message}; continuing.", flush=True)
            mse, nmse, r2 = failure_value, failure_value, -failure_value
        metrics.append((mse, nmse, r2))
        if active_progress_queue is not None:
            active_progress_queue.put(
                (
                    row_position, model_id, dataset_index,
                    mse, nmse, r2, bool(failures and failures[-1].startswith(
                        f"Fit failed at {model_id}/{dataset_index}.csv"
                    )),
                )
            )

    # Keep the marker exact instead of diluting it by averaging one failed
    # dataset with seven successful datasets.
    if failures:
        mean_mse, mean_nmse, mean_r2 = (
            failure_value, failure_value, -failure_value
        )
    else:
        mean_mse = float(np.mean([x[0] for x in metrics]))
        mean_nmse = float(np.mean([x[1] for x in metrics]))
        mean_r2 = float(np.mean([x[2] for x in metrics]))
    return (
        row_position, model_id, mean_mse, mean_nmse, mean_r2,
        formula.inferred_parameters, tuple(failures),
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def monitor_progress(
    progress_queue: Any,
    total_files: int,
    total_rows: int,
    start_time: float,
) -> None:
    """Render file-level progress, elapsed time, throughput, and ETA."""
    completed_files = 0
    completed_rows = 0
    row_file_counts: dict[int, int] = {}
    interactive = sys.stdout.isatty()
    last_noninteractive_percent = -1
    while True:
        try:
            event = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if event is None:
            break
        row_position, model_id, dataset_index, mse, nmse, r2, failed = event
        completed_files += 1
        row_file_counts[row_position] = row_file_counts.get(row_position, 0) + 1
        if row_file_counts[row_position] == len(DATASET_INDICES):
            completed_rows += 1

        elapsed = max(time.perf_counter() - start_time, 1e-9)
        rate = completed_files / elapsed
        eta = (total_files - completed_files) / rate if rate > 0 else math.inf
        fraction = completed_files / total_files
        percent = int(fraction * 100)
        if interactive:
            width = 28
            filled = min(width, int(fraction * width))
            bar = "#" * filled + "-" * (width - filled)
            line = (
                f"\r[{bar}] {fraction * 100:6.2f}% | files {completed_files}/{total_files} "
                f"| ID {completed_rows}/{total_rows} | {model_id}/{dataset_index}.csv "
                f"| {rate:.2f} files/s | elapsed {format_duration(elapsed)} "
                f"| ETA {format_duration(eta)}"
            )
            print(line.ljust(170), end="", flush=True)
        elif percent > last_noninteractive_percent and (
            percent % 5 == 0 or completed_files == total_files
        ):
            last_noninteractive_percent = percent
            print(
                f"[Progress {fraction * 100:6.2f}%] files {completed_files}/{total_files}, "
                f"IDs {completed_rows}/{total_rows}, current {model_id}/{dataset_index}.csv, "
                f"MSE={mse:.6g}, NMSE={nmse:.6g}, R2={r2:.6g}, "
                f"status={'FAILED' if failed else 'OK'}, elapsed {format_duration(elapsed)}, "
                f"ETA {format_duration(eta)}",
                flush=True,
            )
    if interactive:
        print(flush=True)


def run_fits(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    wass_root: Path,
    args: argparse.Namespace,
) -> list[tuple[int, str, float, float, float, tuple[str, ...], tuple[str, ...]]]:
    total_files = len(records) * len(DATASET_INDICES)
    print(
        f"Starting fits: {len(records)} rows x {len(DATASET_INDICES)} datasets = {total_files} files; "
        f"worker processes={args.workers}",
        flush=True,
    )
    start_time = time.perf_counter()
    results: list[
        tuple[int, str, float, float, float, tuple[str, ...], tuple[str, ...]]
    ] = []

    if args.workers == 1:
        progress_queue: Any = queue.Queue() if args.progress else None
        monitor = None
        if progress_queue is not None:
            monitor = threading.Thread(
                target=monitor_progress,
                args=(progress_queue, total_files, len(records), start_time),
                daemon=True,
            )
            monitor.start()
        try:
            for position, row in records:
                try:
                    result = fit_row_task(
                        position, row, str(wass_root), args.max_nfev, args.restarts,
                        args.seed, args.strict_symbols, args.failure_value,
                        progress_queue,
                    )
                except Exception as exc:
                    model_id = str(row.get("ID", "")).strip() or f"table-row-{position + 2}"
                    message = (
                        f"Unexpected row failure for ID={model_id} "
                        f"(table row {position + 2}): {exc}"
                    )
                    print(f"Warning: {message}; writing failure markers and continuing.", flush=True)
                    result = (
                        position, model_id,
                        args.failure_value, args.failure_value, -args.failure_value,
                        (), (message,),
                    )
                results.append(result)
        finally:
            if progress_queue is not None:
                progress_queue.put(None)
            if monitor is not None:
                monitor.join()
    else:
        context_name = "spawn" if os.name == "nt" else "fork"
        process_context = multiprocessing.get_context(context_name)
        progress_queue = process_context.Queue() if args.progress else None
        monitor = None
        if progress_queue is not None:
            monitor = threading.Thread(
                target=monitor_progress,
                args=(progress_queue, total_files, len(records), start_time),
                daemon=True,
            )
            monitor.start()
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=process_context,
                initializer=initialize_worker_progress,
                initargs=(progress_queue,),
            ) as executor:
                futures = {
                    executor.submit(
                        fit_row_task,
                        position,
                        row,
                        str(wass_root),
                        args.max_nfev,
                        args.restarts,
                        args.seed,
                        args.strict_symbols,
                        args.failure_value,
                    ): (position, str(row["ID"]))
                    for position, row in records
                }
                for future in as_completed(futures):
                    position, model_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        message = (
                            f"Unexpected row failure for ID={model_id} "
                            f"(table row {position + 2}): {exc}"
                        )
                        print(
                            f"Warning: {message}; writing failure markers and continuing.",
                            flush=True,
                        )
                        results.append(
                            (
                                position, model_id,
                                args.failure_value, args.failure_value,
                                -args.failure_value, (), (message,),
                            )
                        )
        finally:
            if progress_queue is not None:
                progress_queue.put(None)
            if monitor is not None:
                monitor.join()
            if progress_queue is not None:
                progress_queue.close()

    elapsed = time.perf_counter() - start_time
    failed_rows = sum(bool(result[6]) for result in results)
    print(
        f"Fitting complete: {len(results)}/{len(records)} rows, {total_files} files, "
        f"failed rows marked={failed_rows}, total elapsed {format_duration(elapsed)}",
        flush=True,
    )
    return results


def read_table(path: Path, sheet: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xlsm"}:
        sheet_name: str | int = int(sheet) if isinstance(sheet, str) and sheet.isdigit() else sheet
        return pd.read_excel(path, sheet_name=sheet_name)
    raise FitDataError("The results table must be a .csv, .xlsx, or .xlsm file")


def write_table_atomic(
    frame: pd.DataFrame,
    source_path: Path,
    output_path: Path,
    sheet: str | int,
    metric_columns: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        try:
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return
    if suffix in {".xlsx", ".xlsm"}:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise FitDataError("openpyxl is required to process XLSX/XLSM files") from exc
        temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        try:
            workbook = load_workbook(source_path, keep_vba=source_path.suffix.lower() == ".xlsm")
            worksheet = (
                workbook.worksheets[int(sheet)]
                if isinstance(sheet, str) and sheet.isdigit()
                else workbook[str(sheet)]
            )
            headers = {
                str(worksheet.cell(1, column).value).strip(): column
                for column in range(1, worksheet.max_column + 1)
                if worksheet.cell(1, column).value is not None
            }
            for metric_column in metric_columns:
                if metric_column in headers:
                    column = headers[metric_column]
                else:
                    column = worksheet.max_column + 1
                    headers[metric_column] = column
                    worksheet.cell(1, column).value = metric_column
                    if column > 1:
                        worksheet.cell(1, column)._style = copy.copy(worksheet.cell(1, column - 1)._style)
                        worksheet.cell(1, column).font = copy.copy(worksheet.cell(1, column - 1).font)
                        worksheet.cell(1, column).fill = copy.copy(worksheet.cell(1, column - 1).fill)
                        worksheet.cell(1, column).border = copy.copy(worksheet.cell(1, column - 1).border)
                        worksheet.cell(1, column).alignment = copy.copy(worksheet.cell(1, column - 1).alignment)
                for frame_position, value in enumerate(frame[metric_column], start=2):
                    cell = worksheet.cell(frame_position, column)
                    cell.value = None if pd.isna(value) else float(value)
                    cell.number_format = (
                        "0.0000000000E+00"
                        if metric_column.endswith(("_MSE", "_NMSE"))
                        else "0.0000000000"
                    )
            workbook.save(temporary)
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return
    raise FitDataError(f"Unsupported output format: {output_path.suffix}")


def validate_columns(frame: pd.DataFrame) -> None:
    required = {
        "ID", "TopRank", "Target", "IndependentVars", "ParameterRange",
        "FixedConstantValues", "para_eq", "para_list", "orgpara_list",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FitDataError(f"The results table is missing columns: {missing}")


def append_wass_to_filename(path: Path, wass_label: str) -> Path:
    """Return result_WASS_K.csv/xlsx without changing the parent directory."""
    return path.with_name(f"{path.stem}_{wass_label}{path.suffix}")


def stream_child_output(label: str, process: subprocess.Popen[str]) -> None:
    """Prefix each child line so three concurrent logs remain readable."""
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[{label}] {line}", end="", flush=True)


def run_multiple_wass_processes(
    args: argparse.Namespace,
    table_path: Path,
    wass_labels: Sequence[str],
) -> int:
    """Launch one WASS process per label, each with parallel ID workers."""
    base_output = args.output.expanduser().resolve() if args.output else table_path
    if base_output.suffix.lower() != table_path.suffix.lower():
        raise FitDataError("The --output extension must match the input-table extension")

    jobs: list[tuple[str, Path, list[str]]] = []
    for label in wass_labels:
        wass_root = resolve_wass_root(args.data_root, label)
        if not wass_root.is_dir():
            raise FileNotFoundError(f"WASS data directory not found: {wass_root}")
        output_path = append_wass_to_filename(base_output, label)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(table_path),
            str(args.data_root),
            "--top-rank", str(args.top_rank),
            "--wass", label,
            "--output", str(output_path),
            "--sheet", str(args.sheet),
            "--workers", str(args.workers),
            "--max-nfev", str(args.max_nfev),
            "--restarts", str(args.restarts),
            "--seed", str(args.seed),
            "--failure-value", str(args.failure_value),
        ]
        if not args.progress:
            command.append("--no-progress")
        if args.strict_symbols:
            command.append("--strict-symbols")
        if args.validate_only:
            command.append("--validate-only")
        jobs.append((label, output_path, command))

    print(
        "Starting independent WASS processes: " + ", ".join(
            f"process {index}={label}" for index, label in enumerate(wass_labels)
        )
        + f"; ID workers per WASS={args.workers}; "
        + f"maximum fitting workers={len(wass_labels) * args.workers}",
        flush=True,
    )
    processes: list[tuple[str, Path, subprocess.Popen[str], threading.Thread]] = []
    for label, output_path, command in jobs:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        reader = threading.Thread(
            target=stream_child_output,
            args=(label, process),
            daemon=True,
        )
        reader.start()
        processes.append((label, output_path, process, reader))

    failures: list[tuple[str, int]] = []
    for label, _, process, reader in processes:
        return_code = process.wait()
        reader.join()
        if return_code != 0:
            failures.append((label, return_code))
    if failures:
        detail = ", ".join(f"{label}(exit code={code})" for label, code in failures)
        raise FitDataError(f"One or more WASS processes failed: {detail}")

    if args.validate_only:
        print("All three WASS processes completed validation.", flush=True)
    else:
        print("All three WASS processes completed. Output files:", flush=True)
        for label, output_path, _, _ in processes:
            print(f"  {label}: {output_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit para_eq independently to eight WASS_K datasets and write mean MSE/NMSE/R2 without stopping on individual fit failures"
    )
    parser.add_argument("table", type=Path, help="Input results table (CSV/XLSX); single-WASS mode overwrites it by default")
    parser.add_argument(
        "data_root", type=Path,
        help="Data root: either the parent of WASS_K directories or one WASS_K directory",
    )
    parser.add_argument("--top-rank", type=int, required=True, help="Process only rows whose TopRank equals this value")
    parser.add_argument(
        "--wass", type=normalize_wass_label, nargs="+",
        default=["WASS_0", "WASS_025", "WASS_04"],
        help="One or more WASS labels; default: WASS_0 WASS_025 WASS_04",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Exact output path in single-WASS mode; base filename in multi-WASS mode, with _WASS_K appended",
    )
    parser.add_argument("--sheet", default="0", help="XLSX worksheet name or zero-based index; default: first worksheet")
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help=(
            "Parallel ID workers per WASS label; in multi-WASS mode the "
            "maximum fitting workers equal number_of_WASS_labels * workers"
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV, help="Maximum function evaluations per least-squares fit")
    parser.add_argument("--restarts", type=int, default=1, help="Initial-value attempts per dataset; default: use orgpara_list once")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for restarted fits")
    parser.add_argument(
        "--failure-value",
        type=float,
        default=DEFAULT_FAILURE_METRIC_VALUE,
        help=(
            "Large finite marker written to MSE/NMSE when an ID fails; "
            "R2 receives its negative (default: 1e100)"
        ),
    )
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="Disable file-level progress, throughput, elapsed time, and ETA output",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--validate-only", action="store_true", help="Validate matching formulas and metadata without fitting")
    parser.add_argument(
        "--strict-symbols", action="store_true",
        help="Fail if para_eq contains symbols outside para_list, variables, and constants; by default ParameterRange symbols are inferred as fitted parameters",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.max_nfev <= 0 or args.restarts <= 0:
        raise ValueError("--workers, --max-nfev, and --restarts must be greater than zero")
    if not math.isfinite(args.failure_value) or args.failure_value <= 0:
        raise ValueError("--failure-value must be a finite number greater than zero")
    table_path = args.table.expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Results table not found: {table_path}")
    frame = read_table(table_path, args.sheet)
    validate_columns(frame)
    wass_labels = tuple(args.wass)
    if len(wass_labels) != len(set(wass_labels)):
        raise FitDataError(f"--wass contains duplicate labels: {wass_labels}")
    ranks = pd.to_numeric(frame["TopRank"], errors="coerce")
    positions = np.flatnonzero(ranks.to_numpy() == args.top_rank).tolist()
    if not positions:
        raise FitDataError(f"No rows have TopRank={args.top_rank}")

    records = [(position, frame.iloc[position].to_dict()) for position in positions]
    inferred_summary: list[tuple[str, tuple[str, ...]]] = []
    validation_failure_count = 0
    for _, row in records:
        try:
            formula = prepare_formula(row, args.strict_symbols)
        except Exception as exc:
            validation_failure_count += 1
            print(
                f"Warning: validation failed for ID={row.get('ID')}: {exc}; "
                "the row will receive failure markers and processing will continue.",
                flush=True,
            )
            continue
        if formula.inferred_parameters:
            inferred_summary.append((str(row["ID"]), formula.inferred_parameters))
    print(
        f"Table scan complete: {len(records)} matching rows, TopRank={args.top_rank}, "
        f"formula warnings={validation_failure_count}, "
        f"WASS={', '.join(wass_labels)}"
    )
    if inferred_summary:
        preview = "; ".join(f"{model_id}:{'/'.join(names)}" for model_id, names in inferred_summary[:6])
        print(f"Note: additional physical parameters inferred from ParameterRange: {preview}")
    if len(wass_labels) > 1:
        return run_multiple_wass_processes(args, table_path, wass_labels)
    args.wass = wass_labels[0]
    if args.validate_only:
        return 0

    wass_root = resolve_wass_root(args.data_root, args.wass)
    if not wass_root.is_dir():
        raise FileNotFoundError(f"WASS data directory not found: {wass_root}")
    mse_column = f"{args.wass}_MSE"
    nmse_column = f"{args.wass}_NMSE"
    r2_column = f"{args.wass}_R2"
    if mse_column not in frame.columns:
        frame[mse_column] = np.nan
    if nmse_column not in frame.columns:
        frame[nmse_column] = np.nan
    if r2_column not in frame.columns:
        frame[r2_column] = np.nan

    results = run_fits(records, wass_root, args)

    for position, _, mean_mse, mean_nmse, mean_r2, _, _ in results:
        frame.at[frame.index[position], mse_column] = mean_mse
        frame.at[frame.index[position], nmse_column] = mean_nmse
        frame.at[frame.index[position], r2_column] = mean_r2
    output_path = args.output.expanduser().resolve() if args.output else table_path
    if output_path.suffix.lower() != table_path.suffix.lower():
        raise FitDataError("The --output extension must match the input-table extension")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        write_table_atomic(
            frame, table_path, output_path, args.sheet,
            (mse_column, nmse_column, r2_column),
        )
    print(
        f"Write complete: {mse_column}, {nmse_column}, {r2_column} "
        f"-> {output_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FitDataError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
