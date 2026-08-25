#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate symbolic-regression CSV datasets from physicsMDSR_Range.xlsx.

For each row (ID):
  ID/
    0.csv
    1.csv
    ...
    15.csv

Each CSV contains exactly N_SAMPLES valid rows:
    IndependentVar1, IndependentVar2, ..., Target

Data generation:
1. Read the i-th variable ranges from varRange.
2. Draw each IndependentVar independently from Uniform(low, high).
3. Use the i-th paraValue vector as dataset-specific parameters.
4. Use FixedConstantValues as fixed constants.
5. Evaluate GenerationFormula.
6. Reject NaN/Inf/domain-invalid results and, by default, results outside
   the corresponding targetRange interval.
7. Continue until exactly 5000 valid samples are collected.

The targetRange rejection keeps generated data inside the physical/branch
range designed in the workbook. Set ENFORCE_TARGET_RANGE=False if you want
unconditional rectangular Uniform sampling over varRange.
"""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


N_SAMPLES = 5000
N_GROUPS = 16
BASE_SEED = 20260819
ENFORCE_TARGET_RANGE = True
BATCH_MIN = 4096
MAX_BATCH_ATTEMPTS = 10000


# -----------------------------
# Parsing helpers
# -----------------------------

RANGE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|in)\s*\[\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)
NUMBER_PREFIX_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+.*)?$"
)


def split_names(text: object) -> list[str]:
    if text is None:
        return []
    s = str(text).strip()
    if not s or s.lower() in {"(none)", "none", "nan"}:
        return []
    return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]


def parse_parameter_names(text: object) -> list[str]:
    """Parameter order is exactly the order appearing in ParameterRange."""
    if text is None:
        return []
    return [m[0] for m in RANGE_RE.findall(str(text))]


def parse_fixed_constants(text: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if text is None:
        return out

    s = str(text).strip()
    if not s or s.lower() in {"(none)", "none", "nan"}:
        return out

    for part in s.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        match = NUMBER_PREFIX_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f"Cannot parse FixedConstantValues item {part!r}"
            )
        out[name] = float(match.group(1))
    return out


def parse_json_cell(value: object, label: str):
    if value is None:
        raise ValueError(f"{label} is empty")
    if isinstance(value, (list, tuple)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON-like value in {label}: {value!r}") from exc


def safe_folder_name(name: object) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("Empty ID")
    # Preserve normal IDs such as P01 exactly, but prevent path traversal.
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)


def stable_seed(row_id: str, group_index: int) -> int:
    msg = f"{BASE_SEED}|{row_id}|{group_index}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


# -----------------------------
# Safe vectorized formula evaluator
# -----------------------------

ALLOWED_FUNCS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "acos": np.arccos,
    "arccos": np.arccos,
    "asin": np.arcsin,
    "arcsin": np.arcsin,
    "atan": np.arctan,
    "arctan": np.arctan,
    "atanh": np.arctanh,
    "arctanh": np.arctanh,
    "tanh": np.tanh,
    "abs": np.abs,
}

BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}


def normalize_formula_text(expr: object) -> str:
    s = str(expr).strip()
    if "=" in s:
        s = s.split("=", 1)[1].strip()

    # "lambda" is a Python keyword, so rename it only as a symbol token.
    s = re.sub(r"\blambda\b", "lambda_", s)
    return s


class FormulaEvaluator:
    def __init__(self, formula: object):
        self.formula_text = normalize_formula_text(formula)
        self.tree = ast.parse(self.formula_text, mode="eval")

    def __call__(self, env: dict[str, object]) -> np.ndarray:
        safe_env = dict(env)
        if "lambda" in safe_env:
            safe_env["lambda_"] = safe_env["lambda"]

        # pi is allowed as the mathematical constant if the sheet did not
        # explicitly provide a fixed pi value.
        safe_env.setdefault("pi", math.pi)

        with np.errstate(
            invalid="ignore",
            divide="ignore",
            over="ignore",
            under="ignore",
        ):
            result = self._eval_node(self.tree.body, safe_env)

        return np.asarray(result)

    def _eval_node(self, node: ast.AST, env: dict[str, object]):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"Unknown symbol in formula: {node.id}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return BINOPS[op_type](
                self._eval_node(node.left, env),
                self._eval_node(node.right, env),
            )

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARYOPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return UNARYOPS[op_type](self._eval_node(node.operand, env))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple mathematical function calls are allowed")
            fn_name = node.func.id
            if fn_name not in ALLOWED_FUNCS:
                raise ValueError(f"Unsupported function: {fn_name}")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in formulas")
            args = [self._eval_node(arg, env) for arg in node.args]
            return ALLOWED_FUNCS[fn_name](*args)

        raise ValueError(f"Unsupported syntax node: {type(node).__name__}")


# -----------------------------
# Dataset generation
# -----------------------------

def build_group_environment(
    parameter_names: list[str],
    parameter_vector: list[float],
    fixed_constants: dict[str, float],
) -> dict[str, float]:
    if len(parameter_names) != len(parameter_vector):
        raise ValueError(
            f"Parameter dimension mismatch: names={parameter_names}, "
            f"vector={parameter_vector}"
        )

    env = {
        name: float(parameter_vector[i])
        for i, name in enumerate(parameter_names)
    }
    env.update(fixed_constants)
    return env


def generate_one_group(
    evaluator: FormulaEvaluator,
    independent_vars: list[str],
    variable_ranges: list[list[float]],
    target_name: str,
    target_range: list[float] | None,
    base_env: dict[str, float],
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, float]:
    if len(independent_vars) != len(variable_ranges):
        raise ValueError(
            f"IndependentVars/varRange dimension mismatch: "
            f"{independent_vars} vs {variable_ranges}"
        )

    for pair in variable_ranges:
        if len(pair) != 2:
            raise ValueError(f"Bad variable range: {pair}")
        try:
            lo, hi = map(float, pair)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Non-numeric variable range: {pair}") from exc
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite variable range: {pair}")
        if hi < lo:
            raise ValueError(f"Reversed variable range: {pair}")

    accepted_blocks: list[np.ndarray] = []
    accepted = 0
    proposed = 0

    for attempt in range(MAX_BATCH_ATTEMPTS):
        if accepted >= n_samples:
            break

        need = n_samples - accepted
        # Oversample to make rejection sampling efficient.
        batch_n = max(BATCH_MIN, need * 2)

        columns = []
        env = dict(base_env)

        for var_name, (lo, hi) in zip(independent_vars, variable_ranges):
            lo = float(lo)
            hi = float(hi)

            if hi == lo:
                x = np.full(batch_n, lo, dtype=float)
            else:
                x = rng.uniform(lo, hi, size=batch_n)

            env[var_name] = x
            columns.append(x)

        y = evaluator(env)

        # Broadcast scalar target if needed.
        if y.ndim == 0:
            y = np.full(batch_n, float(y), dtype=float)
        else:
            y = np.asarray(y)
            if y.shape != (batch_n,):
                try:
                    y = np.broadcast_to(y, (batch_n,))
                except ValueError as exc:
                    raise ValueError(
                        f"Formula returned unexpected shape {y.shape}"
                    ) from exc

        # Reject complex outputs except tiny roundoff imaginary parts.
        if np.iscomplexobj(y):
            imag_ok = np.abs(np.imag(y)) <= 1e-12
            y_real = np.real(y)
        else:
            imag_ok = np.ones(batch_n, dtype=bool)
            y_real = np.asarray(y, dtype=float)

        mask = imag_ok & np.isfinite(y_real)

        # targetRange represents the intended physical/branch range for the
        # corresponding parameter group, so enforce it by default.
        if ENFORCE_TARGET_RANGE and target_range is not None:
            t_lo, t_hi = map(float, target_range)
            tol = 1e-10 * max(1.0, abs(t_lo), abs(t_hi))
            mask &= (y_real >= t_lo - tol) & (y_real <= t_hi + tol)

        proposed += batch_n

        if np.any(mask):
            block_cols = [
                np.asarray(col)[mask] for col in columns
            ]
            block_cols.append(y_real[mask])
            block = np.column_stack(block_cols)

            take = min(need, block.shape[0])
            accepted_blocks.append(block[:take])
            accepted += take

    if accepted < n_samples:
        acceptance = accepted / max(proposed, 1)
        raise RuntimeError(
            f"Could obtain only {accepted}/{n_samples} valid samples "
            f"after {proposed} proposals; acceptance={acceptance:.6g}. "
            f"Check varRange/targetRange/domain constraints."
        )

    data = np.vstack(accepted_blocks)
    acceptance_rate = accepted / proposed
    return data, acceptance_rate


def load_sheet_rows(xlsx_path: Path) -> tuple[list[str], list[list[object]]]:
    """
    Read the workbook with openpyxl.

    Prefers the 'Sampling design' sheet. If it does not exist,
    falls back to the first worksheet.
    """
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)

    if "Sampling design" in wb.sheetnames:
        ws = wb["Sampling design"]
    else:
        ws = wb[wb.sheetnames[0]]

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Workbook sheet is empty")

    headers = [
        str(x).strip() if x is not None else ""
        for x in rows[0]
    ]

    data_rows = []
    for row in rows[1:]:
        if not row:
            continue
        first = row[0]
        if first is None or str(first).strip() == "":
            continue
        data_rows.append(list(row))

    wb.close()
    return headers, data_rows

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Benchmark metadata workbook")
    parser.add_argument("output_root", type=Path, help="Output directory containing one folder per ID")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Rows per generated CSV")
    parser.add_argument("--groups", type=int, default=N_GROUPS, help="Datasets generated per ID")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed")
    parser.add_argument(
        "--formula-column",
        choices=("OriginalFormula", "GenerationFormula"),
        default="OriginalFormula",
        help="Workbook formula column to evaluate; default preserves the original workflow",
    )
    parser.add_argument("--allow-outside-target", action="store_true", help="Do not reject values outside targetRange")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing numbered CSV files")
    parser.add_argument("--validate-only", action="store_true", help="Validate workbook structure without generating data")
    return parser


def main(argv=None):
    global N_SAMPLES, N_GROUPS, BASE_SEED, ENFORCE_TARGET_RANGE
    args = build_parser().parse_args(argv)
    if args.samples <= 0 or args.groups <= 0:
        raise ValueError("--samples and --groups must be positive")
    N_SAMPLES = args.samples
    N_GROUPS = args.groups
    BASE_SEED = args.seed
    ENFORCE_TARGET_RANGE = not args.allow_outside_target

    input_xlsx = args.workbook.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_xlsx.exists():
        raise FileNotFoundError(input_xlsx)

    headers, rows = load_sheet_rows(input_xlsx)
    col = {name: i for i, name in enumerate(headers)}

    required = [
        "ID",
        args.formula_column,
        "Target",
        "IndependentVars",
        "ParameterRange",
        "FixedConstantValues",
        "paraValue",
        "varRange",
        "targetRange",
    ]
    missing = [name for name in required if name not in col]
    if missing:
        raise ValueError(f"Missing required workbook columns: {missing}")

    if args.validate_only:
        print(f"Validated {len(rows)} benchmark rows in {input_xlsx}")
        return 0

    existing = [
        output_root / safe_folder_name(str(row[col["ID"]]).strip()) / f"{group}.csv"
        for row in rows
        for group in range(N_GROUPS)
        if (output_root / safe_folder_name(str(row[col["ID"]]).strip()) / f"{group}.csv").exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing CSV files; pass --overwrite to replace them"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    summary = []

    for row_no, row in enumerate(rows, start=2):
        row_id = str(row[col["ID"]]).strip()
        target_name = str(row[col["Target"]]).strip()
        independent_vars = split_names(row[col["IndependentVars"]])
        parameter_names = parse_parameter_names(row[col["ParameterRange"]])
        fixed_constants = parse_fixed_constants(row[col["FixedConstantValues"]])

        para_values = parse_json_cell(row[col["paraValue"]], "paraValue")
        var_ranges = parse_json_cell(row[col["varRange"]], "varRange")
        target_ranges = parse_json_cell(row[col["targetRange"]], "targetRange")

        if len(para_values) != N_GROUPS:
            raise ValueError(
                f"{row_id}: paraValue must contain {N_GROUPS} groups, "
                f"got {len(para_values)}"
            )
        if len(var_ranges) != N_GROUPS:
            raise ValueError(
                f"{row_id}: varRange must contain {N_GROUPS} groups, "
                f"got {len(var_ranges)}"
            )
        if len(target_ranges) != N_GROUPS:
            raise ValueError(
                f"{row_id}: targetRange must contain {N_GROUPS} groups, "
                f"got {len(target_ranges)}"
            )

        evaluator = FormulaEvaluator(row[col[args.formula_column]])

        id_dir = output_root / safe_folder_name(row_id)
        id_dir.mkdir(parents=True, exist_ok=True)

        group_acceptance = []

        for group_idx in range(N_GROUPS):
            base_env = build_group_environment(
                parameter_names,
                para_values[group_idx],
                fixed_constants,
            )

            rng = np.random.default_rng(stable_seed(row_id, group_idx))

            data, acceptance = generate_one_group(
                evaluator=evaluator,
                independent_vars=independent_vars,
                variable_ranges=var_ranges[group_idx],
                target_name=target_name,
                target_range=target_ranges[group_idx],
                base_env=base_env,
                rng=rng,
                n_samples=N_SAMPLES,
            )

            csv_path = id_dir / f"{group_idx}.csv"
            header = ",".join(independent_vars + [target_name])
            np.savetxt(
                csv_path,
                data,
                delimiter=",",
                header=header,
                comments="",
                fmt="%.12g",
            )

            group_acceptance.append(acceptance)
            total_files += 1

        summary.append(
            (
                row_id,
                len(independent_vars),
                len(parameter_names),
                min(group_acceptance),
                max(group_acceptance),
            )
        )

        print(
            f"[{row_no:>3}] {row_id}: "
            f"generated {N_GROUPS} CSVs x {N_SAMPLES} rows; "
            f"acceptance={min(group_acceptance):.3f}..{max(group_acceptance):.3f}"
        )

    print()
    print(f"Done. IDs: {len(summary)}")
    print(f"CSV files: {total_files}")
    print(f"Rows per CSV: {N_SAMPLES}")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
