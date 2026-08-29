#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively add per-file equation IDs and mathematical-redundancy references to
PySR summary CSV files.

Default target filenames:
    pysr_warm_0_7_summary.csv
    pysr_warm_0_7_summary

Every matched file is processed independently:
1. Sort files deterministically by their relative paths.
2. Assign eq_id = 1, 2, 3, ... within each individual file.
3. Parse and simplify different directories concurrently with SymPy.
4. Compare each formula only with earlier formulas from the same file.
5. If a row is mathematically equivalent to an earlier same-file
   representative, write that file-local minimum eq_id into redundant.
6. Write redundant=0 for the first representative and for unique formulas.

Example: if eq_id 1, 11, and 20 are equivalent, their redundant values are
0, 1, and 1 respectively.

The program loads and validates every matched file before writing anything.
Each modified CSV is written through a temporary file and atomically replaced.
Use --dry-run to inspect counts without changing files and --backup to create
<filename>.bak copies before replacement. Use --workers N to control parallel
directory preprocessing; --workers 0 automatically uses at most four workers.
Every expensive SymPy operation runs in an isolated persistent process. A task
that exceeds --simplify-timeout or --compare-timeout is terminated, recorded as
a conservative non-match, and replaced so the remaining formulas can continue.

Incremental runs are the default. Files containing valid per-file eq_id and
redundant columns are retained byte-for-byte and are not included in any later
comparison. A file missing either marker column is treated as unprocessed and
both columns are rebuilt automatically. Use --reprocess to rebuild every
matched file independently.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import io
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any, Callable

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: SymPy. Install it with: python -m pip install sympy"
    ) from exc


DEFAULT_FILENAMES = (
    "pysr_warm_0_7_summary.csv",
)

# Default behavior: skip files that already contain valid eq_id and redundant.
# Set to True only when you intentionally want to recompute every matched file.
REPROCESS_COMPLETED = False

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
WHITESPACE_RE = re.compile(r"\s+")

SYMPY_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "arcsin": sp.asin,
    "acos": sp.acos,
    "arccos": sp.acos,
    "atan": sp.atan,
    "arctan": sp.atan,
    "atanh": sp.atanh,
    "arctanh": sp.atanh,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}

TRANSFORMATIONS = standard_transformations + (convert_xor,)


@dataclass
class CsvDialect:
    delimiter: str = ","
    quotechar: str = '"'
    escapechar: str | None = None
    doublequote: bool = True
    skipinitialspace: bool = False
    quoting: int = csv.QUOTE_MINIMAL


@dataclass
class CsvDocument:
    path: Path
    relative_path: str
    headers: list[str]
    rows: list[list[str]]
    formula_index: int
    encoding: str
    newline: str
    dialect: CsvDialect
    processed: bool = False
    row_refs: list["RowReference"] = field(default_factory=list)


@dataclass
class FormulaInfo:
    expression: sp.Expr
    simplified: sp.Expr
    symbol_key: tuple[str, ...]
    exact_key: str


@dataclass(frozen=True)
class FormulaParseFailure:
    error_type: str
    message: str
    timed_out: bool = False


@dataclass(frozen=True)
class SympyTask:
    task_id: int
    kind: str
    payload: tuple[Any, ...]
    label: str


@dataclass
class WorkerSlot:
    slot_id: int
    process: Any
    connection: Any
    task: SympyTask | None = None
    started_at: float = 0.0


@dataclass
class RowReference:
    document: CsvDocument
    row_index: int
    formula_text: str
    eq_id: int = 0
    redundant: int = 0


@dataclass
class MarkingStats:
    rows: int = 0
    representatives: int = 0
    redundant_rows: int = 0
    parse_failures: int = 0
    comparison_failures: int = 0


class ProgressReporter:
    """Dependency-free terminal progress display with redirected-log support."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stream = sys.stderr
        self.is_terminal = self.stream.isatty()
        self.started_at = time.monotonic()
        self.last_refresh = 0.0
        self.last_line_length = 0
        self.last_plain_value: dict[str, int] = {}

    def update(
        self,
        stage: str,
        current: int,
        total: int,
        detail: str = "",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return

        total = max(total, 1)
        current = min(max(current, 0), total)
        now = time.monotonic()

        if self.is_terminal:
            if not force and current < total and now - self.last_refresh < 0.10:
                return
        else:
            # When output is redirected, print at most about 20 progress lines
            # per stage instead of emitting terminal carriage-return updates.
            step = max(1, total // 20)
            previous = self.last_plain_value.get(stage, -step)
            if not force and current < total and current - previous < step:
                return
            self.last_plain_value[stage] = current

        self.last_refresh = now

        ratio = current / total
        elapsed = now - self.started_at
        if self.is_terminal:
            width = 28
            completed = min(width, int(ratio * width))
            bar = "#" * completed + "-" * (width - completed)
            line = (
                f"{stage} [{bar}] {ratio * 100:6.2f}% "
                f"{current}/{total}{detail} | elapsed={elapsed:.1f}s"
            )
            padding = " " * max(0, self.last_line_length - len(line))
            end = "\n" if current >= total else ""
            print(f"\r{line}{padding}", end=end, file=self.stream, flush=True)
            self.last_line_length = 0 if end else len(line)
        else:
            print(
                f"{stage}: {ratio * 100:6.2f}% ({current}/{total})"
                f"{detail} | elapsed={elapsed:.1f}s",
                file=self.stream,
                flush=True,
            )

    def message(self, value: str) -> None:
        """Print a warning without leaving a partially drawn progress line."""
        if self.is_terminal and self.last_line_length:
            print(file=self.stream, flush=True)
            self.last_line_length = 0
        print(value, file=self.stream, flush=True)


def canonical_text(value: object) -> str:
    """Fallback key used only when SymPy cannot parse a formula."""
    text = "" if value is None else str(value).strip()
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    return WHITESPACE_RE.sub("", text).replace("^", "**")


def parse_equation(value: object) -> sp.Expr:
    """Parse one para_eq expression while treating feature names as symbols."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("empty para_eq")
    if "=" in text:
        text = text.split("=", 1)[1].strip()

    # lambda is a Python keyword but may be a legitimate fitted symbol.
    text = re.sub(r"\blambda\b", "lambda_", text).replace("^", "**")

    local_dict: dict[str, Any] = dict(SYMPY_FUNCTIONS)
    local_dict["pi"] = sp.pi

    for token in set(IDENTIFIER_RE.findall(text)):
        if token not in local_dict:
            local_dict[token] = sp.Symbol(token, real=True)

    expression = parse_expr(
        text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )
    if not isinstance(expression, sp.Expr):
        expression = sp.sympify(expression)
    return expression


def prepare_formula(value: object) -> FormulaInfo:
    expression = parse_equation(value)
    simplified = sp.simplify(expression)
    symbol_key = tuple(sorted(str(symbol) for symbol in simplified.free_symbols))
    return FormulaInfo(
        expression=expression,
        simplified=simplified,
        symbol_key=symbol_key,
        exact_key=sp.srepr(simplified),
    )


def mathematically_equivalent(left: sp.Expr, right: sp.Expr) -> bool:
    """Follow simplify(left-right)==0, then fall back to equals()."""
    try:
        difference = sp.simplify(left - right)
        if difference == 0 or difference.is_zero is True:
            return True
    except Exception:
        pass

    try:
        return left.equals(right) is True
    except Exception:
        return False


def sympy_worker_loop(connection: Any) -> None:
    """Persistent isolated worker used for interruptible SymPy operations."""
    try:
        connection.send(("ready", os.getpid()))
        while True:
            message = connection.recv()
            if message is None:
                return

            task_id, kind, payload = message
            try:
                if kind == "prepare":
                    result: Any = prepare_formula(payload[0])
                elif kind == "equivalent":
                    result = mathematically_equivalent(payload[0], payload[1])
                else:
                    raise ValueError(f"Unknown SymPy task kind: {kind}")
            except BaseException as exc:
                result = FormulaParseFailure(
                    error_type=type(exc).__name__,
                    message=str(exc),
                )

            connection.send(("result", task_id, result))
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        try:
            connection.close()
        except OSError:
            pass


class TimedSympyPool:
    """Small persistent process pool that can kill one timed-out task safely."""

    def __init__(
        self,
        worker_count: int,
        startup_timeout: float = 90.0,
    ) -> None:
        self.worker_count = worker_count
        self.startup_timeout = startup_timeout
        self.context = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
        self.slots: dict[int, WorkerSlot] = {}

    def __enter__(self) -> "TimedSympyPool":
        created = [self._create_slot(slot_id) for slot_id in range(self.worker_count)]
        deadline = time.monotonic() + self.startup_timeout
        try:
            for slot in created:
                self._wait_until_ready(slot, deadline)
                self.slots[slot.slot_id] = slot
        except Exception:
            for slot in created:
                self._stop_slot(slot)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _create_slot(self, slot_id: int) -> WorkerSlot:
        parent_connection, child_connection = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=sympy_worker_loop,
            args=(child_connection,),
            name=f"sympy-worker-{slot_id}",
            daemon=True,
        )
        process.start()
        child_connection.close()
        return WorkerSlot(
            slot_id=slot_id,
            process=process,
            connection=parent_connection,
        )

    def _wait_until_ready(self, slot: WorkerSlot, deadline: float) -> None:
        remaining = max(0.0, deadline - time.monotonic())
        if not slot.connection.poll(remaining):
            raise TimeoutError(
                f"SymPy worker {slot.slot_id} did not start within "
                f"{self.startup_timeout:.1f}s"
            )
        message = slot.connection.recv()
        if not message or message[0] != "ready":
            raise RuntimeError(
                f"SymPy worker {slot.slot_id} returned invalid startup message"
            )

    def _restart_slot(self, slot_id: int) -> WorkerSlot:
        old_slot = self.slots.pop(slot_id, None)
        if old_slot is not None:
            self._stop_slot(old_slot)
        new_slot = self._create_slot(slot_id)
        self._wait_until_ready(
            new_slot,
            time.monotonic() + self.startup_timeout,
        )
        self.slots[slot_id] = new_slot
        return new_slot

    @staticmethod
    def _stop_slot(slot: WorkerSlot) -> None:
        try:
            if slot.process.is_alive():
                slot.process.terminate()
            slot.process.join(timeout=2.0)
            if slot.process.is_alive() and hasattr(slot.process, "kill"):
                slot.process.kill()
                slot.process.join(timeout=2.0)
        finally:
            try:
                slot.connection.close()
            except OSError:
                pass

    @staticmethod
    def _assign(slot: WorkerSlot, task: SympyTask) -> None:
        slot.connection.send((task.task_id, task.kind, task.payload))
        slot.task = task
        slot.started_at = time.monotonic()

    def run_tasks(
        self,
        tasks: list[SympyTask],
        timeout: float,
        on_complete: Callable[
            [int, int, SympyTask, Any],
            None,
        ] | None = None,
    ) -> dict[int, Any]:
        """Run tasks with a timeout measured from assignment to a worker."""
        if not tasks:
            return {}

        pending = deque(tasks)
        results: dict[int, Any] = {}
        total = len(tasks)
        completed = 0

        for slot in self.slots.values():
            if pending:
                self._assign(slot, pending.popleft())

        while completed < total:
            active_slots = [
                slot for slot in self.slots.values() if slot.task is not None
            ]
            if not active_slots:
                raise RuntimeError("SymPy worker pool has pending tasks but no workers")

            connection_to_slot = {
                slot.connection.fileno(): slot for slot in active_slots
            }
            ready_connections = wait_for_connections(
                [slot.connection for slot in active_slots],
                timeout=0.10,
            )

            for connection in ready_connections:
                slot = connection_to_slot.get(connection.fileno())
                if slot is None or slot.task is None:
                    continue
                task = slot.task
                try:
                    message = connection.recv()
                    if (
                        not message
                        or message[0] != "result"
                        or message[1] != task.task_id
                    ):
                        raise RuntimeError("Invalid result from SymPy worker")
                    result = message[2]
                except (EOFError, BrokenPipeError, OSError, RuntimeError) as exc:
                    result = FormulaParseFailure(
                        error_type="WorkerCrashed",
                        message=str(exc),
                    )

                results[task.task_id] = result
                completed += 1
                slot.task = None
                if on_complete is not None:
                    on_complete(completed, total, task, result)

                if isinstance(result, FormulaParseFailure) and (
                    result.error_type == "WorkerCrashed"
                ):
                    slot = self._restart_slot(slot.slot_id)

                if pending:
                    self._assign(slot, pending.popleft())

            now = time.monotonic()
            for slot in list(self.slots.values()):
                task = slot.task
                if task is None:
                    continue

                worker_died = not slot.process.is_alive()
                timed_out = timeout > 0 and now - slot.started_at >= timeout
                if not worker_died and not timed_out:
                    continue

                if timed_out:
                    result = FormulaParseFailure(
                        error_type=f"{task.kind.title()}Timeout",
                        message=f"exceeded {timeout:.2f} seconds",
                        timed_out=True,
                    )
                else:
                    result = FormulaParseFailure(
                        error_type="WorkerCrashed",
                        message="worker exited before returning a result",
                    )

                results[task.task_id] = result
                completed += 1
                if on_complete is not None:
                    on_complete(completed, total, task, result)

                replacement = self._restart_slot(slot.slot_id)
                if pending:
                    self._assign(replacement, pending.popleft())

        return results

    def close(self) -> None:
        slots = list(self.slots.values())
        self.slots.clear()
        for slot in slots:
            try:
                if slot.process.is_alive():
                    slot.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for slot in slots:
            try:
                slot.process.join(timeout=2.0)
            finally:
                if slot.process.is_alive():
                    self._stop_slot(slot)
                else:
                    try:
                        slot.connection.close()
                    except OSError:
                        pass


def decode_csv(raw: bytes, path: Path) -> tuple[str, str]:
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            return raw.decode("gb18030"), "gb18030"
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot decode CSV: {path}") from exc


def detect_dialect(text: str) -> CsvDialect:
    sample = text[:65536]
    try:
        detected = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return CsvDialect(
            delimiter=detected.delimiter,
            quotechar=detected.quotechar or '"',
            escapechar=detected.escapechar,
            doublequote=detected.doublequote,
            skipinitialspace=detected.skipinitialspace,
            quoting=detected.quoting,
        )
    except csv.Error:
        return CsvDialect()


def load_document(
    root: Path,
    path: Path,
    formula_column: str,
    reprocess: bool = False,
) -> CsvDocument:
    raw = path.read_bytes()
    text, encoding = decode_csv(raw, path)
    newline = "\r\n" if "\r\n" in text else "\n"
    dialect = detect_dialect(text)

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=dialect.delimiter,
        quotechar=dialect.quotechar,
        escapechar=dialect.escapechar,
        doublequote=dialect.doublequote,
        skipinitialspace=dialect.skipinitialspace,
        quoting=dialect.quoting,
    )
    table = list(reader)
    if not table:
        raise ValueError(f"Empty CSV: {path}")

    original_headers = [str(value) for value in table[0]]
    normalized_headers = [value.strip() for value in original_headers]
    if formula_column not in normalized_headers:
        raise ValueError(f"{path}: missing column {formula_column!r}")

    has_eq_id = "eq_id" in normalized_headers
    has_redundant = "redundant" in normalized_headers
    if normalized_headers.count("eq_id") > 1 or normalized_headers.count(
        "redundant"
    ) > 1:
        raise ValueError(
            f"{path}: duplicate eq_id/redundant marker columns found"
        )
    # --reprocess must take effect during loading, before existing marker cells
    # are parsed. A partial marker state is never a valid processed file: any
    # existing marker column is removed from the in-memory table and both
    # columns are rebuilt from scratch automatically.
    processed = has_eq_id and has_redundant and not reprocess
    eq_id_index = normalized_headers.index("eq_id") if processed else -1
    redundant_index = (
        normalized_headers.index("redundant") if processed else -1
    )

    # Re-running the program replaces old marking columns instead of creating
    # duplicate eq_id/redundant headers.
    keep_indices = [
        index
        for index, header in enumerate(normalized_headers)
        if header not in {"eq_id", "redundant"}
    ]
    headers = [original_headers[index] for index in keep_indices]
    normalized_kept = [header.strip() for header in headers]
    formula_index = normalized_kept.index(formula_column)

    rows: list[list[str]] = []
    existing_markers: list[tuple[int, int]] = []
    for csv_row, source_row_number in zip(table[1:], range(2, len(table) + 1)):
        if not csv_row or not any(str(value).strip() for value in csv_row):
            continue
        if len(csv_row) > len(original_headers):
            raise ValueError(
                f"{path}:{source_row_number}: row has {len(csv_row)} fields, "
                f"header has {len(original_headers)}"
            )
        padded = list(csv_row) + [""] * (len(original_headers) - len(csv_row))
        if processed:
            marker_values: list[int] = []
            for marker_name, marker_index in (
                ("eq_id", eq_id_index),
                ("redundant", redundant_index),
            ):
                raw_value = str(padded[marker_index]).strip()
                try:
                    numeric_value = float(raw_value)
                    integer_value = int(numeric_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{path}:{source_row_number}: invalid {marker_name}="
                        f"{raw_value!r}"
                    ) from exc
                if (
                    not math.isfinite(numeric_value)
                    or numeric_value != integer_value
                ):
                    raise ValueError(
                        f"{path}:{source_row_number}: {marker_name} must be "
                        f"an integer, got {raw_value!r}"
                    )
                marker_values.append(integer_value)
            eq_id, redundant = marker_values
            if eq_id <= 0 or redundant < 0:
                raise ValueError(
                    f"{path}:{source_row_number}: processed rows require "
                    "eq_id>0 and redundant>=0"
                )
            existing_markers.append((eq_id, redundant))
        rows.append([padded[index] for index in keep_indices])

    relative_path = path.relative_to(root).as_posix()
    document = CsvDocument(
        path=path,
        relative_path=relative_path,
        headers=headers,
        rows=rows,
        formula_index=formula_index,
        encoding=encoding,
        newline=newline,
        dialect=dialect,
        processed=processed,
    )
    document.row_refs = [
        RowReference(
            document=document,
            row_index=index,
            formula_text=row[formula_index],
            eq_id=(existing_markers[index][0] if processed else 0),
            redundant=(existing_markers[index][1] if processed else 0),
        )
        for index, row in enumerate(rows)
    ]
    return document


def validate_processed_documents(
    documents: list[CsvDocument],
) -> None:
    """Validate that every retained file uses independent local IDs."""
    processed_documents = [document for document in documents if document.processed]

    def validate_group(row_refs: list[RowReference], label: str) -> None:
        actual_ids = [row_ref.eq_id for row_ref in row_refs]
        expected_ids = list(range(1, len(row_refs) + 1))
        if actual_ids != expected_ids:
            raise ValueError(
                f"{label}: processed eq_id values must be 1..N in row order "
                "inside each file. Remove the invalid marker columns or use "
                "--reprocess to rebuild this file."
            )

        rows_by_id: dict[int, RowReference] = {}
        for row_ref in row_refs:
            if row_ref.eq_id in rows_by_id:
                raise ValueError(
                    f"{label}: duplicate processed eq_id={row_ref.eq_id}; "
                    "run once with --reprocess"
                )
            rows_by_id[row_ref.eq_id] = row_ref

        for row_ref in row_refs:
            redundant = row_ref.redundant
            if redundant == 0:
                continue
            if redundant >= row_ref.eq_id:
                raise ValueError(
                    f"{label}: eq_id={row_ref.eq_id} has invalid redundant="
                    f"{redundant}; it must point to a smaller eq_id"
                )
            representative = rows_by_id.get(redundant)
            if representative is None:
                raise ValueError(
                    f"{label}: eq_id={row_ref.eq_id} points to missing "
                    f"representative eq_id={redundant}"
                )
            if representative.redundant != 0:
                raise ValueError(
                    f"{label}: eq_id={row_ref.eq_id} points to non-minimal "
                    f"representative eq_id={redundant}"
                )

    for document in processed_documents:
        validate_group(document.row_refs, document.relative_path)


def document_formula_items(
    document: CsvDocument,
    row_refs: list[RowReference] | None = None,
) -> list[tuple[str, str]]:
    """Return unique canonical formula keys for one summary-file directory."""
    unique: dict[str, str] = {}
    selected_rows = document.row_refs if row_refs is None else row_refs
    for row_ref in selected_rows:
        text_key = canonical_text(row_ref.formula_text)
        if text_key and text_key not in unique:
            unique[text_key] = row_ref.formula_text
    return list(unique.items())


def resolve_worker_count(requested: int, task_count: int) -> int:
    if requested < 0:
        raise ValueError("--workers must be 0 or a positive integer")
    if task_count <= 1:
        return 1
    if requested == 0:
        # SymPy processes can use substantial memory. Four is a conservative
        # automatic default; users can explicitly request more.
        requested = min(os.cpu_count() or 1, 4)
    return max(1, min(requested, task_count))


def preprocess_documents(
    documents: list[CsvDocument],
    worker_count: int,
    progress: ProgressReporter,
    worker_pool: TimedSympyPool,
    simplify_timeout: float,
    rows_to_prepare: dict[str, list[RowReference]] | None = None,
) -> dict[str, dict[str, FormulaInfo | FormulaParseFailure]]:
    """Parse/simplify unique formulas with killable per-formula timeouts."""
    first_occurrences: dict[str, tuple[str, str]] = {}
    for document in documents:
        selected_rows = (
            document.row_refs
            if rows_to_prepare is None
            else rows_to_prepare.get(document.relative_path, [])
        )
        for text_key, formula_text in document_formula_items(
            document,
            selected_rows,
        ):
            first_occurrences.setdefault(
                text_key,
                (formula_text, document.relative_path),
            )

    tasks: list[SympyTask] = []
    key_by_task_id: dict[int, str] = {}
    for task_id, (text_key, item) in enumerate(first_occurrences.items()):
        formula_text, relative_path = item
        key_by_task_id[task_id] = text_key
        label_formula = WHITESPACE_RE.sub(" ", formula_text.strip())
        if len(label_formula) > 120:
            label_formula = label_formula[:117] + "..."
        tasks.append(
            SympyTask(
                task_id=task_id,
                kind="prepare",
                payload=(formula_text,),
                label=f"{relative_path} | {label_formula}",
            )
        )

    total = len(tasks)
    failure_count = 0
    timeout_count = 0
    progress.update(
        "[2/4] Simplifying",
        0,
        total,
        detail=(
            f" | workers={worker_count} failures=0 timeouts=0"
        ),
        force=True,
    )

    def report_completion(
        completed: int,
        task_total: int,
        task: SympyTask,
        result: Any,
    ) -> None:
        nonlocal failure_count, timeout_count
        if isinstance(result, FormulaParseFailure):
            failure_count += 1
            if result.timed_out:
                timeout_count += 1
                progress.message(
                    "WARNING: formula simplification timed out; using exact-text "
                    f"fallback: {task.label}"
                )
        progress.update(
            "[2/4] Simplifying",
            completed,
            task_total,
            detail=(
                f" | workers={worker_count} failures={failure_count}"
                f" timeouts={timeout_count} current={task.label}"
            ),
            force=completed == task_total,
        )

    task_results = worker_pool.run_tasks(
        tasks,
        timeout=simplify_timeout,
        on_complete=report_completion,
    )
    shared_prepared = {
        key_by_task_id[task_id]: result
        for task_id, result in task_results.items()
    }

    caches: dict[str, dict[str, FormulaInfo | FormulaParseFailure]] = {}
    for document in documents:
        document_cache: dict[str, FormulaInfo | FormulaParseFailure] = {}
        selected_rows = (
            document.row_refs
            if rows_to_prepare is None
            else rows_to_prepare.get(document.relative_path, [])
        )
        for text_key, _ in document_formula_items(document, selected_rows):
            document_cache[text_key] = shared_prepared[text_key]
        caches[document.relative_path] = document_cache

    return caches


def mark_group(
    row_refs: list[RowReference],
    start_id: int = 1,
    prepared_cache: dict[
        str, FormulaInfo | FormulaParseFailure
    ] | None = None,
    worker_pool: TimedSympyPool | None = None,
    compare_timeout: float = 10.0,
    compare_batch_size: int = 0,
    progress: ProgressReporter | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    base_redundant: int = 0,
    base_parse_failures: int = 0,
    base_comparison_failures: int = 0,
) -> MarkingStats:
    stats = MarkingStats(rows=len(row_refs))
    parse_cache: dict[str, FormulaInfo | FormulaParseFailure] = dict(
        prepared_cache or {}
    )
    exact_representatives: dict[tuple[tuple[str, ...], str], RowReference] = {}
    representatives: dict[tuple[str, ...], list[tuple[RowReference, sp.Expr]]] = {}
    fallback_representatives: dict[str, RowReference] = {}

    overall_total = progress_total if progress_total is not None else len(row_refs)
    if progress is not None:
        progress.update(
            "[3/4] Comparing",
            progress_offset,
            overall_total,
            detail=(
                f" | redundant={base_redundant}"
                f" parse_failures={base_parse_failures}"
                f" compare_failures={base_comparison_failures}"
            ),
            force=progress_offset == 0,
        )

    def report_progress(local_index: int) -> None:
        if progress is None:
            return
        overall_current = progress_offset + local_index
        progress.update(
            "[3/4] Comparing",
            overall_current,
            overall_total,
            detail=(
                f" | redundant={base_redundant + stats.redundant_rows}"
                " parse_failures="
                f"{base_parse_failures + stats.parse_failures}"
                " compare_failures="
                f"{base_comparison_failures + stats.comparison_failures}"
            ),
            force=overall_current >= overall_total,
        )

    next_id = start_id
    for local_index, row_ref in enumerate(row_refs, start=1):
        row_ref.eq_id = next_id
        row_ref.redundant = 0
        next_id += 1

        text_key = canonical_text(row_ref.formula_text)
        if not text_key:
            stats.representatives += 1
            report_progress(local_index)
            continue

        if text_key not in parse_cache:
            parse_cache[text_key] = FormulaParseFailure(
                error_type="MissingPreparedFormula",
                message="formula was not present in the preprocessing cache",
            )

        formula_info = parse_cache[text_key]
        if isinstance(formula_info, FormulaParseFailure):
            stats.parse_failures += 1
            representative = fallback_representatives.get(text_key)
            if representative is None:
                fallback_representatives[text_key] = row_ref
                stats.representatives += 1
            else:
                row_ref.redundant = representative.eq_id
                stats.redundant_rows += 1
            report_progress(local_index)
            continue

        exact_key = (formula_info.symbol_key, formula_info.exact_key)
        exact_representative = exact_representatives.get(exact_key)
        if exact_representative is not None:
            row_ref.redundant = exact_representative.eq_id
            stats.redundant_rows += 1
            report_progress(local_index)
            continue

        equivalent_representative: RowReference | None = None
        candidate_representatives = representatives.get(
            formula_info.symbol_key,
            [],
        )
        if worker_pool is None:
            for representative, representative_expr in candidate_representatives:
                if mathematically_equivalent(
                    formula_info.simplified,
                    representative_expr,
                ):
                    equivalent_representative = representative
                    break
        elif candidate_representatives:
            # Pure SymPy equivalence checking only. No numeric sampling, no
            # heuristic similarity score, and no non-SymPy prefilter is used.
            #
            # Performance optimization: compare representatives in small
            # batches instead of submitting every candidate at once. As soon as
            # SymPy proves one representative equivalent, stop immediately so
            # later candidates are never compared unnecessarily.
            batch_size = compare_batch_size
            if batch_size <= 0:
                batch_size = max(1, worker_pool.worker_count)

            for batch_start in range(0, len(candidate_representatives), batch_size):
                batch = candidate_representatives[
                    batch_start : batch_start + batch_size
                ]
                comparison_tasks = [
                    SympyTask(
                        task_id=position,
                        kind="equivalent",
                        payload=(formula_info.simplified, representative_expr),
                        label=(
                            f"eq_id={row_ref.eq_id} vs "
                            f"eq_id={representative.eq_id}"
                        ),
                    )
                    for position, (representative, representative_expr) in enumerate(batch)
                ]
                comparison_results = worker_pool.run_tasks(
                    comparison_tasks,
                    timeout=compare_timeout,
                )

                for position, (representative, _) in enumerate(batch):
                    result = comparison_results[position]
                    if isinstance(result, FormulaParseFailure):
                        stats.comparison_failures += 1
                        if result.timed_out and progress is not None:
                            progress.message(
                                "WARNING: formula-equivalence comparison timed out; "
                                "treating this pair as not proven equivalent: "
                                f"eq_id={row_ref.eq_id} vs "
                                f"eq_id={representative.eq_id}"
                            )
                        continue
                    if result is True:
                        equivalent_representative = representative
                        break

                if equivalent_representative is not None:
                    break

        if equivalent_representative is not None:
            row_ref.redundant = equivalent_representative.eq_id
            stats.redundant_rows += 1
            # Cache this canonical form so later identical forms immediately
            # point to the same minimum representative.
            exact_representatives[exact_key] = equivalent_representative
        else:
            representatives.setdefault(formula_info.symbol_key, []).append(
                (row_ref, formula_info.simplified)
            )
            exact_representatives[exact_key] = row_ref
            stats.representatives += 1

        report_progress(local_index)

    return stats


def updated_table(document: CsvDocument) -> tuple[list[str], list[list[str]]]:
    insert_at = document.formula_index + 1
    headers = (
        document.headers[:insert_at]
        + ["eq_id", "redundant"]
        + document.headers[insert_at:]
    )
    rows = []
    for row_ref in document.row_refs:
        row = document.rows[row_ref.row_index]
        rows.append(
            row[:insert_at]
            + [str(row_ref.eq_id), str(row_ref.redundant)]
            + row[insert_at:]
        )
    return headers, rows


def write_document(document: CsvDocument, backup: bool) -> None:
    headers, rows = updated_table(document)
    if backup:
        shutil.copy2(document.path, document.path.with_name(document.path.name + ".bak"))

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{document.path.name}.",
        suffix=".tmp",
        dir=document.path.parent,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding=document.encoding,
            newline="",
        ) as handle:
            writer = csv.writer(
                handle,
                delimiter=document.dialect.delimiter,
                quotechar=document.dialect.quotechar,
                escapechar=document.dialect.escapechar,
                doublequote=document.dialect.doublequote,
                skipinitialspace=document.dialect.skipinitialspace,
                quoting=document.dialect.quoting,
                lineterminator=document.newline,
            )
            writer.writerow(headers)
            writer.writerows(rows)
        os.replace(temp_name, document.path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def discover_files(root: Path, filenames: tuple[str, ...]) -> list[Path]:
    wanted = {name.casefold() for name in filenames}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold() in wanted
    ]
    return sorted(
        files,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign eq_id and mark mathematically equivalent para_eq rows with "
            "the minimum representative eq_id."
        )
    )
    parser.add_argument("root", type=Path, help="Root directory to scan recursively.")
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help=(
            "Exact target filename; repeat for multiple names. Defaults to "
            "pysr_warm_0_7_summary.csv and the extensionless variant."
        ),
    )
    parser.add_argument(
        "--formula-column",
        default="para_eq",
        help="Formula column name (default: para_eq).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report without changing CSV files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create <filename>.bak before replacing each CSV.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=REPROCESS_COMPLETED,
        help=(
            "Recompute every matched file even when valid eq_id and redundant "
            "columns already exist. Default: OFF, so completed files are skipped."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Isolated SymPy worker processes. 0 selects automatically up to "
            "4 (default); 1 uses one killable worker process."
        ),
    )
    parser.add_argument(
        "--simplify-timeout",
        type=float,
        default=10.0,
        help=(
            "Maximum seconds for one formula parse/simplify task before its "
            "worker is terminated and replaced (default: 10; 0 disables)."
        ),
    )
    parser.add_argument(
        "--compare-timeout",
        type=float,
        default=5.0,
        help=(
            "Maximum seconds for one nontrivial formula-equivalence check "
            "before its worker is terminated and replaced (default: 5; "
            "0 disables)."
        ),
    )
    parser.add_argument(
        "--compare-batch-size",
        type=int,
        default=0,
        help=(
            "Number of SymPy equivalence comparisons submitted at once. "
            "0 uses the active worker count (default). Smaller batches can "
            "stop earlier after a match and avoid unnecessary comparisons."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable loading, simplification, comparison, and writing progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simplify_timeout < 0:
        raise ValueError("--simplify-timeout must be 0 or greater")
    if args.compare_timeout < 0:
        raise ValueError("--compare-timeout must be 0 or greater")
    if args.compare_batch_size < 0:
        raise ValueError("--compare-batch-size must be 0 or greater")

    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    filenames = tuple(args.filenames or DEFAULT_FILENAMES)
    files = discover_files(root, filenames)
    if not files:
        raise FileNotFoundError(
            f"No matching summary files under {root}; filenames={filenames}"
        )

    # Print effective parameters BEFORE loading/simplifying/comparing any CSV.
    # Use stderr so the configuration appears in the same stream as progress
    # messages and is visible immediately even when stdout is buffered.
    requested_workers = "AUTO (up to 4)" if args.workers == 0 else str(args.workers)
    cfg = [
        "=" * 80,
        "RUN CONFIGURATION (printed before processing)",
        "=" * 80,
        f"Root directory          : {root}",
        f"Target filenames        : {filenames}",
        f"Matched files found     : {len(files)}",
        f"Formula column          : {args.formula_column}",
        "Comparison scope        : within each file only",
        (
            "Reprocess completed    : "
            + (
                "YES -- recompute completed files"
                if args.reprocess
                else "NO -- SKIP completed files (DEFAULT)"
            )
        ),
        f"Worker setting          : {requested_workers}",
        f"Simplify timeout        : {args.simplify_timeout:g} s",
        f"Compare timeout         : {args.compare_timeout:g} s",
        "Compare method          : SymPy only (simplify(left-right), then equals)",
        (
            "Compare batch size      : "
            + (
                "AUTO (= active workers)"
                if args.compare_batch_size == 0
                else str(args.compare_batch_size)
            )
        ),
        f"Dry run                 : {'YES' if args.dry_run else 'NO'}",
        f"Backup before overwrite : {'YES' if args.backup else 'NO'}",
        f"Progress display        : {'OFF' if args.no_progress else 'ON'}",
        "=" * 80,
    ]
    print("\n".join(cfg), file=sys.stderr, flush=True)

    progress = ProgressReporter(enabled=not args.no_progress)

    # Load and validate every file before any write occurs.
    documents: list[CsvDocument] = []
    progress.update("[1/4] Loading", 0, len(files), force=True)
    for file_number, path in enumerate(files, start=1):
        documents.append(
            load_document(
                root,
                path,
                args.formula_column,
                reprocess=args.reprocess,
            )
        )
        progress.update(
            "[1/4] Loading",
            file_number,
            len(files),
            detail=f" | file={path.relative_to(root).as_posix()}",
            force=file_number == len(files),
        )

    validate_processed_documents(documents)
    processed_documents = [
        document for document in documents if document.processed
    ]
    new_documents = [
        document for document in documents if not document.processed
    ]

    # Show exactly which files are skipped and which will be processed BEFORE
    # any SymPy simplification starts. This removes ambiguity when multiple
    # similarly named summary files exist under the root.
    print("\n" + "=" * 80, file=sys.stderr, flush=True)
    print("FILE STATUS (before simplification)", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)
    for document in documents:
        status = "SKIP (completed)" if document.processed else "PROCESS (missing/invalid markers or --reprocess)"
        print(f"[{status}] {document.relative_path}", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)

    retained_rows = sum(
        len(document.row_refs) for document in processed_documents
    )
    total_rows = sum(len(document.row_refs) for document in new_documents)
    total_stats = MarkingStats()
    worker_count = 0

    if new_documents:
        worker_count = resolve_worker_count(
            args.workers,
            len(new_documents),
        )

    plan = [
        "=" * 80,
        "EXECUTION PLAN (before SymPy work)",
        "=" * 80,
        f"Completed files skipped : {len(processed_documents)}",
        f"Files to process        : {len(new_documents)}",
        f"Existing rows retained  : {retained_rows}",
        f"Rows to process         : {total_rows}",
        f"Actual worker processes : {worker_count}",
        f"Will write files        : {'NO (--dry-run)' if args.dry_run else ('YES' if new_documents else 'NO (nothing new)')}",
        "=" * 80,
    ]
    print("\n".join(plan), file=sys.stderr, flush=True)

    if new_documents:
        with TimedSympyPool(worker_count) as worker_pool:
            prepared_by_document = preprocess_documents(
                new_documents,
                worker_count,
                progress,
                worker_pool,
                args.simplify_timeout,
            )

            processed_rows = 0
            for document in new_documents:
                stats = mark_group(
                    document.row_refs,
                    start_id=1,
                    prepared_cache=prepared_by_document[
                        document.relative_path
                    ],
                    worker_pool=worker_pool,
                    compare_timeout=args.compare_timeout,
                    compare_batch_size=args.compare_batch_size,
                    progress=progress,
                    progress_offset=processed_rows,
                    progress_total=total_rows,
                    base_redundant=total_stats.redundant_rows,
                    base_parse_failures=total_stats.parse_failures,
                    base_comparison_failures=(
                        total_stats.comparison_failures
                    ),
                )
                total_stats.rows += stats.rows
                total_stats.representatives += stats.representatives
                total_stats.redundant_rows += stats.redundant_rows
                total_stats.parse_failures += stats.parse_failures
                total_stats.comparison_failures += stats.comparison_failures
                processed_rows += stats.rows
    else:
        progress.message(
            "All matched summary files already contain valid eq_id and "
            "redundant columns; nothing to process."
        )

    if not args.dry_run and new_documents:
        progress.update("[4/4] Writing", 0, len(new_documents), force=True)
        for file_number, document in enumerate(new_documents, start=1):
            write_document(document, backup=args.backup)
            progress.update(
                "[4/4] Writing",
                file_number,
                len(new_documents),
                detail=f" | file={document.relative_path}",
                force=file_number == len(new_documents),
            )
    elif args.dry_run and new_documents and progress.enabled:
        print(
            "[4/4] Writing: skipped (--dry-run)",
            file=sys.stderr,
            flush=True,
        )

    print(f"Root: {root}")
    print("Comparison scope: within each file only")
    print(f"Worker processes: {worker_count}")
    print(f"Matched files: {len(documents)}")
    print(f"Already processed files skipped: {len(processed_documents)}")
    print(f"New files processed: {len(new_documents)}")
    print(f"Existing rows retained: {retained_rows}")
    print(f"New rows assigned eq_id: {total_stats.rows}")
    print(f"Representative/unique rows: {total_stats.representatives}")
    print(f"Redundant rows: {total_stats.redundant_rows}")
    print(f"SymPy parse failures: {total_stats.parse_failures}")
    print(f"SymPy comparison failures/timeouts: {total_stats.comparison_failures}")
    print(
        "Files modified: "
        f"{0 if args.dry_run else len(new_documents)}"
    )


if __name__ == "__main__":
    main()
