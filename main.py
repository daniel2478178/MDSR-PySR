#!/usr/bin/env python3
"""Run the MSSR-PySR experiment pipeline with linked, configurable paths."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
COMMANDS = ("prepare", "discover", "evaluate", "merge", "robustness", "report", "core", "all")


@dataclass(frozen=True)
class Config:
    command: str
    mode: str
    python: Path
    workbook: Path
    master_workbook: Path
    wass_workbook: Path
    structure_results: Path
    data_root: Path
    generated_root: Path
    results_dir: Path
    datasets: str
    formula_column: str
    iterations: int
    outer_processes: int
    pysr_processes: int
    workers: int
    top_n: int
    top_rank: int
    force: bool
    dry_run: bool

    @property
    def candidates(self) -> Path:
        return self.results_dir / "top_shared_formulas.csv"


def path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--mode", choices=("xonly", "with-params"), default="xonly")
    parser.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    parser.add_argument("--workbook", type=path, default=PROJECT_ROOT / "physicsMDSR_Range.xlsx")
    parser.add_argument(
        "--master-workbook", type=path,
        default=PROJECT_ROOT / "physicsMDSR_Range_with_simpFormula.xlsx",
    )
    parser.add_argument(
        "--wass-workbook", type=path,
        default=PROJECT_ROOT / "physicsMDSR_WASS_varRanges.xlsx",
    )
    parser.add_argument(
        "--structure-results", type=path,
        default=PROJECT_ROOT / "physics_formula_structural_similarity_evaluation_withooutparam.xlsx",
    )
    parser.add_argument("--data-root", type=path, default=PROJECT_ROOT / "physicsMDSR_Range_CSV")
    parser.add_argument("--generated-root", type=path, default=PROJECT_ROOT / "generated")
    parser.add_argument("--results-dir", type=path, default=PROJECT_ROOT / "results")
    parser.add_argument("--datasets", default="0-7", help="Dataset indices, e.g. 0-7 or 0,1,2")
    parser.add_argument(
        "--formula-column", choices=("OriginalFormula", "GenerationFormula"),
        default="GenerationFormula",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--outer-processes", type=int, default=3)
    parser.add_argument("--pysr-processes", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-rank", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser


def make_config(args: argparse.Namespace) -> Config:
    numeric = (
        args.iterations, args.outer_processes, args.pysr_processes,
        args.workers, args.top_n, args.top_rank,
    )
    if min(numeric) <= 0:
        raise ValueError("iterations, process counts, top-n, and top-rank must be positive")
    return Config(**vars(args))


def run(config: Config, command: Sequence[object]) -> None:
    resolved = [str(item) for item in command]
    print(f"+ {shlex.join(resolved)}", flush=True)
    if not config.dry_run:
        subprocess.run(resolved, cwd=PROJECT_ROOT, check=True)


def require_file(pathname: Path, *, generated: bool = False, dry_run: bool = False) -> None:
    if pathname.is_file() or (generated and dry_run):
        return
    raise FileNotFoundError(f"Missing required file: {pathname}")


def prepare(config: Config) -> None:
    require_file(config.workbook)
    existing = next(config.data_root.rglob("*.csv"), None) if config.data_root.is_dir() else None
    if existing and not config.force:
        print(f"prepare: existing CSV data found at {config.data_root}; skipping (use --force to replace)")
        return
    command: list[object] = [
        config.python,
        PROJECT_ROOT / "scripts/data/generate_physics_mdsr_csv_openpyxl.py",
        config.workbook,
        config.data_root,
        "--formula-column", config.formula_column,
    ]
    if config.force:
        command.append("--overwrite")
    run(config, command)


def discover(config: Config) -> None:
    require_file(config.workbook)
    if not config.data_root.is_dir() and not config.dry_run:
        raise FileNotFoundError(f"Missing data root: {config.data_root}; run prepare first")
    script = "pysr_xonly.py" if config.mode == "xonly" else "pysr_with_params.py"
    run_id = "warm_0_7_xonly_v2" if config.mode == "xonly" else "warm_0_7_with_params"
    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/discovery" / script,
        config.workbook,
        config.data_root,
        "--datasets", config.datasets,
        "--iterations", config.iterations,
        "--outer-processes", config.outer_processes,
        "--pysr-processes", config.pysr_processes,
        "--top-n", config.top_n,
        "--run-id", run_id,
    ])


def evaluate(config: Config) -> None:
    command: list[object] = [
        config.python,
        PROJECT_ROOT / "scripts/evaluation/evaluate_shared_formulas.py",
        config.data_root,
        "--datasets", config.datasets,
        "--workers", config.workers,
    ]
    if config.force:
        command.append("--force")
    run(config, command)


def merge(config: Config) -> None:
    require_file(config.master_workbook)
    if not config.dry_run:
        config.results_dir.mkdir(parents=True, exist_ok=True)
    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/reporting/merge_top_formulas.py",
        config.data_root,
        config.master_workbook,
        config.candidates,
        "--top-n", config.top_n,
    ])


def generate_wass(config: Config, label: str) -> None:
    destination = config.generated_root / "wass" / label
    existing = next(destination.rglob("*.csv"), None) if destination.is_dir() else None
    if existing and not config.force:
        print(f"robustness: existing {label} data found at {destination}; skipping generation")
        return
    command: list[object] = [
        config.python,
        PROJECT_ROOT / "scripts/data/generate_mdsr_testsets.py",
        config.wass_workbook,
        "--wass", label,
        "--output-root", destination,
    ]
    if config.force:
        command.append("--overwrite")
    run(config, command)


def robustness(config: Config) -> None:
    require_file(config.wass_workbook)
    require_file(config.candidates, generated=True, dry_run=config.dry_run)
    if not config.dry_run:
        (config.generated_root / "wass").mkdir(parents=True, exist_ok=True)
        (config.generated_root / "noise").mkdir(parents=True, exist_ok=True)
        config.results_dir.mkdir(parents=True, exist_ok=True)

    for label in ("WASS_0", "WASS_025", "WASS_04"):
        generate_wass(config, label)
    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/evaluation/fit_wass_metrics.py",
        config.candidates,
        config.generated_root / "wass",
        "--top-rank", config.top_rank,
        "--wass", "WASS_0", "WASS_025", "WASS_04",
        "--workers", config.workers,
        "--output", config.results_dir / "wass_metrics.csv",
    ])

    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/data/add_target_noise.py",
        config.workbook,
        config.data_root,
        config.generated_root / "noise",
    ])
    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/evaluation/fit_noise_toprank_metrics.py",
        config.candidates,
        config.generated_root / "noise",
        "--top-rank", config.top_rank,
        "--noise", "001", "003", "005", "01",
        "--workers", config.workers,
        "--output", config.results_dir / "noise_metrics.csv",
    ])


def report(config: Config) -> None:
    require_file(config.structure_results)
    output_dir = config.results_dir / "plots"
    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    run(config, [
        config.python,
        PROJECT_ROOT / "scripts/reporting/plot_metric_histograms.py",
        config.structure_results,
        "--output-dir", output_dir,
    ])


def execute(config: Config) -> None:
    stages = {
        "prepare": prepare,
        "discover": discover,
        "evaluate": evaluate,
        "merge": merge,
        "robustness": robustness,
        "report": report,
    }
    if config.command == "core":
        selected = ("prepare", "discover", "evaluate", "merge")
    elif config.command == "all":
        selected = ("prepare", "discover", "evaluate", "merge", "robustness", "report")
    else:
        selected = (config.command,)
    for name in selected:
        print(f"\n[{name}]", flush=True)
        stages[name](config)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = make_config(build_parser().parse_args(argv))
        execute(config)
        return 0
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
