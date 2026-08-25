# MSSR-PySR

MSSR-PySR is an experimental pipeline for discovering one symbolic formula that
works across several related physics datasets. PySR proposes candidate
expressions, numeric coefficients are generalized into fitted parameters, and
the resulting structures are ranked by their performance across multiple
datasets.

The repository contains a 59-problem physics benchmark (`P01`–`P59`), generated
CSV data, PySR discovery scripts, shared-formula evaluation, distribution-shift
tests, and reporting utilities.

## Pipeline overview

```text
physics metadata workbook
        │
        ▼
synthetic datasets (16 groups per physics problem)
        │
        ▼
PySR candidate discovery on groups 0–7
        │
        ▼
replace numeric coefficients with p0, p1, ...
        │
        ▼
fit each candidate independently across groups 0–7
        │
        ▼
remove equivalent formulas and rank by shared-fit error
        │
        ├── WASS distribution-shift evaluation
        ├── deterministic target-scaling evaluation
        └── structural-similarity plots
```

`pipeline.sh` is the user-facing launcher. It only starts `main.py`; all path
resolution and stage orchestration live in Python.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- SymPy
- openpyxl
- Matplotlib
- PySR, including its Julia/SymbolicRegression backend

Install the Python dependencies in the environment that will run the pipeline:

```bash
python -m pip install numpy pandas scipy sympy openpyxl matplotlib pysr
```

PySR is only required for the two discovery scripts. It is imported lazily, so
the remaining parsers and utilities can still be inspected without PySR.

## Quick start

Inspect the resolved commands without starting a long run:

```bash
./pipeline.sh core --mode xonly --dry-run
```

Run the core pipeline:

```bash
./pipeline.sh core --mode xonly
```

Use a specific Python environment:

```bash
PYTHON_BIN=/path/to/pysr/python ./pipeline.sh core
```

or:

```bash
./pipeline.sh core --python /path/to/pysr/python
```

Show all path and process options:

```bash
./pipeline.sh --help
```

## Pipeline commands

| Command | Purpose |
| --- | --- |
| `prepare` | Generate the base physics CSV datasets. |
| `discover` | Run PySR in `xonly` or `with-params` mode. |
| `evaluate` | Refit generalized candidate formulas across datasets `0–7`. |
| `merge` | Select non-redundant shared formulas and join benchmark metadata. |
| `robustness` | Generate WASS/noise data and compute robustness metrics. |
| `report` | Plot an existing structural-similarity evaluation workbook. |
| `core` | Run `prepare → discover → evaluate → merge`. |
| `all` | Run every available stage. |

Common examples:

```bash
# Candidate search using only measured independent variables
./pipeline.sh discover --mode xonly --iterations 3

# Include physical parameters and fixed constants as PySR features
./pipeline.sh discover --mode with-params --iterations 3

# Recompute shared metrics instead of resuming existing results
./pipeline.sh evaluate --force

# Use fewer processes on a laptop
./pipeline.sh core --outer-processes 2 --pysr-processes 4 --workers 2

# Override all generated-output locations
./pipeline.sh all \
  --data-root /path/to/base-data \
  --generated-root /path/to/robustness-data \
  --results-dir /path/to/results
```

The pipeline does not start without an explicit command. `prepare` also skips an
existing CSV tree unless `--force` is supplied.

## Repository layout

```text
.
├── main.py                     Pipeline orchestrator
├── pipeline.sh                 Thin launcher for main.py
├── scripts/
│   ├── data/                   Base, WASS, and scaled-target data generation
│   ├── discovery/              PySR candidate discovery
│   ├── evaluation/             Shared and robustness fitting
│   └── reporting/              Candidate merging and plots
├── physicsMDSR_Range.xlsx      Main benchmark metadata
├── physicsMDSR_WASS_varRanges.xlsx
├── physicsMDSR_Range_CSV/      Checked-in/generated base CSV tree
└── results/                    Default pipeline result location
```

## Benchmark and data format

The `Sampling design` worksheet contains one row per physics problem. Important
columns include:

- `ID`: benchmark identifier such as `P01`.
- `OriginalFormula`: original physical equation.
- `GenerationFormula`: explicit target-generating expression.
- `Target`: output column.
- `IndependentVars`: sampled input columns.
- `ParameterRange`: names and ranges of physical parameters.
- `FixedConstantValues`: constants shared by every group.
- `paraValue`: one parameter vector per dataset group.
- `varRange`: sampling ranges for every independent variable and group.
- `targetRange`: accepted target interval for every group.

The base generator creates 16 CSVs per ID. Each CSV normally contains 5,000
rows ordered as:

```text
IndependentVar1, IndependentVar2, ..., Target
```

The current discovery and shared-evaluation stages consume only `0.csv` through
`7.csv`. Files `8.csv` through `15.csv` are generated but are not used by those
stages.

## Python entry points

Every script has an independent `--help` interface. The pipeline supplies all
paths explicitly, so the scripts do not depend on the current working directory
or old Windows drive paths.

### Pipeline orchestration

`main.py` and `pipeline.sh` sit above the four script categories. They connect
the categories without containing the scientific implementation of each stage.

#### `main.py`

Central pipeline parser and orchestrator.

- Resolves workbooks, data roots, generated-data roots, and results paths.
- Selects `xonly` or `with-params` discovery.
- Passes one stage's output path into the next stage.
- Supports individual stages, `core`, `all`, `--force`, and `--dry-run`.
- Runs child scripts with `subprocess.run(..., check=True)` and stops on failure.

```bash
python main.py all --mode xonly --dry-run
```

### 1. Data generation

Scripts in `scripts/data/` create base datasets and robustness-test variants.

#### `scripts/data/generate_physics_mdsr_csv_openpyxl.py`

Generates the base synthetic benchmark data.

For each ID and group, it samples independent variables uniformly, substitutes
the group's parameter vector and fixed constants, evaluates the chosen formula,
and rejects complex, non-finite, or out-of-range targets until it has the
requested number of valid rows.

Seeds are deterministic and derived from the base seed, ID, and group index.
The pipeline selects `GenerationFormula`; the standalone script retains
`OriginalFormula` as its compatibility default, so specify the column explicitly
when using the current rearranged benchmark workbook.

```bash
python scripts/data/generate_physics_mdsr_csv_openpyxl.py \
  physicsMDSR_Range.xlsx physicsMDSR_Range_CSV \
  --formula-column GenerationFormula --samples 5000 --groups 16
```

Useful safety options:

- `--validate-only`: validate workbook columns without writing data.
- `--overwrite`: allow replacement of numbered CSVs.
- `--allow-outside-target`: disable `targetRange` rejection.

#### `scripts/data/generate_mdsr_testsets.py`

Generates eight distribution-shifted test datasets per ID from columns such as
`paraValue_WASS_025` and `varRange_WASS_025`.

It validates formula symbols and metadata before generation, samples uniformly
within the WASS-specific ranges, and rejects non-finite results. Writes are
atomic through temporary files.

```bash
python scripts/data/generate_mdsr_testsets.py \
  physicsMDSR_WASS_varRanges.xlsx \
  --wass WASS_025 --output-root generated/wass/WASS_025
```

#### `scripts/data/add_target_noise_0_7.py`

Creates deterministic target-scaled copies of datasets `0–7`.

Despite the historical “noise” name, it does not add random noise. For level
`n`, it applies:

```text
scaled_target = target × (1 + n)
```

The default levels are `0.01`, `0.03`, `0.05`, and `0.1`, stored in directories
`001`, `003`, `005`, and `01`.

```bash
python scripts/data/add_target_noise_0_7.py \
  physicsMDSR_Range.xlsx physicsMDSR_Range_CSV generated/noise
```

### 2. Formula discovery

Scripts in `scripts/discovery/` run PySR and create generalized candidate
formula summaries.

#### `scripts/discovery/pysr_xonly.py`

Runs resume-safe PySR discovery using only the measured independent variables as
features. Physical parameters and fixed constants remain metadata.

The model is warm-started sequentially across the selected dataset indices.
Completion markers and PySR checkpoints allow an interrupted run to resume only
from a safe contiguous prefix. Broken pre-commit checkpoints are quarantined
instead of silently skipping data.

Outputs are written inside each ID directory:

- `<index>_pysr_all.csv`
- `<index>_pysr_top<N>.csv`
- `pysr_warm_0_7_summary.csv`
- `pysr_runs/<run-id>/`
- `.state_<run-id>/`

```bash
python scripts/discovery/pysr_xonly.py \
  physicsMDSR_Range.xlsx physicsMDSR_Range_CSV \
  --datasets 0-7 --iterations 2 --outer-processes 3 \
  --pysr-processes 6 --top-n 10
```

#### `scripts/discovery/pysr_with_params.py`

Runs PySR with independent variables, dataset-specific physical parameters, and
fixed constants included as feature columns. Parameter and constant columns are
constant within one CSV but can change across groups.

It uses a shared process queue across IDs, warm-starts across datasets, selects
the lowest-loss formulas within the configured complexity range, and writes the
same per-ID summary format as the x-only runner.

```bash
python scripts/discovery/pysr_with_params.py \
  physicsMDSR_Range.xlsx physicsMDSR_Range_CSV \
  --datasets 0-7 --iterations 2 --outer-processes 3 \
  --pysr-processes 6 --top-n 10
```

### 3. Formula evaluation

Scripts in `scripts/evaluation/` refit candidate structures and measure their
performance on shared, shifted, and scaled-target datasets.

#### `scripts/evaluation/evaluate_shared_formulas.py`

Evaluates whether each discovered formula structure works across datasets
`0–7`.

For every selected PySR expression, numeric coefficients have already been
replaced by `p0`, `p1`, and so on. This script:

1. Fits those parameters independently to every dataset.
2. Retries optimizer failures using deterministic perturbed initial values.
3. Computes arithmetic mean, harmonic mean, and RMS variants of MSE and R².
4. Marks mathematically equivalent SymPy expressions as redundant.
5. Updates the per-ID summary using lock-safe temporary/pending files.

```bash
python scripts/evaluation/evaluate_shared_formulas.py \
  physicsMDSR_Range_CSV --datasets 0-7 --workers 4
```

#### `scripts/evaluation/fit_wass_metrics.py`

Fits one requested `TopRank` formula independently to eight datasets under each
WASS condition. It records mean MSE, normalized MSE, and R².

Multiple WASS labels run as separate child processes; each child may also use
multiple ID workers. Therefore the maximum fitting concurrency is approximately
`number of labels × --workers`.

```bash
python scripts/evaluation/fit_wass_metrics.py \
  results/top_shared_formulas.csv generated/wass \
  --top-rank 1 --wass WASS_0 WASS_025 WASS_04 --workers 2
```

#### `scripts/evaluation/fit_noise_metrics.py`

Uses the same bounded least-squares fitting implementation as the WASS evaluator
but reads the value-named target-scaling directories produced by
`add_target_noise_0_7.py`.

```bash
python scripts/evaluation/fit_noise_metrics.py \
  results/top_shared_formulas.csv generated/noise \
  --top-rank 1 --noise 001 003 005 01 --workers 2
```

Both robustness fitters use large finite failure markers so one invalid formula
does not terminate evaluation of the remaining IDs. Use `--validate-only` to
check formula metadata without fitting.

### 4. Reporting

Scripts in `scripts/reporting/` assemble final candidate tables and generate
summary plots.

#### `scripts/reporting/merge_top_formulas.py`

Scans for `pysr_warm_0_7_summary.csv`, retains rows marked as redundancy-group
leaders, removes failed fits, ranks by `shared_fit_mse_0_7`, and keeps up to
`--top-n` formulas per ID.

The selected formulas are joined to metadata and `simpFormula` from
`physicsMDSR_Range_with_simpFormula.xlsx`.

```bash
python scripts/reporting/merge_top_formulas.py \
  physicsMDSR_Range_CSV \
  physicsMDSR_Range_with_simpFormula.xlsx \
  results/top_shared_formulas.csv --top-n 10
```

#### `scripts/reporting/plot_metric_histograms.py`

Reads an existing structural-evaluation workbook, selects one formula per ID,
and produces high-resolution histograms for structural similarity and shared-fit
R². It also prints Pearson and Spearman associations between those metrics.

```bash
python scripts/reporting/plot_metric_histograms.py \
  physics_formula_structural_similarity_evaluation_withooutparam.xlsx \
  --output-dir results/plots
```

The repository currently contains the structural-evaluation workbooks as input
artifacts; it does not contain the step that originally assigned their
human-readable structural-similarity explanations and scores.

## Process tuning

PySR uses nested parallelism:

```text
outer ID processes × PySR processes per ID
```

The default x-only configuration is `3 × 6`. Change based on your computer performance

```bash
./pipeline.sh discover \
  --outer-processes 2 --pysr-processes 3 --iterations 2
```

The evaluation scripts separately use `--workers`. Numerical-library threads
are limited inside the shared evaluator to avoid multiplying BLAS/OpenMP threads
inside every worker.


## Resume and overwrite behavior

- `prepare` skips an existing CSV tree unless `--force` is used.
- WASS generation skips an existing label tree unless `--force` is used.
- `pysr_xonly.py` uses checkpoints and per-dataset completion markers.
- `pysr_with_params.py` treats a non-empty summary as a completed ID.
- `evaluate_shared_formulas.py` skips rows with complete metrics unless
  `--force` is used.
- Temporary and `.pending` result files protect completed calculations when
  Excel or WPS locks the destination file.

Always run `--dry-run` before a large experiment when changing several paths or
process settings:

```bash
./pipeline.sh all --dry-run
```
