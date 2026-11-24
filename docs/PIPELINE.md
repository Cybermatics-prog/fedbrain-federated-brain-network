<!-- SPDX-License-Identifier: MIT -->
# Execution protocol

This document is the single canonical procedure for running the six-site federation and the
recorded demo trace in this repository. The README quick start is its short form. Every step
is a subcommand of `python scripts/pipeline.py`, which runs on Linux, macOS and Windows
without PowerShell, so no shell-specific wrapper is required. Run all commands from the
repository root.

## Contents

- [Scope and recorded facts](#scope-and-recorded-facts)
- [Requirements](#requirements)
- [Obtain the code](#obtain-the-code)
- [Create the environment](#create-the-environment)
- [Download the data](#download-the-data)
- [Build the health-view cache](#build-the-health-view-cache)
- [Train the demo trace](#train-the-demo-trace)
- [Run the tests](#run-the-tests)
- [Verify the repository](#verify-the-repository)
- [Serve the interface](#serve-the-interface)
- [One-shot run](#one-shot-run)
- [Make targets](#make-targets)
- [Direct module equivalents](#direct-module-equivalents)
- [Platform notes](#platform-notes)
- [Generated artifacts](#generated-artifacts)
- [Run-to-run consistency](#run-to-run-consistency)
- [Expected results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Compliance](#compliance)
- [Related documents](#related-documents)

## Scope and recorded facts

The repository implements a six-site federated brain-disorder analysis system on the public
ABIDE-I PCP release with CPAC `filt_noglobal` preprocessing and AAL116: 465 subjects (241 HC /
224 ASD) at six sites, NYU 172, UM 88, USM 67, UCLA 64, KKI 33 and YALE 41. Methodological
background and related federated work: [RELATED_WORK.md](RELATED_WORK.md).

The committed trace records 8 rounds of six-client training in one process on the full
465-subject cohort (292 train, 77 validation, 96 test). The federation has 4113 shared
parameters (16452 bytes pre-encryption) and 18 private parameters per site. Recorded round
summaries show: aggregation arithmetic error is 0.0; private blocks stay unchanged
(`private_preserved = true`); equal-mean replay maximum error is 2.98e-8. `noise_scale =
0.01` is a Laplace scale `b = 0.01`, not a certified epsilon-DP guarantee. Paillier 2048-bit
aggregation is real arithmetic, but the coordinator holds the key in the same process and
broadcasts plaintext. Alignment statistics (marginal mean, class means, class-window counts)
are exchanged in plaintext. Measured results are listed under Expected results and in
[RESULTS.md](RESULTS.md).

## Requirements

- Python 3.10 or newer, available as `python` on PATH. Any Python 3.10+ interpreter works;
  the setup step creates the project virtual environment.
- git, to obtain the repository.
- Network access for the public data download and about 2 GB of free disk space under
  `data/`.
- Optional: a CUDA build of PyTorch and an NVIDIA GPU. CPU works; the configuration uses
  `device: auto` and falls back to CPU.
- Optional: GNU make for the `make` targets. Windows does not need make or PowerShell.

Reference environment: Python 3.10+, torch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX 5070 Ti,
16 GB.

## Obtain the code

Clone https://github.com/Cybermatics-prog/fedbrain-federated-brain-network and run the
commands below from the repository root. The repository contains code, configurations,
documentation and the committed demo trace; it does not contain the ABIDE-I data, the derived
caches or model checkpoints.

## Create the environment

Linux and macOS:

```bash
python scripts/pipeline.py setup
```

Windows, relative paths only (cmd.exe or PowerShell):

On Windows, confirm the interpreter first: `python --version` must print a version. If it
opens the Microsoft Store or prints nothing, `python` on PATH is the Store alias; use the
full path to a real interpreter instead, for example
`"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" scripts\pipeline.py setup`, and replace `python` with that
path in the commands below.

```bat
python scripts\pipeline.py setup
```

`setup` creates `.venv/` when it is missing and installs `requirements.txt`. Later
subcommands detect and reuse `.venv/` automatically; activating it is not required.
`setup.ps1` and `start.ps1` are optional PowerShell wrappers around the same steps, not part
of the canonical path.

## Download the data

```bash
python scripts/pipeline.py data
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py data
```

Downloads the public ABIDE-I PCP derivatives (CPAC `filt_noglobal`, AAL116) into
`data/abide`, about 2 GB, and records a SHA-256 for every downloaded time-series file. The
data are not committed.

## Build the health-view cache

```bash
python scripts/pipeline.py features
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py features
```

Builds the FC and BOLD cache under `artifacts/health_data/` that the Health Data view reads.
The step checks the loaded cohort (465 subjects, 241 HC / 224 ASD, six sites) and does not
train a model.

## Train the demo trace

```bash
python scripts/pipeline.py demo
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py demo
```

Trains eight rounds with seed 42 on the full 465-subject cohort (292 train, 77 validation,
96 test subjects) and writes the federation trace that the interface replays. The command
uses `configs/federation_demo.json` when `runs/full-fedbrain-s42/checkpoint.pt` exists, and
`configs/federation_demo_standalone.json` otherwise; the standalone configuration fits the
sparse selection instead of reusing it. Output goes to `runs/demo-run/` by default, and an
existing run directory is left untouched. The committed trace in
`runs/federation-demo-s42/` is never overwritten. This run demonstrates the
client/upload/aggregate/broadcast process and is not the 50-round performance experiment.

## Run the tests

```bash
python scripts/pipeline.py test
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py test
```

Runs the offline unit, API and compliance suite from the project environment; the underlying
command is `python -m unittest discover -s tests -q`.

## Verify the repository

```bash
python scripts/pipeline.py verify
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py verify
```

Runs, in order: the license and portability compliance check, the English-only repository
scan, the offline test suite and, when `data/abide/manifest.json` exists, the health-data
independence check. That last step starts a temporary server on a free port, points a
headless browser at it with the experiment APIs forced to fail, and stops the server again,
so no manual `serve` is needed. It is skipped with a message when the data are absent or when
`playwright` is not installed; install it with `python -m pip install playwright` followed by
`python -m playwright install msedge`.

## Serve the interface

```bash
python scripts/pipeline.py serve --port 8765
```

```bat
.\.venv\Scripts\python.exe scripts\pipeline.py serve --port 8765
```

Starts the FastAPI server and the interface on http://127.0.0.1:8765. The health views need
the data and the health-view cache. The federation replay needs a recorded trace; the
committed trace is available for that purpose.

## One-shot run

```bash
python scripts/pipeline.py all
```

Runs data, features, demo, test and verify in that order. With GNU make installed:

```bash
make pipeline
```

## Make targets

| Target | Equivalent |
|---|---|
| make setup | python scripts/pipeline.py setup |
| make data | python scripts/pipeline.py data |
| make features | python scripts/pipeline.py features |
| make demo | python scripts/pipeline.py demo |
| make test | python scripts/pipeline.py test |
| make verify | python scripts/pipeline.py verify |
| make serve | python scripts/pipeline.py serve --port 8765 |
| make compliance | python scripts/check_compliance.py |
| make pipeline | python scripts/pipeline.py all |

The Makefile targets delegate to `scripts/pipeline.py` and accept `PYTHON` as an override
for the interpreter name. `make pipeline` is the complete sequence; the individual targets
are for rerunning one step.

## Direct module equivalents

```bash
python -m fedbrain.runner download
python -m fedbrain.runner train --config configs/federation_demo.json --run-id federation-demo-s42
python -m fedbrain.server --port 8765
python -m unittest discover -s tests -q
python scripts/check_no_chinese.py
python scripts/check_compliance.py
```

These are the module-level forms of the data, demo, serve, test, language and compliance
steps. They must run in an environment that has the dependencies installed; the
`scripts/pipeline.py` entry point does that automatically. On a clean clone, prefer
`python scripts/pipeline.py demo`, because it selects the standalone configuration when the
reference checkpoint is absent.

## Platform notes

- Linux and macOS: use the `python scripts/pipeline.py ...` commands. After setup, the
  project environment is detected and used automatically.
- Windows: use relative paths, as in the `.bat` blocks. cmd.exe and PowerShell both accept
  them. PowerShell is optional; `setup.ps1` and `start.ps1` are convenience wrappers, not
  requirements.
- make is available on Linux and macOS and optional on Windows; the Python entry point is
  the portable path.
- The interpreter must be Python 3.10 or newer; no other system packages are required.

## Generated artifacts

| Path | Produced by | Committed |
|---|---|---|
| `.venv/` | setup | no |
| `data/abide/` | data | no, about 2 GB |
| `artifacts/health_data/` | features | no |
| `runs/demo-run/` | demo | no |
| `runs/federation-demo-s42/` | committed trace | yes, `result.json`, `split.json`, `federation/index.json` and `federation/round-0001.json` to `round-0008.json` |
| `web/qa/` | browser QA scripts | no |

## Run-to-run consistency

- The main FedBrain and FedAvg measurements use seeds 42, 43 and 44. The demo trace uses
  seed 42 and eight rounds.
- The committed trace reused the sparse selection from `full-fedbrain-s42`, which is not
  distributed. A clean clone therefore fits the selection with
  `configs/federation_demo_standalone.json`. Both paths were executed on the reference
  machine and matched the committed metrics exactly (pooled 55.21%, macro-site
  53.41%, AUC 0.738, sensitivity 0.128, specificity 0.959); bit-identical values across
  different CPUs, BLAS builds or PyTorch versions are not claimed.
- Candidate models are selected on validation only; the test set is evaluated once the
  candidate is locked. Completed runs are reused rather than recomputed.
- CPU and GPU execution can differ in floating-point details.

## Expected results

After a successful clean-clone run, `python scripts/pipeline.py test` and
`python scripts/pipeline.py verify` exit 0, the interface replays the committed demo trace,
and the health views load the downloaded cohort without a checkpoint. The recorded
measurements are: FedBrain macro-site 60.76% +/- 4.05 over seeds 42/43/44 versus matched
FedAvg 63.69% +/- 3.19; demo run pooled 55.21%, macro-site 53.41%, AUC 0.738, sensitivity
0.128, specificity 0.959; two-round smoke 54.17% (link check only); Paillier versus plaintext
maximum predicted-probability difference 9.69e-8; bounded optimization pooled +1.03 pp with
95% interval [-5.83, +7.89] pp and the pre-registered gate not met.

## Troubleshooting

- `python` is not found: install Python 3.10+ and make sure the interpreter is available as
  `python` on PATH.
- Dependency installation fails: check network access and the Python version, then rerun
  `python scripts/pipeline.py setup`.
- The data step is slow or interrupted: rerun `python scripts/pipeline.py data` to complete
  the download into `data/abide`.
- `make` is not installed: use the `python scripts/pipeline.py ...` commands instead.
- No GPU: a CPU PyTorch build is sufficient; the configuration selects the device
  automatically.
- Port 8765 is already in use: stop the process using it, or pass a free port to the serve
  subcommand.
- The Health Data view is empty before `features` has run: run the data and features steps.
- `verify` reports a compliance problem: run `python scripts/check_compliance.py` and address
  the listed file.

## Compliance

Source files carry `SPDX-License-Identifier: MIT`; `LICENSE` and `LICENSES/MIT.txt` ship with
the repository, alongside `Makefile`, `setup.sh`, `pyproject.toml` and `REUSE.toml`.
`scripts/check_compliance.py` enforces the identifier, the `REUSE.toml`
annotations for JSON records and images, portable run records and dependency metadata.
`scripts/add_spdx_headers.py` adds the identifier to tracked files that can carry a comment,
and the rules themselves are covered by `tests/test_compliance.py`.
`python scripts/pipeline.py verify` runs these checks together with the English-only scan
and the test suite.

## Related documents

- [../README.md](../README.md): project overview, results and interface.
- [RESULTS.md](RESULTS.md): measured results generated from real run records.
- [ACCURACY_DIAGNOSIS.md](ACCURACY_DIAGNOSIS.md): read-only diagnosis of the low accuracy.
- [SPARSE_CONVERGENCE.md](SPARSE_CONVERGENCE.md): sparse optimization diagnosis.
- [FEDERATED_DATA_PROTECTION.md](FEDERATED_DATA_PROTECTION.md): data-protection view.
- [HEALTH_DATA_VIEW.md](HEALTH_DATA_VIEW.md): health-data definitions and usage.
- [UI_LANGUAGE.md](UI_LANGUAGE.md): English-only policy and checks.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): pre-registration and validity gates.
- [PUBLISH.md](PUBLISH.md): publishing notes for this repository.
