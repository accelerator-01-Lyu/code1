# P2-OLSAL — updated experiments

This repository contains the code, datasets, archived experimental results,
and figures from `P2OLSAL_data_code_20261025.zip` (packaged 2026-10-25).
The previous `代码1.zip` snapshot remains available in the Git history at
commit `cb9e20b7896fa3944a459dfcca04c003437b4605`.

The accompanying manuscript is *What a KL-robust acquisition score already
knows: corrected envelope gradients and the redundancy of membership-spread
bonuses*.

## Contents

| Path | Contents |
| --- | --- |
| `code/` | Core algorithms, experiment runners, analysis and manuscript helper scripts |
| `code/results/` | Archived experiment results, including per-seed CSV shards |
| `code/figures/` | Figure scripts and archived figures |
| `data/synthetic/` | 72 synthetic instances and associated metadata |
| `data/controlled_overlap*/` | Controlled overlap diagnostic pools |
| `data/cross_task/` | Packaged cross-task datasets and caches |
| `README.txt`, `code/README.md` | Original package documentation, retained for provenance |

Source code, numerical results, figures and datasets are unchanged from the
supplied package. Runtime logs and Python bytecode caches are omitted.

## Download and environment

Two large dataset files are stored with **Git LFS**:
`data/cross_task/cifar10_full.npz` and
`data/cross_task/cifar-10-python.tar.gz`.
Install Git LFS before cloning, then run:

```sh
git lfs install
git clone https://github.com/accelerator-01-Lyu/code1.git
cd code1
git lfs pull
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\Activate.ps1` on Windows
PowerShell or `source .venv/bin/activate` on macOS/Linux), then install:

```sh
python -m pip install -r requirements.txt
```

The supplied package records Python 3.14 on Windows as its development
environment. See `VALIDATION.md` for the environment used to check this upload.
Optional manuscript and monitoring utilities have additional dependencies
listed in `requirements-tools.txt`.

## Core smoke tests

Run from the repository root:

```sh
python code/fcm_update.py
python code/gain.py
python code/fmis_switch.py
python code/acquisition.py
```

These are small algorithm checks; they do not rerun the paper experiments.

## Reproduction status and known package limitations

- Existing result files are included. Statements such as `UNRUN` / `TBD` in
  the original `code/README.md` describe an earlier development stage.
- The main and several supplementary runners expect
  `code/reference_repo/data/{iris,wine,glass,ecoli,aggregation,compound}.pkl`.
  The supplied archive excludes `reference_repo/`; these six files are not
  included here. The previous `代码1.zip` also did not supply them.
- Several scripts contain original machine-specific Windows paths. For
  example, `run_experiments.py` creates its hard-coded `ASSETS` directory
  when imported. Inspect and adapt paths before running the full workflows.
- Some runners use different data locations or online downloads. In
  particular, `data/cross_task/openml/` mentioned by the original package
  README is absent from this snapshot.
- Default runs of `run_ecoli_comparison.py` and `alpha0_sensitivity.py`
  can overwrite `code/results/prereg_verdicts.json`; use a separate working
  copy when regenerating results.
- Manuscript helper scripts may require local manuscript assets,
  `docx_helpers`, pandoc or Microsoft Word, which are not bundled here.

Full experiments have not been rerun as part of this repository update.
