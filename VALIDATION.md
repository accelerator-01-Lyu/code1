# Upload validation — 2026-10-26

Source package: `P2OLSAL_data_code_20261025.zip` (395,251,473 bytes).

SHA-256:
`3ea206f542f0b4ef65ccac04d9776ab2b5db8a9fa0f400ae12f5c73bbffe8d98`

## Snapshot integrity

- 8,297 package files copied and compared by SHA-256 against the extracted
  source; all matched. No algorithm, numerical result, figure or dataset
  content was edited.
- 54 runtime log / stderr / stdout files were omitted.
- Original package documentation was retained. The root README documents
  the present snapshot and its known reproduction limitations.
- The previous repository snapshot is retained by parent commit
  `cb9e20b7896fa3944a459dfcca04c003437b4605`.
- `cifar10_full.npz` (184,800,490 bytes) and `cifar-10-python.tar.gz`
  (170,498,071 bytes) are tracked by Git LFS.

## Checks performed

All 137 Python source files passed `ast.parse` with Python 3.12.14.
The following existing smoke tests completed with all assertions passing:

| Command | Result |
| --- | --- |
| `python code/fcm_update.py` | PASSED |
| `python code/gain.py` | PASSED |
| `python code/fmis_switch.py` | PASSED |
| `python code/acquisition.py` | PASSED |

Test environment: Windows, Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1.
The test environment was isolated from the source snapshot. The package's
original documented Python 3.14 development environment was not reproduced.

Full paper experiments and figure regeneration were not rerun. Existing
results are archived outputs supplied by the authors; smoke tests do not
independently validate the reported experimental conclusions. Missing data
and original machine-specific paths are described in the root README.
