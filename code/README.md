# P2-OLSAL v2 — Corrected implementation & code audit

Runnable, independent `.py` modules. Smoke tests are **not paper results**.

## Environment
- Python 3.14 (developed on 3.14.7 / Windows)
- Dependencies (standard library only beyond these):
  `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`
- Install: `pip install numpy scipy scikit-learn pandas matplotlib`

## Reference repository
The archived implementation was audited against
`https://github.com/accelerator-01-Lyu/P2OLSAL-experiments` @ commit
`1b20cfa6e14afbd16bc279f95ae6336f9d0c2db0` (2026-10-02, "Complete experiments
upload excluding figures"), downloaded as a ZIP snapshot into
`reference_repo/`. No local `git` binary was available, so no working tree /
commit could be created; the snapshot SHA is recorded from the GitHub API.

## Files & how to run each
| File | Run | What it does |
|---|---|---|
| `fcm_update.py` | `python fcm_update.py` | Correct FCM membership update + synthetic smoke test |
| `gain.py` | `python gain.py` | `calculate_gain(u, alpha_0, epsilon_q, M_MC)` MC gain + exact KL-DRO dual; alpha0 regression smoke test |
| `fmis_switch.py` | `python fmis_switch.py` | FMIS on/off genuinely changes ranking; smoke test asserts it |
| `acquisition.py` | `python acquisition.py` | Four acquisition modes + greedy batch; smoke test |
| `run_ecoli_comparison.py` | `python run_ecoli_comparison.py` (dry-run) / `--full` | 6 datasets x 30 seeds x budgets; long CSV + prereg verdicts |
| `alpha0_sensitivity.py` | `python alpha0_sensitivity.py` (dry-run) / `--full` | alpha0 sweep + P4 TOST; CSV + png |

### dry-run vs --full
- **Default = dry-run**: prints the frozen protocol, writes `results/prereg_verdicts.json`
  with every verdict `PENDING`, does **not** run 30 seeds, does **not** write figure.
- **`--full`**: runs the harness end-to-end. First real runtime is expected on the
  order of tens of minutes (6 datasets x 30 seeds x 4 budgets x 9 methods). Record
  actual wall-clock + environment here once a real run is performed (currently
  UNRUN — all result cells are TBD; no numbers have been produced).

## Bug-fix list (see ../notes/code_audit_memo.md for full audit)
1. **FCM exponent** (`fcm_update.py`): archived applied `−2/(m−1)` to an already
   squared distance; fixed to `−1/(m−1)` on squared distances.
2. **KL-DRO dual** (`gain.py`): archived maximized a wrong expression that
   collapsed to `max(0,·)`; replaced with the exact entropic dual
   `sup_τ {−τκ − τ log E_q e^{−g/τ}}` via stable logsumexp + 1-D τ search, plus an
   Eq.(11) κ-adaptation hook.
3. **FMIS flag inert** (`fmis_switch.py`, `acquisition.py`): flag now actually
   selects order-1 pure-entropy vs order-2 robust gain and changes the ranking.
4. **alpha0 default-arg binding** (`gain.py`): `alpha_0` is a required explicit
   argument used in the body; the sweep now actually changes the gain.

## Pre-registered predictions (auto-judged, no cherry-picking)
`run_ecoli_comparison.py` implements P1 (NoExploration noninferiority), P2
(outlier-selection frequency), P3 (λ2/n neighborhood); `alpha0_sensitivity.py`
implements P4 (α0 TOST equivalence). Verdicts go to
`results/prereg_verdicts.json`: `prediction_id, dataset, estimate, ci_lo, ci_hi,
margin, verdict (CONFIRMED/REJECTED/INCONCLUSIVE/PENDING),
theory_assumption_to_revise`. Seeds, budgets, initial-pool rule, margins are
frozen in the CONFIG block; there is no row-deletion / seed-picking option.

## Reproduction steps
1. `pip install numpy scipy scikit-learn pandas matplotlib`
2. Run each `python <file>.py` smoke test — all must print `PASSED`.
3. (Optional, real compute) place `data/aggregation.csv`, `data/compound.csv` if
   desired, then `python run_ecoli_comparison.py --full` and
   `python alpha0_sensitivity.py --full`.
4. Inspect `results/prereg_verdicts.json` — verdicts are computed automatically,
   not hand-read off plots.
