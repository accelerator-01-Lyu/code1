P2-OLSAL — Data and Code Package
=================================

Accompanying the manuscript
"What a KL-robust acquisition score already knows: corrected envelope
gradients and the redundancy of membership-spread bonuses"
(Knowledge-Based Systems submission, manuscript DRAL_kbs)
and its Supplementary Material.

Contents
--------
code/    Runnable experiment code (.py modules) with README.md,
         per-seed result shards (code/results, 7,872 files incl.
         7,662 per-seed shards referenced in the manuscript),
         and figure-generation scripts (code/figures).
data/    Benchmark datasets used in the main text and the
         Supplementary Material:
           synthetic/                Aggregation / Compound geometries
                                     and synthetic cluster instances
           cross_task/               OpenML-derived checks
                                     (Letter-26, Letter-10, Segment),
                                     Fashion-MNIST binary split
                                     (and CIFAR-10 variants used in
                                     cross-task validation)
           controlled_overlap/       E2 overlap-diagnostic pools
           controlled_overlap_c3/    c3 stratified overlap pools

Excluded
--------
code/reference_repo/  — third-party archived snapshot of
  https://github.com/accelerator-01-Lyu/P2OLSAL-experiments
  @ commit 1b20cfa6e14afbd16bc279f95ae6336f9d0c2db0 (2026-10-02),
  kept for audit traceability only; not required to reproduce.
  See code/README.md for the audit notes.

Environment
-----------
Python 3.14 (developed on 3.14.7 / Windows); dependencies:
numpy, scipy, scikit-learn, pandas, matplotlib.
Install: pip install numpy scipy scikit-learn pandas matplotlib

Entry points (see code/README.md for the full table)
-----------------------------------------------------
  python code/fcm_update.py           FCM membership update + smoke test
  python code/gain.py                 MC gain + exact KL-DRO dual
  python code/acquisition.py          acquisition modes + greedy batch
  python code/run_ecoli_comparison.py --full   6 datasets x 30 seeds
  python code/alpha0_sensitivity.py --full     alpha0 sweep + TOST

Data provenance
---------------
- UCI: Iris, Wine, Glass, E. coli (as packaged under data/).
- Synthetic geometries of Aggregation and Compound (see Section 5.1).
- OpenML-derived larger-scale checks: Letter-26, Letter-10, Segment
  (Section 5.2.8) under data/cross_task/openml/.
- Fashion-MNIST binary subset for cross-task validation (Section 5.2.6).

The manuscript and Supplementary Material describe the frozen protocol,
the locked 30-seed / 50-seed pairings, and the statistical safeguards
(Holm, BH-FDR, TOST); results are archived per seed in code/results.

Packaged 2026-10-25.
