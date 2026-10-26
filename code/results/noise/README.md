# Noise Robustness Post-hoc

Two independent sets, side-by-side:
- n=5 (seeds 0-4): minimal robustness confirmation, see minimal_robust/PROTOCOL.md.
- n=30 (seeds 20261010-20261039): legitimate power thickening, isomorphic to E. coli 30->50.

12 cells (2 datasets x 3 eta x 2 L0) retained. Post-hoc robustness backing; not part of pre-registered P1-P4; does not replace the six real-dataset main results.

Engineering: run_traj gained a run_x2 switch (default True, frozen shards unchanged). Noise uses run_x2=False to skip the x2 gradient diagnostic (~328 grad_w_crn calls per shard), cutting ecoli shard from ~3 min to ~9 s. Five-cell sampling: run_x2 False vs True ACC worst diff = 0.000 (neutral).

Reproduce: run_d.bat run_noise_n30.py.
Files: noise_full_n30.csv (720), noise_summary_n30.csv, noise_contrasts_n30.csv, noise_summary_n5.csv.
