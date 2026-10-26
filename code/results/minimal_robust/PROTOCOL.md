# Minimal Robustness Protocol (Appendix D.3)
- Datasets: ecoli, compound.
- Methods: P2_Full, P2_NoExploration, Random.
- eta in {0, 0.1, 0.2}; l0_frac in {0.02, 0.05}.
- Seeds: 0,1,2,3,4 (first 5 of frozen 30; not cherry-picked).
- Noise: rng=RandomState(seed*1000+l0); flip=choice(n, round(eta*n), replace=False);
  yt[flip]=randint(0,c) uniform over ALL K classes (may keep original; effective flip eta*(K-1)/K).
- 30 cells (eta=0, l0=.02) reused from main seed_set=30 shards, no rerun.
- 150 new cells.
- Evaluation: acc_noisy (Hungarian on yt, primary) + acc_clean (on true y, secondary).
- n=5 paired 95% CI, not narrowed.
