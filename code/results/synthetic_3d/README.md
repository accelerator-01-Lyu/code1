# Post-hoc 3-D Mechanism Validation

**Post-hoc Mechanism Validation — not part of pre-registered P1-P4 verdicts.**
Isolated 3-D synthetic follow-up to sanity-check Theorem 2 (double counting)
and Proposition 2 (Fisher-Rao boundary detection). Does NOT alter or feed the
six real-dataset pre-registered results.

## Scope (final, frozen)
We evaluated **three candidate geometries** (spiral skewer, Swiss roll,
hourglass) **plus a budget-tightness control on the Swiss roll**; all are
reported in `table_synthetic3d_posthoc.csv`. No further geometries will be
added. Main text uses Swiss roll only; spiral (near-inert) and hourglass
(negative-control geometry) are supplementary only.

## Frozen protocol
- Data-gen seed 20261010 (independent of AL seeds 20261010..20261019).
- K=6, 200 normal points/cluster = 1200 normal, +10 y=-1 eccentric decoys.
- P2_Full (rho=0.1 code default, untouched) vs P2_NoExploration (rho=0).
- Budget 10% (=120 labeled incl. |L0|=max(c,floor(.02N))=24).
- Decoys enter FCM fitting and acquisition; excluded from ACC/NMI/ARI;
  Hungarian on 6 normal clusters only.

## Swiss-roll budget-tightness control (pre-registered prediction, post result)
**Pre-registered prediction (written before results):** on the hardest overlap
end (sigma=0.8), the ACC gap NoExplo-Full should *amplify* as the labeled
budget tightens 10% -> 3% -> 2%, and Full's selection rate on y=-1 / high-h_i
points should rise.

**Measured result — prediction NOT supported:**

| sigma | 2% (L0-only) | 3% (12 active picks) | 10% (96 active picks) |
|---|---|---|---|
| 0.4 | 0.00pp (trivially 0) | +0.32pp [-0.48,+1.11] | +0.54pp [-0.09,+1.17] |
| 0.6 | 0.00pp | +0.08pp [-0.58,+0.75] | -2.54pp [-8.48,+3.40] |
| 0.8 | 0.00pp | +0.99pp [-0.70,+2.68] | +0.57pp [+0.01,+1.13] |

- The 2% row is an **L0-only baseline with zero active selections** (L0 =
  floor(.02*1200) = 24 = 2% target): the two methods are identical by
  construction, gap is exactly 0.
- At 3% the sigma=0.8 gap does NOT become significant (CI still crosses 0);
  sigma=0.4 shrinks, sigma=0.6 is reversed and high-variance.
- Decoy hit rate on the Swiss roll is ~0 (sigma=.4: 0.31% both methods; .6:
  0; .8: 0.10%), at or below the random baseline 0.83%.
- Conclusion: even under tightened budgets the Theorem-2 eccentric-decoy
  induction did not clearly manifest on the Swiss roll. The budget curve is
  therefore **supplementary only, not a main-text scope figure**.

## Manipulation check (honest)
- Spiral / Swiss-roll: decoys low-entropy, high-eccentricity; umax ~0.55-0.81
  (FCM m=2 spreads out-of-cloud points; we did NOT move points to force 0.95).
- **Hourglass = NEGATIVE geometry control.** After one principled geometric
  correction, umax stayed 0.32-0.39 vs boundary 0.22-0.28. Geometric reason:
  cone-top bands are rings symmetric about the apex z=0; outward points are
  nearly equidistant to multiple ring points, FCM m=2 shares membership, no
  local single-cluster target forms. This delimits the scope of Theorem 2.

## Files
- `table_synthetic3d_posthoc.csv` — final 15 rows: 3 geometries x 3 sigma
  (budget=0.10) + 6 Swiss-roll budget rows (sigma .4/.6/.8 x budget .02/.03).
- `summary_all.csv` / `trajectory_all.csv` — 180 shards (3 geo x 3 sigma x
  2 methods x 10 seeds), NaN rows retained.
- `summary_swissroll_budget.csv` / `trajectory_swissroll_budget.csv` — 120
  shards (3 sigma x 2 methods x 2 budgets x 10 seeds).
- Figures in `figures/posthoc/` (main-text Swiss-roll set in `figures/posthoc/main/`).

## Reproduce
```
run_d.bat make_synthetic_3d.py
run_d.bat run_synthetic_3d.py --run
run_d.bat run_swissroll_budget.py
```
