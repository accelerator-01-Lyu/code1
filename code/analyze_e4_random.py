"""analyze_e4_random.py — E4 random-baseline comparison.

Compares frac_s_neg of queried selected points (c2_record/diagnostic.csv) vs
uniformly random pool points (random_baseline.csv), per overlap level.
Appends a_random_frac rows to fig13_data.csv (append-only).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "results", "theorem2", "c2_record")
RES = os.path.join(ROOT, "results", "theorem2")
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
B = 10000
RNG = np.random.default_rng(20261024)


def boot_prop(x, rng=RNG):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    bd = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))


def main():
    q = pd.read_csv(os.path.join(REC, "diagnostic.csv"), encoding="utf-8")
    r = pd.read_csv(os.path.join(RES, "random_baseline.csv"), encoding="utf-8")
    q = q.drop_duplicates(subset=["overlap_level", "seed", "method", "point_idx"])
    print(f"queried rows={len(q)}  random rows={len(r)}")

    rows = []
    rand_rows_for_fig = []
    for lv in LEVELS:
        qx = (q[q.overlap_level == lv]["s_i"] < 0).astype(float).values
        rx = (r[r.overlap_level == lv]["s_i"] < 0).astype(float).values
        qf, qlo, qhi = boot_prop(qx)
        rf, rlo, rhi = boot_prop(rx)
        diff = qf - rf
        # bootstrap test: queried > random (paired by level, independent resampling)
        nq, nr = len(qx), len(rx)
        diffs = qx[RNG.integers(0, nq, size=(B, nq))].mean(1) - \
                rx[RNG.integers(0, nr, size=(B, nr))].mean(1)
        p_one = float(np.mean(diffs <= 0))   # P(diff <= 0): evidence queried>random
        rows.append({"overlap_level": lv, "n_queried": nq, "n_random": nr,
                     "queried_frac": qf, "queried_lo": qlo, "queried_hi": qhi,
                     "random_frac": rf, "random_lo": rlo, "random_hi": rhi,
                     "diff_pp": diff * 100, "boot_p_queried_gt_random": p_one})
        rand_rows_for_fig.append({"panel": "a_random_frac", "overlap": lv,
                                  "n": int(nr), "value": rf,
                                  "ci_low": rlo, "ci_high": rhi})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(RES, "random_vs_queried.csv"), index=False, encoding="utf-8")
    print(cmp.round(4).to_string(index=False))

    # append-only to fig13_data.csv
    fig_path = os.path.join(RES, "fig13_data.csv")
    existing = pd.read_csv(fig_path, encoding="utf-8")
    before = len(existing)
    out = pd.concat([existing, pd.DataFrame(rand_rows_for_fig)], ignore_index=True)
    out.to_csv(fig_path, index=False, encoding="utf-8")
    print(f"\n[fig13] appended {len(out)-before} rows (a_random_frac); "
          f"total rows {before} -> {len(out)}")


if __name__ == "__main__":
    main()
