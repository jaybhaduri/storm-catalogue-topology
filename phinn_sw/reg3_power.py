"""Registration 3 power analysis (B1/B3 column power_holm_worst uses alpha/4, the Holm worst case
of an earlier draft; under the registered gatekeeping B1 and B3 face at worst alpha/2: see power_gatekeeping), from Part A's intense-storm like-for-like log ratios only.

Moderate-storm outcomes are simulated as: (Part A fit of log ratio on log|Dst min|, evaluated at each
moderate storm's own Dst minimum, times an effect multiplier) + a residual resampled from that fit.
Tests at the Holm worst case (alpha / 4) and at alpha, 5,000 simulations per scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA, K, NSIM = 0.05, 4, 5000
MARGIN_P1 = np.log(1.10)


def main(out="data/derived/reg3/power.csv"):
    t = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")
    m = pd.read_csv("data/derived/reg3/moderate_storms.csv")
    n = len(m)
    rng = np.random.default_rng(0)
    rows = []
    for p in ["P1", "P2", "P3"]:
        x, y = np.log(-t["dst_min_nT"]), t[p].to_numpy()
        b = np.polyfit(x, y, 1)
        res = y - np.polyval(b, x)
        pred = np.polyval(b, np.log(-m["dst_min_nT"]))
        for mult in [1.0, 0.5, 0.0]:
            hits = {"B1": [0, 0], "B2": [0, 0], "B3": [0, 0]}
            for _ in range(NSIM):
                z = mult * pred + rng.choice(res, n)
                if p == "P2":
                    pv = stats.wilcoxon(z, alternative="greater").pvalue
                    hits["B1"][0] += pv < ALPHA / K; hits["B1"][1] += pv < ALPHA
                if p in ("P2", "P3"):
                    pv = stats.mannwhitneyu(z, y, alternative="less").pvalue
                    hits["B2"][0] += pv < ALPHA / K; hits["B2"][1] += pv < ALPHA
                if p == "P1":
                    pv = stats.wilcoxon(z - MARGIN_P1, alternative="less").pvalue
                    hits["B3"][0] += pv < ALPHA / K; hits["B3"][1] += pv < ALPHA
            for bet, (h4, h1) in hits.items():
                if (bet == "B1" and p == "P2") or (bet == "B2" and p in ("P2", "P3")) or (bet == "B3" and p == "P1"):
                    rows.append({"bet": bet, "proxy": p, "effect_multiplier": mult, "n": n,
                                 "expected_mean_log_ratio": float(mult * pred.mean()),
                                 "power_holm_worst": h4 / NSIM, "power_alpha": h1 / NSIM})
    # forward sample: intense storms, P2 and P3 above controls, by number of storms
    for p in ["P2", "P3"]:
        y = t[p].to_numpy()
        for k in [3, 5, 6, 8, 10]:
            h = sum(stats.wilcoxon(rng.choice(y, k), alternative="greater").pvalue < ALPHA / 2 for _ in range(NSIM))
            rows.append({"bet": "B4_forward", "proxy": p, "effect_multiplier": 1.0, "n": k,
                         "expected_mean_log_ratio": float(y.mean()), "power_holm_worst": h / NSIM, "power_alpha": np.nan})
    r = pd.DataFrame(rows)
    r.to_csv(out, index=False, lineterminator="\n")
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
