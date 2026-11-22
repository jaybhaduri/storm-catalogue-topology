# Registration 3 results: B1-B3 (moderate storms)

Registration 3, DOI [10.5281/zenodo.24423000](https://doi.org/10.5281/zenodo.24423000), published 21 November 2026.
Analysis run 21 November 2026, after publication (DOI commit `131bcb0`), with the frozen code (`52e3b6e`;
no change to `phinn_sw/` between the freeze and the run). Command: `python -m phinn_sw reg3`.
Tables: `data/derived/reg3/primary_tests.csv`, `robustness.csv`, `primary_outcomes.csv`, `run.log`.
No deviations from the registration.

## Verdict

| Bet | Claim | Result | Robust (passes check 1) |
|---|---|---|---|
| **B1** | In-track corrections (P2) rise in moderate storms | **Won** | Yes |
| **B2** | Degradation is smaller in moderate than in intense storms (P2 and P3) | **Won** | Yes |
| **B3** | Staleness (P1) does not rise by 10% or more in moderate storms | **Won** | Yes |
| B4 | Forward intense storms, Jan 2026 to Jun 2027 | Pending (data after 10 July 2027) | |

## Primary tests (n = 99 moderate storms; reference: 107 intense storms from Registration 1)

| Bet | Test | Median ratio (95% CI) | p | Adjusted p | Passed |
|---|---|---|---|---|---|
| B2 | P2 moderate < intense | 1.40 vs 1.71 | 0.0058 | 0.0058 (stage 1) | Yes |
| B2 | P3 moderate < intense | 1.19 vs 1.79 | < 0.0001 | < 0.0001 (stage 1) | Yes |
| B1 | P2 moderate > quiet | **1.40** (1.20 to 1.60) | < 0.0001 | < 0.0001 (Holm) | Yes |
| B3 | P1 moderate < 1.10 × quiet | **0.95** (0.75 to 1.05) | 0.0086 | 0.0086 (Holm) | Yes |

In moderate storms, the share of objects needing large in-track corrections peaks about 40% above
like-for-like quiet levels. The registration predicted about 17%, so the rise is larger than predicted.
It is still clearly smaller than in intense storms (71%), and gate exceedances follow the same order
(19% against 79%). Staleness does not rise (median 0.95 × quiet; the upper confidence limit, 1.05, is
below the 10% margin).

## Registered robustness checks

| Check | n | B2-P2 | B2-P3 | B1 | B3 |
|---|---|---|---|---|---|
| 1 Excluding 36 storms with partial Part A exposure | 65 | pass (1.45 vs 1.71) | pass | pass (1.45) | pass (0.87) |
| 2 Final Dst only | 79 | **fail** (p = 0.076) | pass | not reached (raw p < 0.0001, ratio 1.46) | not reached (raw p = 0.067) |
| 3 Five controls only | 97 | pass | pass | pass (1.35) | pass (0.94) |
| 4 B1 with P3 in place of P2 | 99 | | | P3 ratio 1.19, p = 0.020 | |
| 5a All inactive LEO | 99 | pass | pass | pass (1.28) | pass (0.87) |
| 5b Active payloads | 95-99 | pass | pass | **fail** (1.10, p = 0.12) | pass (0.76) |

Notes:
- **Check 2.** Moderate storms with final Dst show a slightly larger P2 rise (1.46), which narrows the gap
  to intense storms. The B2-P2 gate then misses, so stage 2 is not reached under the registered procedure,
  although B1's raw result is very strong.
- **Check 5b.** Active payloads are manoeuvred and tracked with priority, so a smaller storm response is
  plausible. Their B2 comparison is against the primary-population intense reference, as registered.
- **Check 4.** P3 also rises in moderate storms (ratio 1.19, p = 0.02). The registration predicted no
  P3 rise at moderate intensity.

## Combined reading with Registration 1

Across 206 storms in two independent sets, storms degrade the orbit quality of the public low-Earth-orbit
catalogue. The effect grows with storm class: in-track corrections rise 1.40× in moderate storms and
1.71× in intense storms; gate exceedances rise 1.19× and 1.79×. Publication staleness does not rise. The
catalogue keeps publishing on schedule while the published orbits become less accurate. Registration 1
showed that storm duration and the topological shape of Dst add nothing to intensity in predicting
this degradation.
