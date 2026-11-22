# Part A results: H1 and H2

Registration 1, DOI [10.5281/zenodo.24421887](https://doi.org/10.5281/zenodo.24421887), filed 21 November 2026.
Analysis run 21 November 2026, after filing, with the frozen settings (`config.yaml` `part_a`, commit
`f87999f`). Code: `phinn_sw/part_a.py` (`python -m phinn_sw part_a`); exploratory: `phinn_sw/part_a_explore.py`.
Tables: `data/derived/part_a/`.

## Verdict

| Hypothesis | Primary analysis | Robustness checks 1-4 | Conclusion |
|---|---|---|---|
| **H1** duration and integrated Dst add to minimum Dst | 0 of 4 proxies | 0 of 4 in every check | **Not supported** |
| **H2** Betti features add to the Dst summaries | 0 of 4 proxies | 0 of 4 in every check | **Not supported** |

Both nulls hold across all four checks that the protocol says must agree (Huber regression, severe
storms excluded, the two 2000 pipeline-check storms excluded, final Dst only), and across the secondary
proxies and both secondary populations.

## Primary analysis (n = 107 intense storms, inactive objects below 600 km, OLS)

Relative reduction = 1 − MSE(larger model) / MSE(smaller model), LOSO; positive favours the hypothesis.
95% interval: 2,000 bootstrap resamples of storms. p: one-sided Wilcoxon signed-rank on paired squared errors.

| Hyp. | Proxy | MSE smaller | MSE larger | Reduction | 95% interval | p | Holm p |
|---|---|---|---|---|---|---|---|
| H1 | P1 staleness | 0.749 | 0.772 | −3.2% | −8.2% to 1.2% | 0.93 | 1.00 |
| H1 | P2 in-track z > 3 | 0.254 | 0.260 | −2.3% | −9.7% to 2.7% | 0.73 | 1.00 |
| H1 | P3 gate exceedance | 0.373 | 0.375 | −0.6% | −5.1% to 3.2% | 0.44 | 1.00 |
| H1 | P4 decay | 0.385 | 0.390 | −1.3% | −5.7% to 5.7% | 0.15 | 0.58 |
| H2 | P1 | 0.772 | 0.795 | −2.9% | −12.3% to 4.0% | 0.86 | 1.00 |
| H2 | P2 | 0.260 | 0.224 | **+14.1%** | 0.7% to 25.5% | 0.11 | 0.46 |
| H2 | P3 | 0.375 | 0.377 | −0.5% | −5.9% to 4.0% | 0.59 | 1.00 |
| H2 | P4 | 0.390 | 0.405 | −3.7% | −8.1% to 0.4% | 0.80 | 1.00 |

Largest variance inflation factor: 3.6 (M1), 3.8 (M2), below the threshold of 10, so the ridge refit was
not triggered. Every proxy had 107 storms (minimum 80).

**H2, P2.** The one positive signal: Betti features cut the in-track-correction error by 14%, and the
bootstrap interval just excludes zero, but the Wilcoxon test does not reach significance (p = 0.11;
Holm 0.46). The gain is concentrated: M2 is better for 54% of storms, and five storms (largest:
6 November 2025, then 29 October 2000, 9 November 2024, 23 June 2015, 28 June 2024) give 87% of the
net gain. It is reproduced under
Huber (+11.9%) and without the 2000 storms (+13.9%), weakens without severe storms (+11.5%, interval
includes zero) and on final Dst only (+8.4%). Per the protocol it is not a supported result; it is the
lead for Registration 2 and later work.

## Robustness (full table: `data/derived/part_a/h1_h2_results.csv`)

| Check | n | H1 | H2 |
|---|---|---|---|
| 1 Huber regression | 107 | 0/4 | 0/4 |
| 2 Severe storms excluded | 97 | 0/4 | 0/4 |
| 3 7 April and 16 July 2000 excluded | 105 | 0/4 | 0/4 |
| 4 Final Dst only (2000-2020) | 83 | 0/4 | 0/4 |
| 5 Secondary proxies (6) | 105-107 | 0/6 | 0/6 |
| 6a All inactive LEO | 107 | 0/4 | 0/4 |
| 6b Active payloads | 106-107 | 0/4 | 0/4 |
| 7 Original control rule | 2 | not testable | not testable |

Check 7: the originally planned control rule gives at least three controls for only 2 storms, below
the 80-storm minimum, so it is reported descriptively only (as the protocol requires).

H2 preconditions (WP3, `data/derived/topology_controls.csv`): the hour-shuffle test leaves level
statistics unchanged and changes the Betti curves for all 108 storms; Betti curves separate real storms
from IAAFT surrogates at p ≤ 0.05 in 54% of storms (80% of severe storms). The features do carry
information about storm shape; that shape does not predict catalogue degradation beyond Dst level.

## Exploratory (not preregistered tests)

**The catalogue does degrade during storms.** The confirmatory outcome divides an 11-day peak by a
14-day control mean, which builds in a ratio above 1. Like for like (the same 11-day peak statistic taken
inside each control window; `explore_storm_locked.csv`):

| Proxy | Median storm/control ratio | Storms above controls | p (greater) | Severe storms: median ratio | Spearman with min Dst |
|---|---|---|---|---|---|
| P1 staleness | 0.82 | 35% | 1.00 | 1.02 | −0.12 |
| P2 in-track z > 3 | **1.71** | 85% | < 0.001 | **2.25** | −0.29 |
| P3 gate exceedance | **1.79** | 78% | < 0.001 | **3.03** (10 of 10) | **−0.40** |
| P4 decay | 1.12 | 59% | 0.003 | 1.01 | −0.19 |

Negative Spearman = deeper storms, larger ratios. Orbit-determination quality (P2, P3) degrades clearly
and scales with intensity; publication staleness (P1) does not rise at all, so the catalogue keeps
publishing on schedule while the published orbits get worse.

**Intensity alone predicts.** Adding minimum Dst to the covariates-only model cuts LOSO error for P3 by
11% (95% interval 3.8% to 18.7%, p = 0.049) and P2 by 4.5% (p = 0.003; interval includes zero)
(`explore_dst_min_alone.csv`). So the null for H1 is not "nothing predicts": intensity predicts, and
duration and integrated Dst add nothing once intensity is known (they are correlated with it in this
sample; maximum VIF 3.6).

## Analysis decisions made after filing (deviations)

1. **P4 across bands.** The protocol defines P4 per 50 km band but not how bands combine. A pair-weighted
   mean of band medians was tried first: re-entry outliers in the 100-250 km bands (band medians up to
   10^8 km/day) made the control mean negative for 49 of 108 storms, which would have left P4 below the
   80-storm minimum. Adopted: the pair-weighted median of band medians (dominated by the 400-550 km bands,
   where most pairs are). P4 is null under the adopted definition; under the first it would have been
   descriptive only. Neither changes a verdict.
2. **Count proxies across bands** (P1-P3): numerators and denominators summed over bands 100-550 km
   (mean altitude below 600 km).
3. **Publication cadence covariate:** element-set pairs (consecutive sets no more than 10 days apart)
   per tracked object per day, all populations and bands, over the 30 days before t0. **Catalogue size:**
   tracked objects on the day before t0, all populations and bands.
4. **Check 3 dates:** the 2000 storms with minima at 2000-04-07 00 UT and 2000-07-16 00 UT.
5. **Huber check:** scikit-learn `HuberRegressor` with default epsilon 1.35, on standardised predictors.

None of these was chosen by looking at H1 or H2 test results: decision 1 was taken from the outcome
table before any model was fitted. The run log and every intermediate table are in the repository.
