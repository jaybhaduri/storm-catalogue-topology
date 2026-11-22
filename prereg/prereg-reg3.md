# A Thousand Gannons, Registration 3: Storm-locked degradation of the public satellite catalogue in unseen storms

**Preregistration 3** (secondary-data preregistration; confirmatory replication). It follows Registration 1 (Part A, DOI 10.5281/zenodo.24421887), whose hypotheses H1 and H2 were tested and not supported, and whose exploratory analyses motivate the bets below. Registration 2 (Part B, H4) is separate.

- Author: Jayabrata Bhaduri (CapaCloud Corp; ORCID 0000-0001-6190-145X)
- Authorship note: the version filed on Zenodo (v1) listed a second co-author, who asked on 21 November 2026 to be removed; the Zenodo author list and this text were updated accordingly. Nothing else in the registration was changed.
- Frozen analysis code, sample and settings: repository commit `52e3b6ed0408431b4af25750e7db76b5f93d6c9a` (private until publication). Command: `python -m phinn_sw reg3`.
- Structure: follows the questions of the OSF "Secondary Data Preregistration" template

## 1. Study information

### Description

Registration 1 tested whether storm duration (H1) or the topological shape of the Dst time series (H2) predicts degradation of the public low-Earth-orbit catalogue beyond storm intensity. Neither was supported. Its exploratory analyses found that, in the 107 intense storms of 2000–2025, orbit-determination quality degrades clearly during storms: element-set in-track corrections (P2) and position-gate exceedances (P3) peak at about 1.7–1.8 times their quiet-time level, more in deeper storms. Element-set staleness (P1) does not rise. Those findings were seen in data already analysed, so they cannot be claimed. This registration tests them as predictions on storms not yet analysed: the isolated moderate storms of 2000–2025 (primary) and the intense storms of 2026–2027 (secondary, forward).

### Research questions

- **RQ1.** Does orbit-determination quality in the public catalogue degrade during moderate geomagnetic storms?
- **RQ2.** Is the degradation smaller in moderate storms than in intense storms (dose-response across storm classes)?
- **RQ3.** Does the catalogue's publication staleness stay flat during storms while its orbit quality degrades?

### Hypotheses (bets)

All directional. Outcome definitions in section 3.

- **B1.** In moderate storms, the in-track correction proxy P2 rises above its quiet-time level: median like-for-like log ratio > 0. Predicted size: ratio about 1.17 (Part A's intense-storm fit extrapolated to each moderate storm's Dst minimum).
- **B2.** Moderate-storm degradation is smaller than intense-storm degradation, for both P2 and P3: moderate-storm log ratios are stochastically lower than the frozen Part A intense-storm log ratios (median ratios 1.71 for P2, 1.79 for P3). Won only if both tests pass.
- **B3.** In moderate storms, staleness P1 does not rise by 10% or more: median like-for-like log ratio < log(1.10).
- **B4 (secondary, forward).** In intense storms from 1 January 2026 to 30 June 2027, P2 and P3 rise above their quiet-time levels (median log ratio > 0 for each). Won only if both tests pass.

No bet on storm duration or storm shape is made. Design work on the Part A storms (section 4, item 4) found no shape representation that adds predictive power, so such a bet is not expected to be won and is left to exploratory analysis.

## 2. Data description

### Datasets used

The same frozen public snapshots as Registration 1 (all cut at 31 December 2025): Space-Track GP element-set history for low-Earth-orbit objects with SATCAT and decay records (2000–2025); WDC Kyoto hourly Dst; GFZ Kp and Hp30; NASA OMNI2 hourly (F10.7 for control matching). For B4: Kyoto Dst and Space-Track GP history from 1 November 2025 to 10 July 2027, retrieved after 10 July 2027 with the same scripts, plus GFZ Kp and OMNI F10.7 for the same period for controls.

### Data availability

Publicly available; Space-Track requires a free account.

### Data access

As Registration 1: Dst, Kp, Hp30, OMNI and JB2008 snapshots are in the repository. Space-Track records are not redistributed (User Agreement); anyone with an account can re-download them with the repository's script.

### Data identifiers

Kyoto Dst doi:10.17593/14515-74000; GFZ Kp doi:10.5880/Kp.0001; OMNI https://spdf.gsfc.nasa.gov/pub/data/omni/; Space-Track https://www.space-track.org. Snapshot hashes: Registration 1's `manifest.json` (attached again). Frozen inputs, SHA-256: `moderate_storms.csv` 599a202f25608291ff9a58c618f5374db8339c65875a101191d58cdbedb6a96d; `moderate_controls.csv` 576b8c417bf9def59ce0c110d41f24f4e7fcdaf55f48d203d32e09b7772eae99; Part A reference outcomes `explore_like_for_like.csv` a49d99f606febfe7b0d64b14ed71cce1eb8436ec2631f33a8fe4853dbde26037.

### Access date

Snapshots: 20 November 2026; Space-Track 2000–2025: 20–21 November 2026. B4 data: after 10 July 2027 (2026 Dst values to 20 November 2026 were viewed on 21 November 2026 to count intense storms; see section 4).

### Data collection procedures

As Registration 1 (element sets from the US Space Surveillance Network via Space-Track, no covariance; Dst from four low-latitude observatories; known biases: classified objects omitted, catalogue growth after 2019, provisional Dst after 2020, cadence drop in 2013–2014).

### Codebook

Space-Track GP/OMM documentation; OMNI2 word list; Kyoto HAPI hour_dst; GFZ Kp documentation. Derived variables: Registration 1's `config.yaml` and the module docstrings of `phinn_sw/reg3.py` and `phinn_sw/reg3_sample.py` at the frozen commit (summarised in section 3).

## 3. Variables

### Manipulated variables

Not applicable (observational).

### Measured variables

**Moderate storm (primary sample).** A run of hours with Dst below −50 nT (runs separated by less than 24 hours merged) whose minimum stays above −100 nT, with t0 (hour of minimum) from 1 January 2000 to 31 December 2025. **Isolated:** no other storm with minimum below −50 nT has its minimum within 10 days before or after t0. This keeps every peak window clear of the intense storms of Registration 1 and of each other. The February 2022 Starlink event is excluded (seen in Part A). Result: 101 storms (minimum −99 to −51 nT, median −62 nT; 81 with final and 20 with provisional Dst at minimum), listed in `moderate_storms.csv`.

**Quiet-time controls.** Registration 1's rule (14 days with Kp at most 5−, mean F10.7 within 40 sfu, any season, year ±3, reuse allowed), blocking every storm window with minimum below −50 nT. Built from Kp and F10.7 only: 97 storms have 5 controls, 2 have 3–4, 2 have fewer than 3 (`moderate_controls.csv`).

**Intense storm (B4).** Registration 1's definition (Dst below −100 nT, 24-hour merging), t0 from 1 January 2026 to 30 June 2027, controls by the same rule from data to 10 July 2027.

**Custody proxies.** Registration 1's daily proxies for inactive objects (debris, rocket bodies, payloads without an orbit raise in the past year) with mean altitude below 600 km, counts summed over the 50 km bands from 100 to 550 km:

- P1 staleness: share of tracked objects whose newest element set is older than 24 h.
- P2 in-track correction: share of element-set pairs whose in-track correction z-score (against the object's own quiet-day distribution) exceeds 3.
- P3 gate exceedance: share of tracked objects whose position correction exceeds the band's quiet-time 99th percentile.

**Outcome (like for like).** For each storm and proxy: log of [the maximum daily value over storm days 0 to +10] divided by [the mean, over the storm's controls, of the maximum daily value over control days 0 to +10]. The same statistic sits in the numerator and the denominator. This corrects the upward bias of Registration 1's peak-over-mean ratio.

**Predictors.** Storm class (moderate vs intense) for B2. Minimum Dst for exploratory dose-response.

### Missing data

Storms with fewer than 3 controls are excluded: 2 storms, expected n = 99. Storms whose peak or control statistic is zero or missing for a proxy are excluded for that proxy. If fewer than 80 storms remain for a primary test, that test is reported descriptively and counts as not passed.

### Unit of analysis

The storm. Primary: isolated moderate storms with at least 3 controls (n = 99). Reference (B2): Registration 1's 107 intense storms, with their like-for-like outcomes frozen in `explore_like_for_like.csv`. B4: intense storms in the forward window; at least 6 are required for a test; otherwise B4 is reported descriptively.

### Statistical outliers

No storm is removed. Rank-based tests are used throughout.

### Sampling weights

None; the sample is the full set of qualifying storms.

## 4. Knowledge of data

### Prior publication/dissemination

Registration 1 (DOI 10.5281/zenodo.24421887) and its results (H1 and H2 not supported) are being written up. No results on moderate storms have been produced or shared.

### Prior knowledge

1. **Registration 1 results and exploratory analyses** (intense storms, 2000–2025), all of which motivate these bets: P2 and P3 like-for-like median ratios 1.71 and 1.79; P1 0.82; Spearman correlations with minimum Dst −0.29 (P2) and −0.40 (P3).
2. **Moderate-storm sample and controls** were built from Dst, Kp and F10.7 only. No custody proxy has been computed or viewed for any moderate-storm window.
3. **Partial exposure.** The daily proxy tables for 2000–2025 exist, and were checked for coverage in Registration 1. For 36 of the 101 moderate storms, some of days 0 to +10 fall inside a Registration 1 control window. Those days entered Registration 1 only as part of 5-window control averages, never as storm outcomes or daily values that were inspected. A registered robustness check excludes these 36 storms (section 5).
4. **Design work on Registration 1's intense storms** (exploratory, `dev_shape_variants.csv`). Adding Betti features of Dst or of Hp30 over a focused window (t0 − 1 d to t0 + 3 d), or heating totals (integrated Hp30, JB2008 DTC), did not reduce leave-one-storm-out error for P2 or P3: every variant was 0.2% to 7.9% worse than the Dst-summary model. No shape bet is therefore registered.
5. **Power analysis** (`power.csv`, `power_gatekeeping.csv`) used Registration 1's intense-storm outcomes only (section 5).
6. **2026 Dst** to 20 November 2026 was viewed to count intense storms for B4 (three so far: 20 January, 22 March and 4 July 2026). No 2026 catalogue data has been retrieved.

Jayabrata Bhaduri directed and saw items 1–6.

## 5. Analyses

### Statistical models

Rank-based tests, no regression model.

- B1: one-sided Wilcoxon signed-rank test of moderate-storm P2 log ratios against 0 (alternative: greater).
- B2: two one-sided Mann–Whitney U tests, moderate-storm against Registration 1 intense-storm log ratios (alternative: moderate lower), for P2 and for P3.
- B3: one-sided Wilcoxon signed-rank test of (P1 log ratio − log 1.10) (alternative: less).
- B4: one-sided Wilcoxon signed-rank tests of forward intense-storm P2 and P3 log ratios against 0.

Effect sizes are reported as median ratios with 95% bootstrap intervals (2,000 resamples of storms); for B2, the difference in median log ratio.

### Effect size

Predicted, from Registration 1's intense storms (log ratio regressed on log |Dst minimum|, evaluated at each moderate storm's minimum):
- **P2 (B1):** median ratio about 1.17 in moderate storms.
- **P3:** about 1.0.
- **P1:** below 1.
- **Smallest effects of interest:** for B1, a 5% rise in P2 (ratio 1.05); for B3, the margin of a 10% rise in P1.

### Statistical power

Monte Carlo, 4,000–5,000 simulations per scenario. Moderate-storm outcomes were simulated as the predicted effect times a multiplier, plus residuals resampled from the Registration 1 fit; n = 101.

| Bet | Predicted effect | ¾ of predicted | ½ of predicted | No effect |
|---|---|---|---|---|
| B1 | 0.86 | 0.63 | 0.34 | 0.03 (false-positive rate) |
| B2 (each test) | 1.00 | 1.00 | 1.00 | 1.00 |
| B3 | 1.00 | 1.00 | 0.99 | 0.34 |

B4 power with Holm over its two tests: 0.37 (P2) and 0.22 (P3) with 6 storms; 0.80 and 0.55 with 10 storms. B4 is secondary because the number of storms in the forward window is not known in advance.

### Inference criteria

Family-wise alpha 0.05 over the primary bets, by gatekeeping:
- **Stage 1:** B2-P2, then B2-P3, in fixed order, each at 0.05.
- **Stage 2:** only if both stage-1 tests pass, B1 and B3 are tested with Holm correction at 0.05.

A test not reached counts as not passed. A bet is **won** when all of its tests pass, and **lost** otherwise. B4 is a separate family: Holm over its two tests at 0.05, and tested only with at least 6 storms. Every result is reported whichever way it falls.

### Assumption violation / model non-convergence

The tests are distribution-free. Ties and zero differences use SciPy's defaults (Wilcoxon `zero_method="wilcox"`; exact or normal approximation chosen automatically). If a proxy is undefined for a storm (zero peak or zero control statistic), that storm is dropped for that proxy.

### Reliability and robustness testing

Each of the following repeats the primary tests. They are reported whichever way they fall, and do not change the bet verdicts.
1. **Partial exposure:** excluding the 36 storms whose peak window overlaps a Registration 1 control window.
2. **Final Dst only:** storms whose Dst at minimum is final.
3. **Five controls:** storms with all five controls.
4. **B1 with P3:** P3 in place of P2.
5. **Other populations:** all inactive LEO objects (any altitude) and active payloads.

A won bet is described as robust if it also passes check 1.

### Exploratory analysis

Labelled exploratory in any report:
- dose-response within moderate storms (Spearman of log ratio with minimum Dst);
- P4 (decay) and recovery times;
- superposed-epoch curves of daily proxies;
- shape models (Betti features of Dst and Hp30, heating totals) applied to moderate storms;
- altitude-band differences.

## Attached files

| File | Contents |
| --- | --- |
| `moderate_storms.csv` | The 101 isolated moderate storms (primary sample) |
| `moderate_controls.csv` | Their quiet-time control windows |
| `explore_like_for_like.csv` | Registration 1 intense-storm like-for-like outcomes (B2 reference, frozen) |
| `power.csv`, `power_gatekeeping.csv` | Power simulations |
| `dev_shape_variants.csv` | Design work on shape models (prior knowledge item 4) |
| `manifest.json` | Hashes of the public data snapshots |

Not attached: Space-Track records (redistribution restricted) and the code (private until publication; the commit hash pins the version).

## References

Registration 1: Bhaduri, J. (2026). A Thousand Gannons, Part A (preregistration 1 of 2). Zenodo. doi:10.5281/zenodo.24421887. · Berger, T. et al. (2020). Space Weather 18. · Flohrer, T., Krag, H. and Klinkrad, H. (2008). AMOS. · Oliveira, D. M. and Zesta, E. (2019). Space Weather 17, 1510–1533. · Parker, W. E. and Linares, R. (2024). J. Spacecraft and Rockets.
