# Storm topology and satellite-catalogue degradation

Code and derived data for *Topological storm features do not predict satellite-catalogue degradation:
a preregistered machine-learning test across 206 geomagnetic storms* (J. Bhaduri, CapaCloud Corp).
Correspondence: ceo@capa.cloud.

The study asks whether persistent-homology (Betti-curve) features of the Dst time series predict how
much the public low-Earth-orbit catalogue degrades during a geomagnetic storm, beyond storm intensity.
It runs in two preregistered stages:

| Registration | DOI | Result |
| --- | --- | --- |
| 1: duration (H1) and topology (H2), 107 intense storms | [10.5281/zenodo.24421887](https://doi.org/10.5281/zenodo.24421887) | H1 and H2 not supported |
| 3: confirmatory replication, 101 unseen moderate storms | [10.5281/zenodo.24423000](https://doi.org/10.5281/zenodo.24423000) | B1, B2, B3 won (B4, forward test, pending) |

Write-ups: `prereg/results-part-a.md` and `prereg/results-reg3.md`. Registration texts: `prereg/`.

## Provenance of the frozen code

This public repository holds the code and data as of 22 November 2026, in a single commit. The
development history, including the commits frozen at each registration (`f87999f` for Registration 1,
`52e3b6e` for Registration 3), is kept in a private archive. Diffs against those commits are available
on request, so readers can verify that the confirmatory analyses ran on the registered code.

## Data

Every input is public. Parquet snapshots in `data/snapshots/` (WDC Kyoto Dst, NASA OMNI2, GFZ Kp and
Hp30, JB2008 indices), each with a SHA-256 content hash in `data/snapshots/manifest.json`.

Space-Track element sets are **not** included: the Space-Track User Agreement does not permit
redistribution. `phinn_sw/spacetrack.py` re-downloads the same records for any registered user
(`SPACETRACK_USER`, `SPACETRACK_PASSWORD`). Only derived, aggregated daily custody targets are included
(`data/derived/custody_proxies_daily.parquet`: day x 50 km band x population).

## Run

```bash
pip install -r requirements.txt
python -m phinn_sw pull          # re-pull the public snapshots
python -m phinn_sw storms        # storm catalogue and quiet-time controls
python -m phinn_sw spacetrack    # LEO element-set history (needs a Space-Track account)
python -m phinn_sw proxies       # daily custody targets from element sets
python -m phinn_sw topology      # Betti curves of Dst, shuffle and IAAFT checks
python -m phinn_sw part_a        # Registration 1 analysis (H1, H2)
python -m phinn_sw reg3          # Registration 3 analysis (B1-B3)
python -m phinn_sw.paper         # figure data, placebo windows, dose-response
python -m phinn_sw.ml_bench      # exploratory benchmark: ridge, random forest, gradient boosting
python -m pytest -q -m "not network"
```

## Layout

| Path | Contents |
| --- | --- |
| `phinn_sw/` | the analysis package |
| `config.yaml` | every frozen setting |
| `data/snapshots/` | public input snapshots and their hashes |
| `data/derived/` | storms, controls, daily custody targets, Betti curves, results tables |
| `prereg/` | registration texts, filed attachments, results write-ups |
| `tests/` | unit tests and pipeline dry runs |

## License

Code: [MIT](LICENSE). The public input snapshots keep their providers' terms (GFZ Kp and Hp30: CC BY 4.0;
cite WDC Kyoto, NASA SPDF and Space Environment Technologies as listed in `data/snapshots/manifest.json`).
