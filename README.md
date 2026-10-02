# STEMI routing on statewide Texas discharge data

Code, public data and aggregated results for the manuscript

> **Imitation accuracy is not routing quality: a case study of learning and counterfactual evaluation of STEMI routing policies on statewide discharge data**
> G. O. Bondarenko (ITMO University). Submitted to *Health Services and Outcomes Research Methodology*.

The study asks two questions about routing patients with ST-elevation myocardial infarction (STEMI) to hospitals:

1. **Which hospital do patients reach?** Hospital-ranking models (conditional logit, neural ranker, gradient boosting) are trained on dispatch-time information and evaluated prospectively (train 2017–2018, test 2019).
2. **What happens if the first hospital cannot perform PCI?** The effect of a non-PCI first hospital is estimated with an instrumental-variable design based on differential distance, and it is used to calibrate a detour rule.

The main lesson is that imitation accuracy and routing quality are different targets. On the decision-relevant patients, the simple rule "nearest PCI centre" outperforms the most accurate learned model.

## Repository contents

| Path | Content |
|---|---|
| `scripts/` | All code: PUDF extraction (awk), hospital geocoding (Node.js/Python), analyses and figures (Python) |
| `data/hospital_geocodes.csv` | 371 Texas hospitals (THCIC identifiers) with street addresses and coordinates, built from public sources only |
| `data/zcta2010_tx_internal_points.csv` | Internal points of Texas ZIP code tabulation areas (US Census Bureau, 2010 vintage) |
| `results/` | Aggregated tables reported in the paper (no patient-level data) |
| `figures/` | Figures 1–9 of the paper (PDF and PNG) |

## Data access

**Patient data are not included and may not be redistributed.**

The study uses the *Texas Hospital Inpatient Discharge Public Use Data File, 1st Quarter 2016 – 4th Quarter 2019* (Texas Department of State Health Services, Austin, Texas). The file is available from DSHS under a data use agreement at https://www.dshs.texas.gov/thcic. You need the quarterly `PUDF*_tab-delimited.zip` files for 2016–2019. All results in this repository and in the paper are aggregates, and no reported cell contains fewer than five patients.

Public sources used to build the hospital and geography inputs:

| Source | Used for |
|---|---|
| THCIC reporting-status reports (DSHS) | THCIC identifier → hospital name and city |
| CMS Provider of Services file, Q4 2019 (data.cms.gov API) | Street addresses |
| NPPES NPI Registry API | Campus-level addresses of multi-campus systems |
| US Census Geocoder and TIGERweb | Coordinates of hospital addresses; ZCTA internal points |
| OpenStreetMap Nominatim | Coordinates of 26 hospitals on highway addresses unknown to the Census geocoder |
| OSRM (public demo server) | Road travel times between ZIP codes and hospitals |
| USDA ERS Rural–Urban Commuting Area codes, 2010 (ZIP file) | Rurality |
| American Community Survey 5-year, via Census Reporter API | ZIP-level socioeconomic covariates |

## Requirements

- Python 3.12 with the packages in `requirements.txt`.
- Node.js ≥ 18, needed only for the geocoding scripts (built-in `fetch`).
- `bash`, `awk` (gawk) and `unzip`. On Windows, Git Bash works.

```bash
pip install -r requirements.txt
```

## How to reproduce

All scripts are run from a **working directory** that contains the `scripts/` folder and the input files.

```bash
mkdir work && cd work
cp -r ../scripts ../data/*.csv .
mkdir -p results ext figures
```

### 1. Extract the cohort and hospital aggregates from the PUDF

```bash
bash scripts/run_extract.sh /path/to/pudf_zips
# -> stemi_patients_2016_2019.tsv   patient-level STEMI cohort (keep private)
# -> hosp_year_2016_2019.tsv        per-hospital, per-year aggregates (keep private: small cells)
# -> facility_flags.tsv             teaching / psychiatric / pediatric flags
python scripts/build_hospital_table.py      # -> hospitals_tx_2016_2019.csv
```

### 2. Public covariates

```bash
curl -L -o ext/ruca2010_zip.csv "https://www.ers.usda.gov/media/5440/2010-rural-urban-commuting-area-codes-zip-code-file.csv"
curl -o ext/acs_latest_tx_zcta.json "https://api.censusreporter.org/1.0/data/show/latest?table_ids=B01003,B19013,B17001,B27010,B01001,B15003&geo_ids=860|04000US48"
python scripts/zip_covariates.py            # -> ext/zip_covariates.csv
python scripts/osrm_drive_times.py          # -> ext/osrm_zip_hosp.csv (about 100 requests, 1 per second)
```

### 3. Analyses

| Script | Produces | Paper |
|---|---|---|
| `stemi_models.py` | Cohort description by arrival mode, conditional logit, first IV estimates | Table A2 |
| `stemi_ranker_ml.py` | Neural ranker and gradient boosting (5 seeds), road-time variants, subgroups | Table 2, Fig. 3–4 |
| `stemi_ml_significance.py` | Paired cluster bootstrap and McNemar tests between models | Section 6.1 |
| `stemi_iv_robust.py` | Ten IV specifications, effective F, Anderson–Rubin, Hansen J, endogeneity and placebo tests, leave-one-region-out | Table 3, Table A1, Fig. 6–7 |
| `stemi_revision.py` | Analytic sample flow, Table 1, out-of-choice-set accuracy, models without clinical proxies, PCI-only constraint, detour rule | Tables 1, 2 and 4, Fig. 1 |
| `stemi_composed_rule.py` | Composed decision rule on 2019 routing failures | Section 6.3 |
| `stemi_policy_table.py` | Routing policies on 2019 routing failures with a non-PCI nearest hospital | Table 5 |
| `make_figures_hsorm.py` | Figures 1–9 | Fig. 1–9 |

`stemi_ranker_ml.py` also writes `results/ml_test_scores.npz`. This file contains patient-level predictions; keep it private.

### Optional: rebuild the hospital geocoding table

`data/hospital_geocodes.csv` was produced by the following steps (network access required):

1. `pdftext.js` and `parse_isr.js` — THCIC reporting-status PDFs (4Q2016–4Q2019) → identifier, name and city.
2. `match_pos.py` (with `namematch.py`) — name matching to the CMS Provider of Services file (Texas hospitals, Q4 2019).
3. `fetch_nppes.js` and `resolve2.py` — campus-level addresses from the NPI Registry for multi-campus systems.
4. `finalize_addresses.py` — manual campus overrides (71 addresses), documented in the script.
5. `geocode_census.py`, `geocode_retry.js` and `geocode_osm.js` — Census batch geocoder, house-number neighbours on the same street, and OpenStreetMap by name.

The exploratory scans `tx_ami_scan.awk` and `tx_stemi_years.awk` produce descriptive counts used during the study design.

## Use of AI tools

During the study, an AI assistant (Claude, Anthropic) was used as a programming and writing aid: for drafting and debugging code and for drafting and editing text. The author directed and approved all design and analytic decisions and verified the code and reported results.

## Citation

If you use this code or the hospital geocoding table, please cite the paper (details will be added after publication) and the data source:

> Texas Hospital Inpatient Discharge Public Use Data File, 1st Quarter 2016 – 4th Quarter 2019. Texas Department of State Health Services, Austin, Texas.

## License

Code: MIT License (see `LICENSE`). Data derived from public sources remain subject to the terms of their providers.

## Contact

Gleb O. Bondarenko, ITMO University — gobondarenko@itmo.ru — ORCID [0009-0008-2516-5692](https://orcid.org/0009-0008-2516-5692)
