#!/usr/bin/env bash
# Extract the analysis inputs from the Texas Inpatient PUDF (tab-delimited quarterly zips, 2016-2019).
# Usage (from the working directory): bash scripts/run_extract.sh /path/to/pudf_zips
# Writes: stemi_patients_2016_2019.tsv, hosp_year_2016_2019.tsv, facility_flags.tsv
set -euo pipefail
PUDF="${1:?path to the folder with PUDF*_tab-delimited.zip files}"

zips() {  # quarterly tab-delimited zips in chronological order (file names differ between releases)
  for y in 2016 2017 2018 2019; do
    for q in 1 2 3 4; do
      ls "$PUDF"/*"${q}Q${y}"*tab-delimited.zip 2>/dev/null | head -1
    done
  done
}

base1() { for z in $(zips); do unzip -p "$z" 'PUDF_base1_*'; done; }

base1 | awk -f scripts/extract_stemi_cohort.awk > stemi_patients_2016_2019.tsv
base1 | awk -f scripts/tx_hosp_year.awk > hosp_year_2016_2019.tsv
for z in $(zips); do unzip -p "$z" 'Facility_type*'; done | awk -f scripts/facility_flags.awk > facility_flags.tsv
wc -l stemi_patients_2016_2019.tsv hosp_year_2016_2019.tsv facility_flags.tsv
