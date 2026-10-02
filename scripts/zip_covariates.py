"""ZIP-level covariates for robustness: USDA RUCA 2010 (ZIP version) and ACS 5-year (latest release available
without an API key, via Census Reporter) for Texas ZCTAs. Output: ext/zip_covariates.csv. Run from 'derived'."""
import json

import numpy as np
import pandas as pd

r = pd.read_csv("ext/ruca2010_zip.csv", dtype=str)
r.columns = [c.strip("'") for c in r.columns]
r["ZIP_CODE"] = r["ZIP_CODE"].str.strip("'")
r = r[r.STATE == "TX"].copy()
r["ruca1"] = pd.to_numeric(r.RUCA1, errors="coerce")
r["ruca_cat"] = pd.cut(r.ruca1, [0, 3, 6, 9, 10.5], labels=["metro", "micropolitan", "small town", "rural"])

j = json.load(open("ext/acs_latest_tx_zcta.json", encoding="utf-8"))
release = j["release"]["name"]
rows = []
for geo, t in j["data"].items():
    e = {tab: t[tab]["estimate"] for tab in t}
    g = lambda tab, col: e.get(tab, {}).get(f"{tab}{col:03d}")  # noqa: E731
    pop = g("B01003", 1)
    a65 = sum((g("B01001", c) or 0) for c in list(range(20, 26)) + list(range(44, 50)))
    uninsured = sum((g("B27010", c) or 0) for c in (17, 33, 50, 66))
    bach = sum((g("B15003", c) or 0) for c in (22, 23, 24, 25))
    rows.append({"pat_zip": geo[-5:], "population": pop, "median_income": g("B19013", 1),
                 "pct_poverty": 100 * (g("B17001", 2) or 0) / g("B17001", 1) if g("B17001", 1) else np.nan,
                 "pct_uninsured": 100 * uninsured / g("B27010", 1) if g("B27010", 1) else np.nan,
                 "pct_65plus": 100 * a65 / g("B01001", 1) if g("B01001", 1) else np.nan,
                 "pct_bachelor": 100 * bach / g("B15003", 1) if g("B15003", 1) else np.nan})
a = pd.DataFrame(rows)
Z = pd.read_csv("zcta2010_tx_internal_points.csv", dtype={"zcta5": str}).rename(columns={"zcta5": "pat_zip"})
out = Z[["pat_zip", "arealand"]].merge(a, on="pat_zip", how="left").merge(
    r[["ZIP_CODE", "ruca1", "ruca_cat"]].rename(columns={"ZIP_CODE": "pat_zip"}), on="pat_zip", how="left")
out["log_pop_density"] = np.log1p(out.population / (out.arealand / 1e6))
out["log_income"] = np.log(out.median_income.where(out.median_income > 0))
out["acs_release"] = release
out.to_csv("ext/zip_covariates.csv", index=False)
P = pd.read_csv("stemi_patients_2016_2019.tsv", sep="\t", dtype=str, usecols=["pat_zip"])
m = P.merge(out, on="pat_zip", how="left")
print(f"ACS release: {release} | TX ZCTAs with ACS: {a.population.notna().sum()} | RUCA TX ZIPs: {len(r)}")
print("coverage of STEMI patients:", {c: f"{100 * m[c].notna().mean():.1f}%" for c in ["ruca1", "log_income", "pct_poverty", "pct_uninsured", "log_pop_density"]})
print(out[["ruca1", "log_pop_density", "median_income", "pct_poverty", "pct_uninsured", "pct_65plus", "pct_bachelor"]].describe().round(1).to_string())
