"""Match THCIC hospitals (ID, name, city) to CMS Provider of Services (Dec 2019, Texas hospitals) by name within city."""
import json
import re
from difflib import SequenceMatcher

import pandas as pd

ABBR = {
    "SAINT": "ST", "MED": "MEDICAL", "CTR": "CENTER", "CNTR": "CENTER", "CENTRE": "CENTER", "HOSP": "HOSPITAL",
    "REG": "REGIONAL", "RGNL": "REGIONAL", "MEM": "MEMORIAL", "HLTH": "HEALTH", "SYS": "SYSTEM",
    "UNIV": "UNIVERSITY", "COMM": "COMMUNITY", "CMNTY": "COMMUNITY", "HOSPS": "HOSPITALS", "FT": "FORT",
    "MT": "MOUNT", "BAPT": "BAPTIST", "METH": "METHODIST", "PRESBY": "PRESBYTERIAN", "REHAB": "REHABILITATION",
    "REHABILITATIVE": "REHABILITATION", "BEHAV": "BEHAVIORAL", "SPEC": "SPECIALTY", "CHILDRENS": "CHILDREN",
    "CHILDREN'S": "CHILDREN", "DAVIDS": "DAVID", "LUKES": "LUKE", "JOSEPHS": "JOSEPH", "MARKS": "MARK",
    "HCA": "", "THE": "", "OF": "", "AT": "", "INC": "", "LLC": "", "LP": "", "LTD": "", "DBA": "", "AND": "",
}
GENERIC = {"HOSPITAL", "MEDICAL", "CENTER", "HEALTH", "REGIONAL", "SYSTEM", "CAMPUS", "TEXAS", "MEMORIAL", "COMMUNITY"}


def norm_tokens(s):
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    out = []
    for t in s.split():
        t = ABBR.get(t, t)
        if t:
            out.append(t)
    return out


def norm_city(s):
    return "".join(norm_tokens(s)).replace("SAINT", "ST")


def score(a, b):
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    overlap = len(inter) / min(len(ta), len(tb))
    distinctive = len(inter - GENERIC) / max(1, min(len(ta - GENERIC), len(tb - GENERIC)))
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return round(0.4 * overlap + 0.4 * distinctive + 0.2 * seq, 3)


thcic = pd.read_csv("thcic_hospitals_2016_2019.tsv", sep="\t", dtype=str).fillna("")
fac = pd.read_csv("pudf_facilities_2016_2019.tsv", sep="\t", dtype=str, header=None, names=["THCIC_ID", "acute", "pudf_name"])
thcic = thcic.merge(fac, on="THCIC_ID", how="inner")
pos = pd.DataFrame(json.load(open("pos_tx_dec19.json", encoding="utf-8")))
pos = pos[["PRVDR_NUM", "FAC_NAME", "ST_ADR", "CITY_NAME", "ZIP_CD", "PRVDR_CTGRY_SBTYP_CD", "PGM_TRMNTN_CD",
           "TRMNTN_EXPRTN_DT", "CRTFD_BED_CNT", "BED_CNT"]].copy()
pos["city_n"] = pos["CITY_NAME"].map(norm_city)
pos["active"] = (pos["PGM_TRMNTN_CD"] == "00").astype(int)

rows = []
for _, h in thcic.iterrows():
    cands = pos[pos["city_n"] == norm_city(h["city"])]
    scope = "city"
    best = None
    for names in ([h["name"], h["pudf_name"]],):
        for _, p in cands.iterrows():
            s = max(score(n, p["FAC_NAME"]) for n in names if n)
            key = (s, p["active"], p["PRVDR_CTGRY_SBTYP_CD"] in ("01", "11"))
            if best is None or key > best[0]:
                best = (key, p)
    if best is None or best[0][0] < 0.55:  # fall back to statewide search
        for _, p in pos.iterrows():
            s = max(score(n, p["FAC_NAME"]) for n in [h["name"], h["pudf_name"]] if n)
            key = (s, p["active"], p["PRVDR_CTGRY_SBTYP_CD"] in ("01", "11"))
            if s >= 0.8 and (best is None or key > best[0]):
                best = (key, p)
                scope = "state"
    rec = {"THCIC_ID": h["THCIC_ID"], "thcic_name": h["name"], "thcic_city": h["city"], "acute": h["acute"],
           "years_listed": h["years_listed"], "notes": h["notes"]}
    if best is not None:
        p = best[1]
        rec.update({"score": best[0][0], "scope": scope, "CCN": p["PRVDR_NUM"], "pos_name": p["FAC_NAME"],
                    "address": p["ST_ADR"], "pos_city": p["CITY_NAME"], "zip": str(p["ZIP_CD"])[:5],
                    "pos_subtype": p["PRVDR_CTGRY_SBTYP_CD"], "pos_active": p["active"],
                    "pos_term_date": p["TRMNTN_EXPRTN_DT"]})
    rows.append(rec)

out = pd.DataFrame(rows)
out["match"] = pd.cut(out["score"].fillna(0), [-1, 0.55, 0.75, 2], labels=["none", "review", "auto"])
out.to_csv("thcic_pos_match.csv", index=False, encoding="utf-8")
print(out["match"].value_counts().to_string())
print("acute only:\n", out[out["acute"] == "1"]["match"].value_counts().to_string())
