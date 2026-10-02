"""Second-pass address resolution with a symmetric name score.
Candidates (same city): CMS POS Dec-2019 Texas hospitals + NPPES hospital NPIs (organisation and DBA names).
THCIC side: every name the hospital had in the THCIC status reports and PUDF facility files 2016-2019.
Auto-accept only a confident, unambiguous best address; everything else goes to review."""
import json
import re
from difflib import SequenceMatcher

import pandas as pd

from namematch import GENERIC, norm_city, norm_tokens

CAMPUS = {"NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "CENTRAL",
          "DOWNTOWN", "HEIGHTS", "MIDTOWN", "UPTOWN", "LAKESIDE", "SHORELINE", "WESTOVER", "MAINLAND"}


def score2(a, b):
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    da, db = ta - GENERIC, tb - GENERIC
    jd = len(da & db) / len(da | db) if (da | db) else 1.0
    ja = len(ta & tb) / len(ta | tb)
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    s = 0.5 * jd + 0.3 * ja + 0.2 * seq
    if (ta & CAMPUS) != (tb & CAMPUS):
        s -= 0.15
    return round(s, 3)


def norm_addr(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\b(STE|SUITE|BLDG|FL|UNIT)\b.*$", "", s)
    rep = {"DRIVE": "DR", "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "PARKWAY": "PKWY", "HIGHWAY": "HWY",
           "FREEWAY": "FWY", "ROAD": "RD", "LANE": "LN", "EXPRESSWAY": "EXPY", "SOUTH": "S", "NORTH": "N",
           "EAST": "E", "WEST": "W", "LOOP": "LOOP"}
    return " ".join(rep.get(t, t) for t in s.split())


rel = pd.read_csv("hosp_relevance.csv", dtype={"THCIC_ID": str})
rel = rel[rel.relevant].copy()
thc = pd.read_csv("thcic_hospitals_2016_2019.tsv", sep="\t", dtype=str).fillna("")
isr = pd.read_csv("isr_all.tsv", sep="\t", dtype=str, header=None, names=["year", "THCIC_ID", "name", "city", "note"]).fillna("")
pn = pd.read_csv("pudf_names_all.tsv", sep="\t", dtype=str, header=None, names=["THCIC_ID", "name"]).fillna("")
names = (pd.concat([isr[["THCIC_ID", "name"]], pn]).drop_duplicates().groupby("THCIC_ID").name.apply(list).to_dict())
rel = rel.merge(thc[["THCIC_ID", "name", "city", "years_listed", "notes"]], on="THCIC_ID", how="left")
rel["city"] = rel["city"].replace({"Rowlet te": "Rowlett"})

pos = pd.DataFrame(json.load(open("pos_tx_dec19.json", encoding="utf-8")))
cand = [dict(src="POS", cname=r.FAC_NAME, addr=r.ST_ADR, city=r.CITY_NAME, zip=str(r.ZIP_CD)[:5],
             ref=r.PRVDR_NUM, active=r.PGM_TRMNTN_CD == "00") for r in pos.itertuples()]
npi = json.load(open("nppes_tx_hosp.json", encoding="utf-8"))
for r in npi:
    for nm in [r["org"]] + r["other"]:
        cand.append(dict(src="NPI", cname=nm, addr=(r["addr1"] + " " + r["addr2"]).strip(), city=r["city"], zip=r["zip"],
                         ref=r["npi"], active=True, qcity=r["query_city"]))
cand = pd.DataFrame(cand)
cand["city_n"] = cand.city.map(norm_city)
cand["qcity_n"] = cand.get("qcity", pd.Series([""] * len(cand))).fillna("").map(norm_city)
cand["addr_n"] = cand.addr.map(norm_addr)
def loc_key(a):
    t = norm_addr(a).split()
    t = [x for x in t if x not in ("N", "S", "E", "W")]
    return " ".join(t[:2])
cand["loc"] = cand.addr.map(loc_key)
shared = {}  # CCNs used by several THCIC campuses are not campus-specific


rows = []
for h in rel.itertuples():
    cn = norm_city(h.city)
    cc = cand[(cand.city_n == cn) | (cand.qcity_n == cn)]
    hn = names.get(h.THCIC_ID, [h.name])
    scored = [(max(score2(a, c.cname) for a in hn), c) for c in cc.itertuples()]
    best_loc = {}
    for s, c in scored:
        if c.loc not in best_loc or s > best_loc[c.loc][0]:
            best_loc[c.loc] = (s, c)
    ranked = sorted(best_loc.values(), key=lambda t: -t[0])
    top_s, top_c = ranked[0] if ranked else (0, None)
    pos_scored = sorted([(s, c) for s, c in scored if c.src == "POS"], key=lambda t: (-t[0], not t[1].active))
    if pos_scored and pos_scored[0][0] >= 0.8 and pos_scored[0][0] >= top_s - 0.05:
        chosen_s, chosen = pos_scored[0]
    else:
        chosen_s, chosen = top_s, top_c
    rivals = [(s, c) for s, c in ranked if chosen is not None and c.loc != chosen.loc and s >= chosen_s - 0.1]
    rec = dict(THCIC_ID=h.THCIC_ID, name=h.name, city=h.city, stemi=h.stemi_dir + h.stemi_in, pci_max=h.pci_max,
               ed_max=h.ed_max, years=h.years_listed, notes=h.notes, all_names=" | ".join(hn), s1=chosen_s,
               s_rival=max([s for s, _ in rivals], default=0))
    if chosen is not None:
        rec.update(src=chosen.src, cname=chosen.cname, addr=chosen.addr, a_city=chosen.city, a_zip=chosen.zip,
                   ref=chosen.ref, loc=chosen.loc)
    rec["alternatives"] = " || ".join(f"{s:.2f} {c.src}:{c.cname} @ {c.addr}" for s, c in
                                      [t for t in ranked if chosen is None or t[1].loc != chosen.loc][:2])
    rec["status"] = "auto" if (chosen_s >= 0.8 and not rivals) else "review"
    rows.append(rec)

out = pd.DataFrame(rows)
# several THCIC IDs on one location with overlapping reporting years = campuses collapsed onto one address
yrs = out.years.fillna("").str.split(",")
for loc, grp in out[out.status == "auto"].groupby("loc"):
    if len(grp) < 2:
        continue
    ys = [set(yrs[i]) for i in grp.index]
    overlap = any(ys[i] & ys[j] for i in range(len(ys)) for j in range(i + 1, len(ys)))
    if overlap:
        out.loc[grp.index, "status"] = "review"
out.to_csv("hosp_addresses_stage2.csv", index=False)
print(out.status.value_counts().to_string())
print("STEMI admissions by status:", out.groupby("status").stemi.sum().to_dict())
dup = out[out.status == "auto"].groupby("addr").THCIC_ID.transform("count") > 1
print("auto rows sharing an address with another THCIC ID:", int(dup.sum()))
