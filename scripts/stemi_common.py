"""Shared data preparation for the Texas STEMI routing models (run scripts from the 'derived' folder)."""
import numpy as np
import pandas as pd

YEARS = [2016, 2017, 2018, 2019]
K = 30  # alternatives per patient: the K nearest eligible hospitals
COMORB = ["dm", "htn", "hf", "ckd", "prior_mi", "prior_pci", "prior_cabg", "af", "copd", "cvd", "pvd", "dementia",
          "obesity", "smoking", "cancer"]
AGE = {"06": "18-44", "07": "18-44", "08": "18-44", "09": "18-44", "10": "18-44", "11": "18-44", "12": "45-54",
       "13": "45-54", "14": "55-64", "15": "55-64", "16": "65-74", "17": "65-74", "18": "75-84", "19": "75-84",
       "20": "85+", "21": "85+", "23": "18-44", "24": "55-64", "25": "65-74", "26": "75-84"}
AGE_ORD = {"18-44": 0, "45-54": 1, "55-64": 2, "65-74": 3, "75-84": 4, "85+": 5}
PAY = {"MA": "Medicare", "MB": "Medicare", "16": "Medicare", "MC": "Medicaid", "ZZ": "Self-pay", "09": "Self-pay"}
LOGIT_FEATS = ["dist_10km", "log1p_dist", "nearest", "nearest_pci", "pci_capable", "log_vol_prev", "teaching",
               "dist_x_age75", "dist_x_arrest", "pci_x_arrest", "dist_x_rural"]
ALT_FEATS = ["dist_km", "log1p_dist", "rank_frac", "nearest", "dist_minus_min", "pci_capable", "nearest_pci",
             "dist_minus_nearest_pci", "log_vol_prev", "teaching", "prev_pci_day0_rate", "prev_transfer_in_share"]
PAT_BASE = ["age_ord", "female", "arrest", "shock", "site_anterior", "site_inferior", "weekend", "log_area",
            "log1p_d_any", "log1p_dd"]


def hav(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load(pci_threshold=10):
    P = pd.read_csv("stemi_patients_2016_2019.tsv", sep="\t", dtype=str)
    for c in ["year", "quarter", "los", "weekday", "pci", "pci_day", "cabg", "cath", "died", "hospice",
              "transfer_out", "arrest", "shock"] + COMORB:
        P[c] = pd.to_numeric(P[c], errors="coerce")
    Z = pd.read_csv("zcta2010_tx_internal_points.csv", dtype={"zcta5": str}).set_index("zcta5")
    H = pd.read_csv("hospitals_tx_2016_2019.csv", dtype={"THCIC_ID": str})
    H = H.merge(pd.read_csv("facility_flags.tsv", sep="\t", dtype={"THCIC_ID": str}), on="THCIC_ID", how="left")
    HY = pd.read_csv("hosp_year_2016_2019.tsv", sep="\t", dtype={"THCIC_ID": str})

    P = P[P.pat_zip.isin(Z.index)].copy()
    P["zlat"] = Z.loc[P.pat_zip, "lat"].values
    P["zlon"] = Z.loc[P.pat_zip, "lon"].values
    P["log_area"] = np.log1p(Z.loc[P.pat_zip, "arealand"].values / 1e6)
    P["transfer"] = P.src.isin(["4", "6"]).astype(int)
    P["pci_day0"] = ((P.pci == 1) & (P.pci_day <= 0)).astype(int)
    P["weekend"] = P.weekday.isin([6, 7]).astype(int)
    P["age"] = P.age_grp.map(AGE)
    P["age_ord"] = P.age.map(AGE_ORD).astype(float)
    P["age75"] = P.age.isin(["75-84", "85+"]).astype(int)
    P["payer_g"] = P.payer.map(PAY)
    P.loc[P.payer_g.isna() & P.payer.isin(["BL", "CI", "HM", "12", "13", "14", "15"]), "payer_g"] = "Commercial"
    P["payer_g"] = P.payer_g.fillna("Other")
    P["raceeth"] = np.where(P.ethnicity == "1", "Hispanic",
                            P.race.map({"3": "Black", "4": "White", "2": "Asian/PI"}).fillna("Other"))
    P["female"] = (P.sex == "F").astype(int)
    P["site_anterior"] = (P.stemi_site == "anterior").astype(int)
    P["site_inferior"] = (P.stemi_site == "inferior").astype(int)
    P = P.reset_index(drop=True)

    # eligible EMS destinations by year, with attributes known before the year starts
    hy = HY.set_index(["THCIC_ID", "year"])
    H["childrens"] = H.name.str.contains("child", case=False, na=False)
    grp = P.groupby(["THCIC_ID", "year"])
    vol, pday0, tshare = grp.size(), grp.pci_day0.mean(), grp.transfer.mean()

    def ed_year(h, y):
        if (h, y) not in hy.index:
            return False
        r = hy.loc[(h, y)]
        return bool(r.ed_flag_admits >= 100) if y >= 2017 else bool(r.emerg_urgent_admits >= 300)

    HS = {}
    for y in YEARS:
        prev = y - 1 if y > 2016 else y
        s = H[H[f"active_{y}"] & ~H.childrens].copy()
        s["ed"] = [ed_year(h, y) for h in s.THCIC_ID]
        s["pci_prev"] = (s[f"pci_ami_{prev}"].fillna(0) >= pci_threshold).astype(int)  # threshold 10 = pci_capable
        s = s[s.ed | (s.pci_prev == 1) | s[f"pci_capable_{y}"]].reset_index(drop=True)
        s["vol_prev"] = [vol.get((h, prev), 0) for h in s.THCIC_ID]
        s["pday0_prev"] = [pday0.get((h, prev), 0.0) for h in s.THCIC_ID]
        s["tshare_prev"] = [tshare.get((h, prev), 0.0) for h in s.THCIC_ID]
        s["teach"] = s.teaching.fillna(0).astype(int)
        HS[y] = s

    # patient access geography (same hospital sets)
    for c in ["d_any", "d_pci"]:
        P[c] = np.nan
    for y in YEARS:
        i = np.where(P.year.values == y)[0]
        s = HS[y]
        pc = s[s.pci_prev == 1]
        P.loc[i, "d_any"] = hav(P.zlat.values[i, None], P.zlon.values[i, None], s.lat.values[None], s.lon.values[None]).min(1)
        P.loc[i, "d_pci"] = hav(P.zlat.values[i, None], P.zlon.values[i, None], pc.lat.values[None], pc.lon.values[None]).min(1)
    P["dd"] = P.d_pci - P.d_any
    P["log1p_d_any"] = np.log1p(P.d_any)
    P["log1p_dd"] = np.log1p(P.dd.clip(lower=0))
    return P, H, HS


def build(df, y, HS, rural_cut, tmap=None):
    """Choice sets of the K nearest eligible hospitals (straight line); returns arrays for linear logit and ML rankers.
    With tmap {(zip, THCIC_ID): road minutes} the same choice sets are re-ranked and described by road travel time."""
    s = HS[y]
    d = hav(df.zlat.values[:, None], df.zlon.values[:, None], s.lat.values[None, :], s.lon.values[None, :])
    order = np.argsort(d, axis=1)[:, :K]
    dk = np.take_along_axis(d, order, 1)
    if tmap is not None:
        zips = df.pat_zip.to_numpy(dtype=object)
        hid = s.THCIC_ID.to_numpy(dtype=object)
        tk = np.array([[tmap.get((z, hid[j]), np.nan) for j in row] for z, row in zip(zips, order)])
        tk = np.where(np.isnan(tk), dk * 0.9, tk)  # rare gaps: ~67 km/h incl. circuity
        o2 = np.argsort(tk, axis=1)
        order = np.take_along_axis(order, o2, 1)
        dk = np.take_along_axis(tk, o2, 1)  # from here on "distance" = minutes
    ids = s.THCIC_ID.to_numpy(dtype=object)[order]
    ch = ids == df.THCIC_ID.to_numpy(dtype=object)[:, None]
    keep = ch.any(1)
    pci = s.pci_prev.values[order].astype(float)
    near = np.zeros_like(dk); near[:, 0] = 1
    dpc = np.where(pci == 1, dk, np.inf)
    has = np.isfinite(dpc.min(1))
    npc = np.zeros_like(dk)
    npc[np.where(has)[0], dpc.argmin(1)[has]] = 1
    d_npc = np.where(has, dpc.min(1), dk[:, -1])[:, None]
    lvol = np.log1p(s.vol_prev.values[order])
    teach = s.teach.values[order].astype(float)
    a75 = df.age75.values[:, None]; arr = df.arrest.values[:, None]; rur = (df.log_area.values > rural_cut)[:, None]
    X_logit = np.stack([dk / 10, np.log1p(dk), near, npc, pci, lvol, teach,
                        dk / 10 * a75, dk / 10 * arr, pci * arr, dk / 10 * rur], axis=2)
    X_alt = np.stack([dk, np.log1p(dk), np.tile(np.arange(1, K + 1) / K, (len(dk), 1)), near, dk - dk[:, :1], pci, npc,
                      dk - d_npc, lvol, teach, s.pday0_prev.values[order], s.tshare_prev.values[order]], axis=2)
    phr = pd.get_dummies(df.phr.fillna("NA"), prefix="phr", dtype=float)
    return dict(X_logit=X_logit[keep], X_alt=X_alt[keep], pat=df[PAT_BASE].to_numpy(float)[keep], phr=phr[keep],
                ids=ids[keep], y=ch[keep].argmax(1), d=dk[keep], pci=pci[keep], vol=s.vol_prev.values[order][keep],
                rows=df.index.values[keep], n_all=len(df))


def stack(df, years, HS, rural_cut, tmap=None):
    parts = [build(df[df.year == y], y, HS, rural_cut, tmap) for y in years]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ["X_logit", "X_alt", "pat", "ids", "y", "d", "pci", "vol", "rows"]}
    out["phr"] = pd.concat([p["phr"] for p in parts]).fillna(0.0)
    out["n_all"] = sum(p["n_all"] for p in parts)
    return out


def rank_metrics(score, y):
    order = np.argsort(-score, axis=1, kind="stable")
    rank = (order == y[:, None]).argmax(1) + 1
    return {"Top-1": (rank == 1).mean(), "Top-3": (rank <= 3).mean(), "Top-5": (rank <= 5).mean(),
            "MRR": (1 / rank).mean(), "mean rank": rank.mean()}
