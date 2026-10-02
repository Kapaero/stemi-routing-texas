"""Texas STEMI routing study (inpatient PUDF 2016-2019): first models.
1) analytic cohort + descriptive table by arrival mode;
2) conditional logit of the first-destination hospital (direct arrivals), train 2017-18 / test 2019, vs rule baselines;
3) first stage: arrival via inter-facility transfer ~ differential distance to a PCI centre;
4) 2SLS for in-hospital death and PCI on the day of admission.
Run from the 'derived' folder; results go to derived/results/."""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

os.makedirs("results", exist_ok=True)
YEARS = [2016, 2017, 2018, 2019]
K = 30  # alternatives per patient: the K nearest eligible hospitals


def hav(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ------------------------------------------------------------------ data
P = pd.read_csv("stemi_patients_2016_2019.tsv", sep="\t", dtype=str)
for c in ["year", "quarter", "los", "weekday", "pci", "pci_day", "cabg", "cath", "died", "hospice", "transfer_out",
          "dm", "htn", "hf", "ckd", "prior_mi", "prior_pci", "prior_cabg", "af", "copd", "cvd", "pvd", "dementia",
          "obesity", "smoking", "cancer", "arrest", "shock"]:
    P[c] = pd.to_numeric(P[c], errors="coerce")
COMORB = ["dm", "htn", "hf", "ckd", "prior_mi", "prior_pci", "prior_cabg", "af", "copd", "cvd", "pvd", "dementia",
          "obesity", "smoking", "cancer"]
Z = pd.read_csv("zcta2010_tx_internal_points.csv", dtype={"zcta5": str}).set_index("zcta5")
H = pd.read_csv("hospitals_tx_2016_2019.csv", dtype={"THCIC_ID": str})
H = H.merge(pd.read_csv("facility_flags.tsv", sep="\t", dtype={"THCIC_ID": str}), on="THCIC_ID", how="left")
HY = pd.read_csv("hosp_year_2016_2019.tsv", sep="\t", dtype={"THCIC_ID": str})

n0 = len(P)
P = P[P.pat_zip.isin(Z.index)].copy()
P["zlat"] = Z.loc[P.pat_zip, "lat"].values
P["zlon"] = Z.loc[P.pat_zip, "lon"].values
P["log_area"] = np.log1p(Z.loc[P.pat_zip, "arealand"].values / 1e6)
P["transfer"] = P.src.isin(["4", "6"]).astype(int)
P["pci_day0"] = ((P.pci == 1) & (P.pci_day <= 0)).astype(int)
P["weekend"] = P.weekday.isin([6, 7]).astype(int)
AGE = {"06": "18-44", "07": "18-44", "08": "18-44", "09": "18-44", "10": "18-44", "11": "18-44", "12": "45-54",
       "13": "45-54", "14": "55-64", "15": "55-64", "16": "65-74", "17": "65-74", "18": "75-84", "19": "75-84",
       "20": "85+", "21": "85+", "23": "18-44", "24": "55-64", "25": "65-74", "26": "75-84"}
P["age"] = P.age_grp.map(AGE)
P["age75"] = P.age.isin(["75-84", "85+"]).astype(int)
PAY = {"MA": "Medicare", "MB": "Medicare", "16": "Medicare", "MC": "Medicaid", "ZZ": "Self-pay", "09": "Self-pay"}
P["payer_g"] = P.payer.map(PAY)
P.loc[P.payer_g.isna() & P.payer.isin(["BL", "CI", "HM", "12", "13", "14", "15"]), "payer_g"] = "Commercial"
P["payer_g"] = P.payer_g.fillna("Other")
P["raceeth"] = np.where(P.ethnicity == "1", "Hispanic", P.race.map({"3": "Black", "4": "White", "2": "Asian/PI"}).fillna("Other"))
P["female"] = (P.sex == "F").astype(int)
P = P.merge(H[["THCIC_ID", "lat", "lon"]].rename(columns={"lat": "hlat", "lon": "hlon"}), on="THCIC_ID", how="left")
P["d_admit"] = hav(P.zlat, P.zlon, P.hlat, P.hlon)
print(f"cohort: {n0} STEMI | with ZIP centroid: {len(P)} | admitting hospital geocoded: {P.hlat.notna().sum()}")

# ------------------------------------------------------------------ hospitals eligible as EMS destinations, by year
vol = P.groupby(["THCIC_ID", "year"]).size()
hy = HY.set_index(["THCIC_ID", "year"])
H["childrens"] = H.name.str.contains("child", case=False, na=False)


def ed_year(h, y):
    if (h, y) not in hy.index:
        return False
    r = hy.loc[(h, y)]
    return bool(r.ed_flag_admits >= 100) if y >= 2017 else bool(r.emerg_urgent_admits >= 300)


HS = {}
for y in YEARS:
    cap = f"pci_capable_{y - 1}" if y > 2016 else f"pci_capable_{y}"  # capability known before the year starts
    s = H[H[f"active_{y}"] & ~H.childrens].copy()  # FAC_PSYCH_IND also marks general hospitals with psych units
    s["ed"] = [ed_year(h, y) for h in s.THCIC_ID]
    s["pci_prev"] = s[cap].fillna(False).astype(int)
    s = s[s.ed | (s.pci_prev == 1) | s[f"pci_capable_{y}"]].reset_index(drop=True)
    yv = y - 1 if y > 2016 else y
    s["vol_prev"] = [vol.get((h, yv), 0) for h in s.THCIC_ID]
    s["teach"] = s.teaching.fillna(0).astype(int)
    HS[y] = s
print("eligible destination hospitals by year:", {y: len(s) for y, s in HS.items()},
      "| PCI-capable (prior year):", {y: int(s.pci_prev.sum()) for y, s in HS.items()})

# access measures with the same hospital sets
P["d_any"] = np.nan
P["d_pci"] = np.nan
P["nearest_any"] = ""
P["nearest_pci"] = ""
for y in YEARS:
    i = np.where(P.year.values == y)[0]
    s = HS[y]
    pc = s[s.pci_prev == 1]
    dA = hav(P.zlat.values[i, None], P.zlon.values[i, None], s.lat.values[None, :], s.lon.values[None, :])
    dP = hav(P.zlat.values[i, None], P.zlon.values[i, None], pc.lat.values[None, :], pc.lon.values[None, :])
    P.iloc[i, P.columns.get_loc("d_any")] = dA.min(1)
    P.iloc[i, P.columns.get_loc("d_pci")] = dP.min(1)
    P.iloc[i, P.columns.get_loc("nearest_any")] = s.THCIC_ID.to_numpy(dtype=object)[dA.argmin(1)]
    P.iloc[i, P.columns.get_loc("nearest_pci")] = pc.THCIC_ID.to_numpy(dtype=object)[dP.argmin(1)]
P["dd"] = P.d_pci - P.d_any

# ------------------------------------------------------------------ 1) descriptive table
def desc(g):
    return pd.Series({
        "N": len(g), "age 75+ %": 100 * g.age75.mean(), "female %": 100 * g.female.mean(),
        "Hispanic %": 100 * (g.raceeth == "Hispanic").mean(), "Black %": 100 * (g.raceeth == "Black").mean(),
        "Medicare %": 100 * (g.payer_g == "Medicare").mean(), "self-pay %": 100 * (g.payer_g == "Self-pay").mean(),
        "cardiac arrest %": 100 * g.arrest.mean(), "cardiogenic shock %": 100 * g.shock.mean(),
        "anterior STEMI %": 100 * (g.stemi_site == "anterior").mean(),
        "km to nearest hospital (median)": g.d_any.median(), "km to nearest PCI centre (median)": g.d_pci.median(),
        "differential distance >5 km %": 100 * (g.dd > 5).mean(), "km to admitting hospital (median)": g.d_admit.median(),
        "PCI on day 0 %": 100 * g.pci_day0.mean(), "PCI any %": 100 * g.pci.mean(), "in-hospital death %": 100 * g.died.mean(),
    })


T1 = pd.concat({"direct": desc(P[P.transfer == 0]), "via transfer": desc(P[P.transfer == 1]), "all": desc(P)}, axis=1).round(1)
T1.to_csv("results/table1_by_arrival.csv")
print("\nTable 1 (by arrival mode):\n" + T1.to_string())

# ------------------------------------------------------------------ 2) conditional logit of the first destination
FEATS = ["dist_10km", "log1p_dist", "nearest", "nearest_pci", "pci_capable", "log_vol_prev", "teaching",
         "dist_x_age75", "dist_x_arrest", "pci_x_arrest", "dist_x_rural"]
RURAL_CUT = P.log_area.median()


def build(df, y):
    s = HS[y]
    d = hav(df.zlat.values[:, None], df.zlon.values[:, None], s.lat.values[None, :], s.lon.values[None, :])
    order = np.argsort(d, axis=1)[:, :K]
    dk = np.take_along_axis(d, order, 1)
    ids = s.THCIC_ID.to_numpy(dtype=object)[order]
    ch = ids == df.THCIC_ID.to_numpy(dtype=object)[:, None]
    pci = s.pci_prev.values[order].astype(float)
    near = np.zeros_like(dk); near[:, 0] = 1
    dpc = np.where(pci == 1, dk, np.inf)
    npc = np.zeros_like(dk)
    has = np.isfinite(dpc.min(1))
    npc[np.where(has)[0], dpc.argmin(1)[has]] = 1
    a75 = df.age75.values[:, None]; arr = df.arrest.values[:, None]; rur = (df.log_area.values > RURAL_CUT)[:, None]
    X = np.stack([dk / 10, np.log1p(dk), near, npc, pci, np.log1p(s.vol_prev.values[order]), s.teach.values[order],
                  dk / 10 * a75, dk / 10 * arr, pci * arr, dk / 10 * rur], axis=2)
    keep = ch.any(1)
    return dict(X=X[keep], y=ch[keep].argmax(1), d=dk[keep], pci=pci[keep], vol=s.vol_prev.values[order][keep],
                keep=keep, ids=ids[keep])


def stack(df, years):
    parts = [build(df[df.year == y], y) for y in years]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ["X", "y", "d", "pci", "vol", "ids"]}
    out["n_all"] = sum(len(p["keep"]) for p in parts)
    return out


def nll(beta, X, y):
    u = X @ beta
    u = u - u.max(1, keepdims=True)
    e = np.exp(u)
    p = e / e.sum(1, keepdims=True)
    idx = np.arange(len(y))
    ll = np.log(p[idx, y] + 1e-300).sum()
    g = (X[idx, y] - (p[:, :, None] * X).sum(1)).sum(0)
    return -ll, -g


def fit_clogit(X, y):
    r = minimize(nll, np.zeros(X.shape[2]), args=(X, y), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    u = X @ r.x
    u -= u.max(1, keepdims=True)
    p = np.exp(u); p /= p.sum(1, keepdims=True)
    xb = (p[:, :, None] * X).sum(1)
    Hs = np.einsum("nk,nki,nkj->ij", p, X, X) - xb.T @ xb
    se = np.sqrt(np.diag(np.linalg.pinv(Hs)))
    return r.x, se, -r.fun


def rank_metrics(score, y):
    order = np.argsort(-score, axis=1, kind="stable")
    rank = (order == y[:, None]).argmax(1) + 1
    return {"Top-1": (rank == 1).mean(), "Top-3": (rank <= 3).mean(), "Top-5": (rank <= 5).mean(),
            "MRR": (1 / rank).mean(), "mean rank": rank.mean()}


D = P[P.transfer == 0]
tr, te = stack(D, [2017, 2018]), stack(D, [2019])
print(f"\nchoice model: train {len(tr['y'])} of {tr['n_all']} direct arrivals 2017-18 "
      f"(chosen hospital among {K} nearest), test {len(te['y'])} of {te['n_all']} in 2019")
beta, se, ll = fit_clogit(tr["X"], tr["y"])
coef = pd.DataFrame({"coef": beta, "se": se, "z": beta / se}, index=FEATS).round(3)
coef.to_csv("results/clogit_coefficients.csv")
print(coef.to_string())
fi = [FEATS.index(f) for f in ["dist_10km", "log1p_dist", "nearest"]]
bd, _, _ = fit_clogit(tr["X"][:, :, fi], tr["y"])

rows = {}
n = len(te["y"])
rows["nearest hospital"] = rank_metrics(-te["d"], te["y"])
rows["nearest PCI centre"] = rank_metrics(-te["d"] + 1e4 * te["pci"], te["y"])
rows["largest prior-year volume"] = rank_metrics(te["vol"] - 1e-3 * te["d"], te["y"])
rows["logit: distance only"] = rank_metrics(te["X"][:, :, fi] @ bd, te["y"])
rows["logit: full"] = rank_metrics(te["X"] @ beta, te["y"])
M = pd.DataFrame(rows).T
u = te["X"] @ beta
top1 = u.argmax(1)
M["top-1 is PCI centre %"] = np.nan
M.loc["logit: full", "top-1 is PCI centre %"] = 100 * te["pci"][np.arange(n), top1].mean()
M.loc["nearest hospital", "top-1 is PCI centre %"] = 100 * te["pci"][:, 0].mean()
M.loc["nearest PCI centre", "top-1 is PCI centre %"] = 100.0
M = M.round(3)
M.to_csv("results/choice_model_test2019.csv")
print("\nTest 2019 (direct arrivals):\n" + M.to_string())
print(f"observed: admitted to a PCI centre {100 * te['pci'][np.arange(n), te['y']].mean():.1f}% | "
      f"to the nearest hospital {100 * (te['y'] == 0).mean():.1f}%")

# ------------------------------------------------------------------ 3-4) first stage and 2SLS
S = P[~((P.transfer == 0) & (P.transfer_out == 1))].copy()  # first legs of later transfers are counted at the receiver
S["dd_gt5"] = (S.dd > 5).astype(float)
S["dd_gt20"] = (S.dd > 20).astype(float)
S["log1p_d_any"] = np.log1p(S.d_any)


def design(df):
    X = pd.get_dummies(df[["age", "raceeth", "payer_g", "stemi_site"]].astype(str), drop_first=False, dtype=float)
    X = X.drop(columns=["age_65-74", "raceeth_White", "payer_g_Medicare", "stemi_site_inferior"])
    X = pd.concat([X, df[COMORB + ["female", "arrest", "shock", "weekend", "log1p_d_any", "log_area"]].astype(float),
                   pd.get_dummies(df.year.astype(str), prefix="y", drop_first=True, dtype=float),
                   pd.get_dummies(df.phr.fillna("NA"), prefix="phr", drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    return X.loc[:, X.std() > 0].assign(const=1.0)


def cluster_vcov(Xh, u, cl, bread):
    G = pd.Series(cl).astype("category").cat.codes.values
    S_ = np.zeros((Xh.shape[1], Xh.shape[1]))
    m = Xh * u[:, None]
    agg = np.zeros((G.max() + 1, Xh.shape[1]))
    np.add.at(agg, G, m)
    S_ = agg.T @ agg
    ng, nn, kk = G.max() + 1, len(u), Xh.shape[1]
    c = ng / (ng - 1) * (nn - 1) / (nn - kk)
    return c * bread @ S_ @ bread


def tsls(df, yname, tname, znames):
    X = design(df)
    y = df[yname].values.astype(float)
    t = df[tname].values.astype(float)
    Zm = df[znames].values.astype(float)
    cl = df.pat_zip.to_numpy(dtype=object)
    W = np.column_stack([t, X.values])
    Zf = np.column_stack([Zm, X.values])
    # first stage
    bf, *_ = np.linalg.lstsq(Zf, t, rcond=None)
    uf = t - Zf @ bf
    Vf = cluster_vcov(Zf, uf, cl, np.linalg.inv(Zf.T @ Zf))
    q = Zm.shape[1]
    bz, Vz = bf[:q], Vf[:q, :q]
    F = float(bz @ np.linalg.solve(Vz, bz) / q)
    # 2SLS
    What = Zf @ np.linalg.lstsq(Zf, W, rcond=None)[0]
    b = np.linalg.solve(What.T @ W, What.T @ y)
    u = y - W @ b
    V = cluster_vcov(What, u, cl, np.linalg.inv(What.T @ What))
    # OLS
    bo, *_ = np.linalg.lstsq(W, y, rcond=None)
    Vo = cluster_vcov(W, y - W @ bo, cl, np.linalg.inv(W.T @ W))
    return {"outcome": yname, "N": len(y), "mean outcome %": 100 * y.mean(), "first-stage coef": np.round(bz, 4).tolist(),
            "first-stage F": F, "OLS effect (pp)": 100 * bo[0], "OLS se": 100 * np.sqrt(Vo[0, 0]),
            "2SLS effect (pp)": 100 * b[0], "2SLS se": 100 * np.sqrt(V[0, 0]),
            "2SLS 95% CI": (round(100 * (b[0] - 1.96 * np.sqrt(V[0, 0])), 1), round(100 * (b[0] + 1.96 * np.sqrt(V[0, 0])), 1))}


S = S.dropna(subset=["dd", "d_any", "age"])
# routing failure = first hospital is not PCI-capable: arrived via transfer, or admitted directly to a non-PCI hospital
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
S["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(S.THCIC_ID, S.year)]
S["first_nonpci"] = ((S.transfer == 1) | (S.admit_pci == 0)).astype(int)
S["died_or_hospice"] = ((S.died == 1) | (S.hospice == 1)).astype(int)
S["log1p_dd"] = np.log1p(S.dd.clip(lower=0))
res = []
for tname in ["first_nonpci", "transfer"]:
    for z in (["dd_gt5", "dd_gt20"], ["log1p_dd"]):
        for yv in ["died", "died_or_hospice", "pci_day0"]:
            r = tsls(S, yv, tname, z)
            r["treatment"] = tname
            r["instruments"] = "+".join(z)
            res.append(r)
R = pd.DataFrame(res)
R.to_csv("results/iv_results.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nIV sample: {len(S)} STEMI (first legs of later transfers removed) | via transfer {100 * S.transfer.mean():.1f}% | "
      f"first hospital non-PCI {100 * S.first_nonpci.mean():.1f}%")
cols = ["treatment", "instruments", "outcome", "N", "mean outcome %", "first-stage F", "OLS effect (pp)", "OLS se",
        "2SLS effect (pp)", "2SLS se", "2SLS 95% CI"]
print(R[cols].round(2).to_string(index=False))

# ------------------------------------------------------------------ 5) scenario: "take to the nearest PCI centre if the detour is <= X km"
main = R[(R.treatment == "first_nonpci") & (R.instruments == "dd_gt5+dd_gt20") & (R.outcome == "pci_day0")].iloc[0]
eff, eff_se = main["2SLS effect (pp)"] / 100, main["2SLS se"] / 100
ny = S.year.nunique()
sc = []
for X in [5, 10, 20, 30, 50]:
    g = S[(S.dd > 0.01) & (S.dd <= X)]
    fail = g.first_nonpci.sum() / ny
    sc.append({"detour limit, km": X,
               "STEMI/yr with non-PCI nearest hospital and detour <= limit": len(g) / ny,
               "of them first hospital non-PCI, /yr": fail,
               "extra same-day PCI/yr if routed directly": -eff * fail,
               "95% CI": (round(-(eff + 1.96 * eff_se) * fail, 1), round(-(eff - 1.96 * eff_se) * fail, 1)),
               "mean detour, km": g.dd.mean()})
SC = pd.DataFrame(sc).round(1)
SC.to_csv("results/scenario_detour_rule.csv", index=False)
print("\nScenario (2SLS effect of a non-PCI first hospital on same-day PCI applied to current routing failures):\n"
      + SC.to_string(index=False))

# ------------------------------------------------------------------ 6) instrument balance: observed risk by differential-distance group
from sklearn.linear_model import LogisticRegression

risk_X = pd.get_dummies(S[["age", "raceeth", "payer_g", "stemi_site"]].astype(str), dtype=float)
risk_X = pd.concat([risk_X, S[COMORB + ["female", "arrest", "shock"]].astype(float)], axis=1)
lr = LogisticRegression(max_iter=2000).fit(risk_X, S.died)
S["pred_death"] = lr.predict_proba(risk_X)[:, 1]
S["dd_group"] = pd.cut(S.dd, [-0.01, 5, 20, 1e4], labels=["<=5 km", "5-20 km", ">20 km"])
B = S.groupby("dd_group", observed=True).agg(
    N=("died", "size"), age75=("age75", "mean"), female=("female", "mean"), hispanic=("raceeth", lambda x: (x == "Hispanic").mean()),
    medicare=("payer_g", lambda x: (x == "Medicare").mean()), selfpay=("payer_g", lambda x: (x == "Self-pay").mean()),
    diabetes=("dm", "mean"), heart_failure=("hf", "mean"), ckd=("ckd", "mean"), arrest=("arrest", "mean"),
    shock=("shock", "mean"), anterior=("stemi_site", lambda x: (x == "anterior").mean()),
    predicted_death=("pred_death", "mean"), first_nonpci=("first_nonpci", "mean"), pci_day0=("pci_day0", "mean"),
    died=("died", "mean"))
B.iloc[:, 1:] = (100 * B.iloc[:, 1:]).round(1)
B.to_csv("results/instrument_balance.csv")
print("\nBalance by differential distance (%; predicted death from demographics + POA comorbidities only):\n" + B.T.to_string())
