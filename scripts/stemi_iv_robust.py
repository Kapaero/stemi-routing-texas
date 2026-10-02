"""IV robustness and inference for the effect of a non-PCI first hospital on STEMI outcomes.
Specs: baseline (km instruments), + ZIP SES/rurality, road travel-time instruments, sample restrictions,
alternative cut-offs, continuous instrument, leave-one-region-out.
Inference: cluster-robust (patient ZIP) SEs and p-values, Anderson-Rubin weak-IV-robust CI, Hansen J (over-id),
control-function endogeneity test, instrument balance (placebo) test. Run from 'derived'."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "scripts")
from stemi_common import COMORB, YEARS, load  # noqa: E402

P, H, HS = load()
C = pd.read_csv("ext/zip_covariates.csv", dtype={"pat_zip": str})
SES = ["log_income", "pct_poverty", "pct_uninsured", "pct_65plus", "pct_bachelor", "log_pop_density"]
P = P.merge(C[["pat_zip", "ruca_cat"] + SES], on="pat_zip", how="left")
P["ses_missing"] = P[SES].isna().any(axis=1).astype(float)
for c in SES:
    P[c] = P[c].fillna(P[c].median())
P["ruca_cat"] = P.ruca_cat.fillna("metro")

# ---- road travel-time access (OSRM), same hospital sets as the km measures
HAVE_T = os.path.exists("ext/osrm_zip_hosp.csv")
if HAVE_T:
    T = pd.read_csv("ext/osrm_zip_hosp.csv", dtype={"pat_zip": str, "THCIC_ID": str}).dropna(subset=["minutes"])
    P["t_any"] = np.nan
    P["t_pci"] = np.nan
    for y in YEARS:
        s = HS[y]
        ta = T[T.THCIC_ID.isin(s.THCIC_ID)].groupby("pat_zip").minutes.min()
        tp = T[T.THCIC_ID.isin(s.THCIC_ID[s.pci_prev == 1])].groupby("pat_zip").minutes.min()
        i = P.year == y
        P.loc[i, "t_any"] = P.loc[i, "pat_zip"].map(ta).values
        P.loc[i, "t_pci"] = P.loc[i, "pat_zip"].map(tp).values
    fb = P.t_pci.isna() | P.t_any.isna()
    P.loc[P.t_any.isna(), "t_any"] = P.d_any * 0.9   # fallback ~67 km/h incl. circuity (rare, remote ZIPs)
    P.loc[P.t_pci.isna(), "t_pci"] = P.d_pci * 0.9
    P["dt"] = (P.t_pci - P.t_any).clip(lower=0)
    print(f"travel-time access: fallback to straight line for {100 * fb.mean():.2f}% of patients | "
          f"median min to nearest hospital {P.t_any.median():.1f}, to nearest PCI {P.t_pci.median():.1f}")

# ---- analytic sample and variables (as in stemi_models.py)
S = P[~((P.transfer == 0) & (P.transfer_out == 1))].dropna(subset=["dd", "d_any", "age"]).copy()
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
S["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(S.THCIC_ID, S.year)]
S["first_nonpci"] = ((S.transfer == 1) | (S.admit_pci == 0)).astype(int)
S["died_or_hospice"] = ((S.died == 1) | (S.hospice == 1)).astype(int)
for c, v in [("dd_gt5", 5), ("dd_gt10", 10), ("dd_gt20", 20), ("dd_gt30", 30)]:
    S[c] = (S.dd > v).astype(float)
S["log1p_dd"] = np.log1p(S.dd.clip(lower=0))
if HAVE_T:
    S["dt_gt5"] = (S.dt > 5).astype(float)
    S["dt_gt15"] = (S.dt > 15).astype(float)
    S["log1p_t_any"] = np.log1p(S.t_any)
S = S.reset_index(drop=True)


def controls(df, ses=False, time_access=False, drop=()):
    X = pd.get_dummies(df[["age", "raceeth", "payer_g", "stemi_site"]].astype(str), dtype=float)
    X = X.drop(columns=["age_65-74", "raceeth_White", "payer_g_Medicare", "stemi_site_inferior"], errors="ignore")
    cols = [c for c in COMORB + ["female", "arrest", "shock", "weekend", "log_area"] if c not in drop]
    cols.append("log1p_t_any" if time_access else "log1p_d_any")
    parts = [X, df[cols].astype(float),
             pd.get_dummies(df.year.astype(str), prefix="y", drop_first=True, dtype=float),
             pd.get_dummies(df.phr.fillna("NA"), prefix="phr", drop_first=True, dtype=float)]
    if ses:
        parts += [df[SES + ["ses_missing"]].astype(float),
                  pd.get_dummies(df.ruca_cat.astype(str), prefix="ruca", dtype=float).drop(columns="ruca_metro", errors="ignore")]
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    X = X.loc[:, ~X.T.duplicated()]
    X.insert(0, "const", 1.0)
    return X.to_numpy(float)


def clus(Xm, u, G, bread):
    agg = np.zeros((G.max() + 1, Xm.shape[1]))
    np.add.at(agg, G, Xm * u[:, None])
    n, k, g = len(u), Xm.shape[1], G.max() + 1
    return g / (g - 1) * (n - 1) / (n - k) * bread @ (agg.T @ agg) @ bread, agg


def iv(df, y, t, znames, ses=False, time_access=False, ar_grid=None, drop=()):
    X = controls(df, ses, time_access, drop)
    G = pd.Series(df.pat_zip.to_numpy(dtype=object)).astype("category").cat.codes.values
    yv, tv, Zm = df[y].to_numpy(float), df[t].to_numpy(float), df[znames].to_numpy(float)
    W, Zf = np.column_stack([tv, X]), np.column_stack([Zm, X])
    q = Zm.shape[1]
    ZZi = np.linalg.inv(Zf.T @ Zf)
    # first stage
    bf = ZZi @ Zf.T @ tv
    vf = tv - Zf @ bf
    Vf, _ = clus(Zf, vf, G, ZZi)
    F = float(bf[:q] @ np.linalg.solve(Vf[:q, :q], bf[:q]) / q)
    # effective F (Montiel Olea & Pflueger 2013): instruments partialled out on controls, cluster-robust
    Zt = Zm - X @ np.linalg.lstsq(X, Zm, rcond=None)[0]
    pi = np.linalg.lstsq(Zt, tv - X @ np.linalg.lstsq(X, tv, rcond=None)[0], rcond=None)[0]
    vt = tv - X @ np.linalg.lstsq(X, tv, rcond=None)[0] - Zt @ pi
    Qzz = Zt.T @ Zt / len(tv)
    Vpi, _ = clus(Zt, vt, G, np.linalg.inv(Zt.T @ Zt))
    F_eff = float(pi @ Qzz @ pi / np.trace(Qzz @ Vpi))
    # 2SLS
    What = Zf @ (ZZi @ Zf.T @ W)
    b = np.linalg.solve(What.T @ W, What.T @ yv)
    u = yv - W @ b
    V, _ = clus(What, u, G, np.linalg.inv(What.T @ What))
    se = np.sqrt(V[0, 0])
    p = 2 * stats.norm.sf(abs(b[0] / se))
    # OLS
    bo = np.linalg.lstsq(W, yv, rcond=None)[0]
    Vo, _ = clus(W, yv - W @ bo, G, np.linalg.inv(W.T @ W))
    # control-function endogeneity test
    Wc = np.column_stack([W, vf])
    bc = np.linalg.lstsq(Wc, yv, rcond=None)[0]
    Vc, _ = clus(Wc, yv - Wc @ bc, G, np.linalg.inv(Wc.T @ Wc))
    p_endog = 2 * stats.norm.sf(abs(bc[-1] / np.sqrt(Vc[-1, -1])))
    # Hansen J (two-step GMM) when over-identified
    pJ = np.nan
    if q > 1:
        _, A = clus(Zf, u, G, np.eye(Zf.shape[1]))
        Sm = A.T @ A
        Si = np.linalg.pinv(Sm)
        ZW, Zy = Zf.T @ W, Zf.T @ yv
        bg = np.linalg.solve(ZW.T @ Si @ ZW, ZW.T @ Si @ Zy)
        _, A2 = clus(Zf, yv - W @ bg, G, np.eye(Zf.shape[1]))
        gsum = A2.sum(0)
        J = float(gsum @ np.linalg.pinv(A2.T @ A2) @ gsum)
        pJ = stats.chi2.sf(J, q - 1)
    # Anderson-Rubin CI by test inversion: regress y - beta*t on [Z, X], test Z = 0
    ar = (np.nan, np.nan)
    if ar_grid is not None:
        by, bt = ZZi @ Zf.T @ yv, ZZi @ Zf.T @ tv
        uy, ut = yv - Zf @ by, tv - Zf @ bt
        ny, nt = np.zeros((G.max() + 1, Zf.shape[1])), np.zeros((G.max() + 1, Zf.shape[1]))
        np.add.at(ny, G, Zf * uy[:, None])
        np.add.at(nt, G, Zf * ut[:, None])
        n, k, g = len(yv), Zf.shape[1], G.max() + 1
        c = g / (g - 1) * (n - 1) / (n - k)
        keep = []
        for beta in ar_grid:
            m = ny - beta * nt
            Vb = c * ZZi @ (m.T @ m) @ ZZi
            bz = (by - beta * bt)[:q]
            stat = float(bz @ np.linalg.solve(Vb[:q, :q], bz))
            if stats.chi2.sf(stat, q) > 0.05:
                keep.append(beta)
        if keep:
            lo_open = keep[0] <= ar_grid[0] + 1e-12
            hi_open = keep[-1] >= ar_grid[-1] - 1e-12
            ar = ("<" + f"{100 * ar_grid[0]:.0f}" if lo_open else round(100 * keep[0], 1),
                  ">" + f"{100 * ar_grid[-1]:.0f}" if hi_open else round(100 * keep[-1], 1))
    return {"N": len(yv), "first-stage F": round(F, 1), "effective F": round(F_eff, 1), "OLS (pp)": round(100 * bo[0], 2),
            "2SLS (pp)": round(100 * b[0], 2), "SE": round(100 * se, 2),
            "95% CI": (round(100 * (b[0] - 1.96 * se), 1), round(100 * (b[0] + 1.96 * se), 1)),
            "p": float(f"{p:.2g}"), "AR 95% CI": ar, "Hansen J p": None if np.isnan(pJ) else round(pJ, 3),
            "endogeneity p": float(f"{p_endog:.2g}")}


GRID = {"pci_day0": np.round(np.arange(-0.70, 0.301, 0.0025), 4), "died": np.round(np.arange(-0.30, 0.301, 0.0025), 4),
        "died_or_hospice": np.round(np.arange(-0.30, 0.301, 0.0025), 4)}
specs = [("1 baseline: km instruments (>5, >20 km)", S, ["dd_gt5", "dd_gt20"], {}),
         ("2 + ZIP SES & rurality (ACS, RUCA)", S, ["dd_gt5", "dd_gt20"], {"ses": True})]
if HAVE_T:
    specs += [("3 road-time instruments (>5, >15 min)", S, ["dt_gt5", "dt_gt15"], {"time_access": True}),
              ("4 road-time instruments + SES & rurality", S, ["dt_gt5", "dt_gt15"], {"time_access": True, "ses": True})]
specs += [("5 baseline, 2017-2019 only", S[S.year >= 2017].reset_index(drop=True), ["dd_gt5", "dd_gt20"], {}),
          ("6 baseline, without cardiac arrest", S[S.arrest == 0].reset_index(drop=True), ["dd_gt5", "dd_gt20"], {}),
          ("7 alternative cut-offs (>10, >30 km)", S, ["dd_gt10", "dd_gt30"], {}),
          ("8 continuous instrument log(1+km)", S, ["log1p_dd"], {}),
          ("9 without shock/arrest controls (possibly post-treatment)", S, ["dd_gt5", "dd_gt20"], {"drop": ("arrest", "shock")}),
          ("10 excluding transferred patients with CABG", S[~((S.transfer == 1) & (S.cabg == 1))].reset_index(drop=True), ["dd_gt5", "dd_gt20"], {})]

rows = []
for name, df, z, kw in specs:
    for outc in ["pci_day0", "died", "died_or_hospice"]:
        r = iv(df, outc, "first_nonpci", z, ar_grid=GRID[outc], **kw)
        rows.append({"specification": name, "outcome": outc, **r})
    print(f"  {name}: done", flush=True)
R = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
R.to_csv("results/iv_robustness.csv", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 45)
print("\n" + R.to_string(index=False))

# ---- leave-one-region-out (baseline, same-day PCI)
lo = []
for ph in sorted(S.phr.dropna().unique()):
    d = S[S.phr != ph].reset_index(drop=True)
    r = iv(d, "pci_day0", "first_nonpci", ["dd_gt5", "dd_gt20"])
    lo.append({"dropped PHR": ph, "N": r["N"], "2SLS (pp)": r["2SLS (pp)"], "95% CI": r["95% CI"], "F": r["first-stage F"]})
LO = pd.DataFrame(lo)
LO.to_csv("results/iv_leave_one_region_out.csv", index=False)
print(f"\nLeave-one-region-out (same-day PCI): estimates from {LO['2SLS (pp)'].min()} to {LO['2SLS (pp)'].max()} pp; "
      f"first-stage F from {LO.F.min()} to {LO.F.max()}")

# ---- placebo: instruments should not predict observed baseline risk once geography is controlled
rx = pd.get_dummies(S[["age", "raceeth", "payer_g", "stemi_site"]].astype(str), dtype=float)
rx = pd.concat([rx, S[COMORB + ["female", "arrest", "shock"]].astype(float)], axis=1)
S["pred_death"] = LogisticRegression(max_iter=3000).fit(rx, S.died).predict_proba(rx)[:, 1]
Xg = np.column_stack([S[["dd_gt5", "dd_gt20"]].to_numpy(float),
                      pd.get_dummies(S.year.astype(str), drop_first=True, dtype=float).to_numpy(),
                      pd.get_dummies(S.phr.fillna("NA"), drop_first=True, dtype=float).to_numpy(),
                      S[["log1p_d_any", "log_area"]].to_numpy(float), np.ones(len(S))])
G = pd.Series(S.pat_zip.to_numpy(dtype=object)).astype("category").cat.codes.values
for outc in ["pred_death", "arrest", "shock", "age75"]:
    yv = S[outc].to_numpy(float)
    b = np.linalg.lstsq(Xg, yv, rcond=None)[0]
    V, _ = clus(Xg, yv - Xg @ b, G, np.linalg.inv(Xg.T @ Xg))
    w = float(b[:2] @ np.linalg.solve(V[:2, :2], b[:2]))
    print(f"placebo: {outc:10s} ~ instruments | diff vs dd<=5 km: >5 km {100 * b[0]:+.2f} pp, >20 km {100 * (b[0] + b[1]):+.2f} pp; "
          f"joint p = {stats.chi2.sf(w, 2):.3f}")
