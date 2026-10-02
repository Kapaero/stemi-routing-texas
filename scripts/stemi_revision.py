"""Revision analyses (run from 'derived'):
1) cohort flow and Table 1 on the analytic sample (direct to PCI centre / direct to non-PCI hospital / via transfer);
2) composition of routing failures by differential distance; detour rule excluding the <=5 km band;
3) ranking: out-of-choice-set patients counted as misses; gradient boosting with and without discharge-coded
   proxies (POA arrest, shock, STEMI site); PCI-only constraint on the same ensemble;
4) the ranker applied to 2019 patients whose first hospital was not PCI-capable."""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from stemi_common import K, load, rank_metrics, stack, hav  # noqa: E402

out = open("results/revision_summary.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")


P, H, HS = load()
n_raw = len(pd.read_csv("stemi_patients_2016_2019.tsv", sep="\t", usecols=["year"]))
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
P["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(P.THCIC_ID, P.year)]
first_legs = (P.transfer == 0) & (P.transfer_out == 1)
S = P[~first_legs].dropna(subset=["dd"]).copy()
S["group"] = np.where(S.transfer == 1, "via transfer", np.where(S.admit_pci == 1, "direct, PCI centre", "direct, non-PCI hospital"))
S["first_nonpci"] = (S.group != "direct, PCI centre").astype(int)
say(f"FLOW: extracted {n_raw} -> with ZCTA {len(P)} -> first legs of later transfers removed {int(first_legs.sum())} "
    f"-> analytic {len(S)}: " + ", ".join(f"{k} {v}" for k, v in S.group.value_counts().items()))
say(f"routing failures: {int(S.first_nonpci.sum())} ({100 * S.first_nonpci.mean():.1f}%)")

def desc(g):
    return pd.Series({"Patients, n": len(g), "Age >=75, %": 100 * g.age75.mean(), "Female, %": 100 * g.female.mean(),
                      "Hispanic, %": 100 * (g.raceeth == "Hispanic").mean(), "Medicare, %": 100 * (g.payer_g == "Medicare").mean(),
                      "Cardiac arrest (POA), %": 100 * g.arrest.mean(), "Cardiogenic shock (POA), %": 100 * g.shock.mean(),
                      "Anterior STEMI, %": 100 * g.site_anterior.mean(), "km to nearest hospital (median)": g.d_any.median(),
                      "km to nearest PCI centre (median)": g.d_pci.median(), "PCI on day of admission, %": 100 * g.pci_day0.mean(),
                      "In-hospital death, %": 100 * g.died.mean()})
T1 = pd.concat({k: desc(S[S.group == k]) for k in ["direct, PCI centre", "direct, non-PCI hospital", "via transfer"]}, axis=1).round(1)
T1.to_csv("results/table1_analytic.csv")
say("\nTABLE 1 (analytic sample)\n" + T1.to_string())

# failures by detour band
S["band"] = pd.cut(S.dd, [-0.01, 0.01, 5, 10, 20, 1e4], labels=["0", "0-5", "5-10", "10-20", ">20"])
fb = S[S.first_nonpci == 1].groupby("band", observed=True).agg(n=("died", "size"), via_transfer=("transfer", "mean"),
                                                                src6=("src", lambda x: (x == "6").mean()))
say("\nFAILURES by differential distance band (share via transfer; share from 'other facility' src=6):\n" + fb.round(3).to_string())
tau, tau_c = 0.2337, 0.1655
ny = S.year.nunique()
rows = []
for lo, hi in [(0.01, 5), (0.01, 10), (5, 10), (0.01, 20), (5, 20), (0.01, 50), (5, 50)]:
    g = S[(S.dd > lo) & (S.dd <= hi)]
    f = g.first_nonpci.sum() / ny
    rows.append({"band km": f"{0 if lo < 1 else lo}-{hi}", "failures/yr": round(f), "gain/yr (tau=23.4)": round(tau * f, 1),
                 "gain/yr (tau=16.6)": round(tau_c * f, 1), "mean detour": round(g.dd.mean(), 1)})
say("\nDETOUR RULE\n" + pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- ranking
RC = P.log_area.median()
D = P[P.transfer == 0]
tr, te = stack(D, [2017, 2018], HS, RC), stack(D, [2019], HS, RC)
n_te_all = te["n_all"]
phr_cols = sorted(set(tr["phr"].columns) | set(te["phr"].columns))
PAT_COLS = ["age_ord", "female", "arrest", "shock", "site_anterior", "site_inferior", "weekend", "log_area", "log1p_d_any", "log1p_dd"]
POA_IDX = [PAT_COLS.index(c) for c in ["arrest", "shock", "site_anterior", "site_inferior"]]


def pat(s, drop_poa):
    m = s["pat"].copy()
    if drop_poa:
        m = np.delete(m, POA_IDX, axis=1)
    return np.hstack([m, s["phr"].reindex(columns=phr_cols, fill_value=0.0).to_numpy(float)])


def flat(Xa, Q):
    n, k, f = Xa.shape
    return np.hstack([Xa.reshape(n * k, f), np.repeat(Q, k, axis=0)])


def gbm_fit(drop_poa):
    ytr = np.zeros((len(tr["y"]), K)); ytr[np.arange(len(tr["y"])), tr["y"]] = 1
    Xtr = flat(tr["X_alt"], pat(tr, drop_poa))
    models = []
    for s in range(5):
        g = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.15, random_state=s)
        models.append(g.fit(Xtr, ytr.ravel()))
    return models


def gbm_score(models, s, drop_poa):
    X = flat(s["X_alt"], pat(s, drop_poa))
    return np.mean([np.log(m.predict_proba(X)[:, 1] + 1e-12).reshape(-1, K) for m in models], axis=0)


def nll(beta, X, y):
    u = X @ beta; u = u - u.max(1, keepdims=True); p = np.exp(u); p /= p.sum(1, keepdims=True)
    i = np.arange(len(y))
    return -np.log(p[i, y] + 1e-300).sum(), -(X[i, y] - (p[:, :, None] * X).sum(1)).sum(0)


def logit_score(drop_poa):
    keep = [i for i in range(tr["X_logit"].shape[2]) if not (drop_poa and i in (8, 9))]
    Xt, Xe = tr["X_logit"][:, :, keep], te["X_logit"][:, :, keep]
    mu, sd = Xt.reshape(-1, Xt.shape[2]).mean(0), Xt.reshape(-1, Xt.shape[2]).std(0) + 1e-6
    b = minimize(nll, np.zeros(len(keep)), args=((Xt - mu) / sd, tr["y"]), jac=True, method="L-BFGS-B").x
    return ((Xe - mu) / sd) @ b


def row(name, u):
    m = rank_metrics(u, te["y"])
    i = np.arange(len(u))
    m["Top-1 all patients"] = m["Top-1"] * len(u) / n_te_all
    m["Top-5 all patients"] = m["Top-5"] * len(u) / n_te_all
    m["top-1 PCI %"] = 100 * te["pci"][i, u.argmax(1)].mean()
    return name, m


R = {}
G_full = gbm_fit(False)
u_full = gbm_score(G_full, te, False)
for name, u in [("nearest hospital", -te["d"]), ("nearest PCI centre", -te["d"] + 1e4 * te["pci"]),
                ("logit", logit_score(False)), ("logit, no discharge-coded proxies", logit_score(True)),
                ("gradient boosting", u_full), ("gradient boosting, PCI centres only", u_full + 1e4 * te["pci"])]:
    k, m = row(name, u); R[k] = m
G_np = gbm_fit(True)
k, m = row("gradient boosting, no discharge-coded proxies", gbm_score(G_np, te, True)); R[k] = m
RT = pd.DataFrame(R).T.round(3)
RT.to_csv("results/ranking_revision.csv")
say(f"\nRANKING (2019 test: {len(te['y'])} in choice set of {n_te_all} direct admissions)\n" + RT.to_string())

# ---------------------------------------------------------------- ranker applied to routing failures (2019)
F19 = S[(S.year == 2019) & (S.first_nonpci == 1)].copy()
s19 = HS[2019]
dd_ = hav(F19.zlat.values[:, None], F19.zlon.values[:, None], s19.lat.values[None], s19.lon.values[None])
F19["THCIC_ID"] = s19.THCIC_ID.to_numpy(dtype=object)[dd_.argmin(1)]  # placeholder so that every choice set is kept
bf = stack(F19, [2019], HS, RC)
u = gbm_score(G_full, bf, False)
i = np.arange(len(u))
t1, t1c = u.argmax(1), (u + 1e4 * bf["pci"]).argmax(1)
dmin = bf["d"][:, 0]
dpci = np.where(bf["pci"] == 1, bf["d"], np.inf).min(1)
say(f"\nRANKER ON 2019 ROUTING FAILURES (n={len(u)}): unconstrained top-1 is a PCI centre {100 * bf['pci'][i, t1].mean():.1f}%; "
    f"constrained recommendation = nearest PCI centre {100 * (bf['d'][i, t1c] == dpci).mean():.1f}%")
say(f"  median km: nearest hospital {np.median(dmin):.1f}, nearest PCI centre {np.median(dpci):.1f}, "
    f"recommended PCI centre {np.median(bf['d'][i, t1c]):.1f}; mean extra km vs nearest hospital {np.mean(bf['d'][i, t1c] - dmin):.1f}")
say(f"  if routed to the recommended PCI centre: +{tau * len(u):.0f} same-day PCI in 2019 (tau=23.4), +{tau_c * len(u):.0f} (tau=16.6)")
out.close()
