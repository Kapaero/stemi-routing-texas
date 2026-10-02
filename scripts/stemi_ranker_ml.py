"""ML rankers vs conditional logit for the first-destination hospital of STEMI patients (direct arrivals).
Train 2017-2018, test 2019; choice set = 30 nearest eligible hospitals; features known at dispatch time only
(race/ethnicity and payer are excluded from the routing models and used only for the subgroup check).
Run from the 'derived' folder; results go to derived/results/."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from stemi_common import K, load, rank_metrics, stack  # noqa: E402

torch.set_num_threads(8)
os.makedirs("results", exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
t0 = time.time()

P, H, HS = load()
RURAL_CUT = P.log_area.median()
D = P[P.transfer == 0]
tr, te = stack(D, [2017, 2018], HS, RURAL_CUT), stack(D, [2019], HS, RURAL_CUT)
phr_cols = sorted(set(tr["phr"].columns) | set(te["phr"].columns))


def pat_matrix(s):
    return np.hstack([s["pat"], s["phr"].reindex(columns=phr_cols, fill_value=0.0).to_numpy(float)])


Ptr, Pte = pat_matrix(tr), pat_matrix(te)


def zscore(a_tr, a_te):
    flat = a_tr.reshape(-1, a_tr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return (a_tr - mu) / sd, (a_te - mu) / sd


A_tr, A_te = zscore(tr["X_alt"], te["X_alt"])
L_tr, L_te = zscore(tr["X_logit"], te["X_logit"])
Q_tr, Q_te = zscore(Ptr, Pte)
hosp = {h: i + 1 for i, h in enumerate(sorted(set(tr["ids"].ravel())))}  # 0 = hospital unseen in training
I_tr = np.vectorize(lambda h: hosp.get(h, 0))(tr["ids"])
I_te = np.vectorize(lambda h: hosp.get(h, 0))(te["ids"])
print(f"train {len(tr['y'])} / test {len(te['y'])} patients, {K} alternatives each | "
      f"hospitals seen in training {len(hosp)} | test alternatives unseen in training {100 * (I_te == 0).mean():.1f}%")


class Ranker(nn.Module):
    def __init__(self, n_alt, n_pat, n_hosp, mlp, hosp_effects, hidden=64):
        super().__init__()
        self.mlp = mlp
        self.net = (nn.Sequential(nn.Linear(n_alt + n_pat, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
                    if mlp else nn.Linear(n_alt, 1, bias=False))
        self.emb = nn.Embedding(n_hosp + 1, 1, padding_idx=0) if hosp_effects else None
        if self.emb is not None:
            nn.init.zeros_(self.emb.weight)

    def forward(self, A, Q, I):
        x = torch.cat([A, Q[:, None, :].expand(-1, A.shape[1], -1)], dim=2) if self.mlp else A
        u = self.net(x).squeeze(-1)
        return u + self.emb(I).squeeze(-1) if self.emb is not None else u


def train(mlp, hosp_effects, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(tr["y"])
    idx = rng.permutation(n)
    va, trn = idx[: int(0.15 * n)], idx[int(0.15 * n):]
    Atr, Ate = (A_tr, A_te) if mlp else (L_tr, L_te)
    model = Ranker(Atr.shape[2], Q_tr.shape[1], len(hosp), mlp, hosp_effects)
    groups = [{"params": model.net.parameters(), "weight_decay": 1e-5}]
    if hosp_effects:
        groups.append({"params": model.emb.parameters(), "weight_decay": 1e-3})
    opt = torch.optim.Adam(groups, lr=1e-3 if mlp else 1e-2)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    At, Qt, It, yt = T(Atr), T(Q_tr), torch.tensor(I_tr), torch.tensor(tr["y"])
    best, state, bad = np.inf, None, 0
    for ep in range(300):
        model.train()
        perm = torch.tensor(rng.permutation(trn))
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            loss = nn.functional.cross_entropy(model(At[j], Qt[j], It[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(model(At[va], Qt[va], It[va]), yt[va]).item()
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        u = model(T(Ate), T(Q_te), torch.tensor(I_te)).numpy()
    return u, best, ep + 1


def test_nll(u):
    u = u - u.max(1, keepdims=True)
    return float(-(u[np.arange(len(u)), te["y"]] - np.log(np.exp(u).sum(1))).mean())


def summarize(name, utils, extra=None):
    ms = []
    for u in utils:
        m = rank_metrics(u, te["y"])
        m["test NLL"] = np.nan if "only" in name else test_nll(u)
        m["top-1 is PCI centre %"] = 100 * te["pci"][np.arange(len(u)), u.argmax(1)].mean()
        ms.append(m)
    df = pd.DataFrame(ms)
    row = {c: f"{df[c].mean():.3f} ± {df[c].std(ddof=0):.3f}" if len(df) > 1 else f"{df[c].mean():.3f}" for c in df}
    row.update(extra or {})
    return name, row


rows = {}
for name, score in [("rule: nearest hospital", -te["d"]), ("rule: nearest PCI centre", -te["d"] + 1e4 * te["pci"])]:
    m = rank_metrics(score, te["y"])
    m["top-1 is PCI centre %"] = 100 * te["pci"][np.arange(len(score)), score.argmax(1)].mean()
    rows[name] = {k: f"{v:.3f}" for k, v in m.items()}

U = {}
for name, mlp, he in [("logit", False, False), ("logit + hospital effects", False, True),
                      ("MLP ranker", True, False), ("MLP ranker + hospital effects", True, True)]:
    seeds = [0] if not mlp else SEEDS
    outs = [train(mlp, he, s) for s in seeds]
    U[name] = [o[0] for o in outs]
    k, r = summarize(name, U[name], {"epochs": "/".join(str(o[2]) for o in outs)})
    rows[k] = r
    print(f"  {name}: done ({time.time() - t0:.0f}s)")


def flat(Xa, Q):
    n, k, f = Xa.shape
    return np.hstack([Xa.reshape(n * k, f), np.repeat(Q, k, axis=0)])


ytr_flat = np.zeros((len(tr["y"]), K))
ytr_flat[np.arange(len(tr["y"])), tr["y"]] = 1
Xf_tr, Xf_te = flat(tr["X_alt"], Ptr), flat(te["X_alt"], Pte)
gb = []
for s in SEEDS:
    g = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15, random_state=s)
    g.fit(Xf_tr, ytr_flat.ravel())
    p = g.predict_proba(Xf_te)[:, 1].reshape(-1, K)
    gb.append(np.log(p + 1e-12))
U["gradient boosting (pointwise)"] = gb
k, r = summarize("gradient boosting (pointwise)", gb)
rows[k] = r
print(f"  gradient boosting: done ({time.time() - t0:.0f}s)")

# road travel time instead of straight-line km (same choice sets, re-ranked by minutes)
YT = {}
if os.path.exists("ext/osrm_zip_hosp.csv"):
    from scipy.optimize import minimize
    TT = pd.read_csv("ext/osrm_zip_hosp.csv", dtype={"pat_zip": str, "THCIC_ID": str}).dropna(subset=["minutes"])
    tmap = dict(zip(zip(TT.pat_zip, TT.THCIC_ID), TT.minutes))
    trT, teT = stack(D, [2017, 2018], HS, RURAL_CUT, tmap), stack(D, [2019], HS, RURAL_CUT, tmap)
    assert (trT["rows"] == tr["rows"]).all() and (teT["rows"] == te["rows"]).all()
    LtT, LeT = zscore(trT["X_logit"], teT["X_logit"])

    def nll(beta, X, y):
        u = X @ beta
        u = u - u.max(1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(1, keepdims=True)
        i = np.arange(len(y))
        return -np.log(p[i, y] + 1e-300).sum(), -(X[i, y] - (p[:, :, None] * X).sum(1)).sum(0)

    bT = minimize(nll, np.zeros(LtT.shape[2]), args=(LtT, trT["y"]), jac=True, method="L-BFGS-B").x
    U["logit, road time"] = [LeT @ bT]
    YT["logit, road time"] = teT["y"]
    gbT = []
    XfT_tr, XfT_te = flat(trT["X_alt"], Ptr), flat(teT["X_alt"], Pte)
    ytT = np.zeros((len(trT["y"]), K))
    ytT[np.arange(len(trT["y"])), trT["y"]] = 1
    for s in SEEDS:
        g = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.15, random_state=s)
        g.fit(XfT_tr, ytT.ravel())
        gbT.append(np.log(g.predict_proba(XfT_te)[:, 1].reshape(-1, K) + 1e-12))
    U["gradient boosting, road time"] = gbT
    YT["gradient boosting, road time"] = teT["y"]
    for name in ["logit, road time", "gradient boosting, road time"]:
        y_save, pci_save = te["y"], te["pci"]
        te["y"], te["pci"] = teT["y"], teT["pci"]
        k, r = summarize(name, U[name])
        rows[k] = r
        te["y"], te["pci"] = y_save, pci_save
    print(f"  road-time variants: done ({time.time() - t0:.0f}s)")

# guideline constraint: recommend PCI centres only (non-PCI alternatives pushed to the bottom)
best_name = "MLP ranker + hospital effects"
k, r = summarize(best_name + ", PCI centres only", [u + 1e4 * te["pci"] for u in U[best_name]])
rows[k] = r

R = pd.DataFrame(rows).T
R.to_csv("results/ml_ranker_test2019.csv")
pd.set_option("display.width", 250)
print("\nTest 2019, direct arrivals (mean ± sd over 5 seeds where applicable):\n" + R.to_string())

# subgroup check on the 2019 test set (ensemble of seeds for the ML model)
te_df = D.loc[te["rows"]].reset_index(drop=True)
te_df["rural"] = te_df.log_area > RURAL_CUT
sub = []
for name in ["logit", best_name]:
    u = np.mean(U[name], axis=0)
    order = np.argsort(-u, axis=1, kind="stable")
    rank = (order == te["y"][:, None]).argmax(1) + 1
    te_df[f"top1_{name}"] = rank == 1
    te_df[f"top5_{name}"] = rank <= 5
for col in ["raceeth", "payer_g", "rural", "age75", "female"]:
    g = te_df.groupby(col)
    t = pd.DataFrame({"N": g.size()})
    for name in ["logit", best_name]:
        t[f"Top-1 {name}"] = g[f"top1_{name}"].mean().round(3)
        t[f"Top-5 {name}"] = g[f"top5_{name}"].mean().round(3)
    t.index = [f"{col}={v}" for v in t.index]
    sub.append(t)
SUB = pd.concat(sub)
SUB.to_csv("results/ml_subgroups_test2019.csv")
print("\nSubgroups (2019 test):\n" + SUB.to_string())

# per-patient rank and log-probability of the chosen hospital (seed ensemble) for paired significance tests
save = {"pat_zip": te_df.pat_zip.to_numpy(dtype=str)}
rules = {"rule: nearest hospital": -te["d"], "rule: nearest PCI centre": -te["d"] + 1e4 * te["pci"]}
for name, u in list(rules.items()) + [(n, np.mean(v, axis=0)) for n, v in U.items()]:
    y_ = YT.get(name, te["y"])
    order = np.argsort(-u, axis=1, kind="stable")
    key = name.replace(" ", "_").replace(":", "").replace(",", "")
    save[f"rank__{key}"] = (order == y_[:, None]).argmax(1) + 1
    if not name.startswith("rule"):
        z = u - u.max(1, keepdims=True)
        save[f"logp__{key}"] = z[np.arange(len(z)), y_] - np.log(np.exp(z).sum(1))
np.savez("results/ml_test_scores.npz", **save)
print(f"\ntotal time {time.time() - t0:.0f}s")
