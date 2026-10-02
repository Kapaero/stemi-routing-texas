"""Policy comparison on 2019 routing failures whose nearest hospital is not a PCI centre (Delta > 0).
For each policy: share of these patients whose recommended first hospital is a PCI centre, expected additional
same-day PCIs (share x n x tau, primary and baseline), and mean detour beyond the nearest hospital. Run from 'derived'."""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from stemi_common import K, hav, load, stack  # noqa: E402

TAU_P, TAU_B = 0.1655, 0.2337
P, H, HS = load()
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
P["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(P.THCIC_ID, P.year)]
S = P[~((P.transfer == 0) & (P.transfer_out == 1))].dropna(subset=["dd"]).copy()
S["first_nonpci"] = ((S.transfer == 1) | (S.admit_pci == 0)).astype(int)
RC = P.log_area.median()
tr = stack(P[P.transfer == 0], [2017, 2018], HS, RC)

F = S[(S.year == 2019) & (S.first_nonpci == 1) & (S.dd > 0.01)].copy()
s19 = HS[2019]
d_ = hav(F.zlat.values[:, None], F.zlon.values[:, None], s19.lat.values[None], s19.lon.values[None])
F["THCIC_ID"] = s19.THCIC_ID.to_numpy(dtype=object)[d_.argmin(1)]  # placeholder: keep every choice set
bf = stack(F, [2019], HS, RC)
dd = F.loc[bf["rows"], "dd"].values
d, pci = bf["d"], bf["pci"]
n = len(d)
i = np.arange(n)

# conditional logit (same specification as in the paper)
Xt = tr["X_logit"]
mu, sd = Xt.reshape(-1, Xt.shape[2]).mean(0), Xt.reshape(-1, Xt.shape[2]).std(0) + 1e-6


def nll(b, X, y):
    u = X @ b; u = u - u.max(1, keepdims=True); p = np.exp(u); p /= p.sum(1, keepdims=True)
    j = np.arange(len(y))
    return -np.log(p[j, y] + 1e-300).sum(), -(X[j, y] - (p[:, :, None] * X).sum(1)).sum(0)


b = minimize(nll, np.zeros(Xt.shape[2]), args=((Xt - mu) / sd, tr["y"]), jac=True, method="L-BFGS-B").x
u_logit = ((bf["X_logit"] - mu) / sd) @ b

# gradient boosting (5 seeds)
phr = sorted(set(tr["phr"].columns) | set(bf["phr"].columns))


def flat(s):
    Q = np.hstack([s["pat"], s["phr"].reindex(columns=phr, fill_value=0.0).to_numpy(float)])
    nn, k, f = s["X_alt"].shape
    return np.hstack([s["X_alt"].reshape(nn * k, f), np.repeat(Q, k, axis=0)])


ytr = np.zeros((len(tr["y"]), K)); ytr[np.arange(len(tr["y"])), tr["y"]] = 1
Xtr, Xbf = flat(tr), flat(bf)
u_gbm = np.mean([np.log(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                                        l2_regularization=1.0, early_stopping=True,
                                                        validation_fraction=0.15, random_state=s)
                        .fit(Xtr, ytr.ravel()).predict_proba(Xbf)[:, 1] + 1e-12) for s in range(5)], axis=0).reshape(-1, K)

nearest_pci = np.where(pci == 1, d, np.inf).argmin(1)
choices = {
    "Nearest hospital": np.zeros(n, dtype=int),
    "Nearest PCI centre": nearest_pci,
    "Conditional logit": u_logit.argmax(1),
    "Gradient boosting": u_gbm.argmax(1),
}
for X in (10, 20):
    ok = (pci == 1) & (d <= d[:, :1] + X)
    c = np.where(ok, u_gbm, -np.inf).argmax(1)
    c = np.where(dd <= X, c, 0)
    choices[f"Detour rule, {X} km (+ ranker)"] = c
rows = []
for name, c in choices.items():
    to_pci = pci[i, c] == 1
    rows.append({"policy": name, "PCI first, %": round(100 * to_pci.mean(), 1),
                 "redirected to PCI, n": int(to_pci.sum()),
                 "additional same-day PCI (primary-baseline)": f"{TAU_P * to_pci.sum():.0f}-{TAU_B * to_pci.sum():.0f}",
                 "mean detour, km (all)": round((d[i, c] - d[:, 0]).mean(), 1),
                 "mean detour, km (redirected)": round((d[i, c] - d[:, 0])[to_pci].mean(), 1) if to_pci.any() else 0.0})
R = pd.DataFrame(rows)
R.to_csv("results/policy_table_2019.csv", index=False)
print(f"2019 routing failures with Delta > 0: n = {n}")
print(R.to_string(index=False))
