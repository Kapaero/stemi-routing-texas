"""Composed decision rule on 2019 routing failures (run from 'derived'):
if 0 < detour <= X: recommend the ranker's top PCI centre among those reachable within the detour limit
(distance <= distance to nearest hospital + X); report detours versus the nearest PCI centre."""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from stemi_common import K, hav, load, stack  # noqa: E402

P, H, HS = load()
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
P["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(P.THCIC_ID, P.year)]
S = P[~((P.transfer == 0) & (P.transfer_out == 1))].dropna(subset=["dd"]).copy()
S["first_nonpci"] = ((S.transfer == 1) | (S.admit_pci == 0)).astype(int)
RC = P.log_area.median()
tr = stack(P[P.transfer == 0], [2017, 2018], HS, RC)
F19 = S[(S.year == 2019) & (S.first_nonpci == 1)].copy()
s19 = HS[2019]
d_ = hav(F19.zlat.values[:, None], F19.zlon.values[:, None], s19.lat.values[None], s19.lon.values[None])
F19["THCIC_ID"] = s19.THCIC_ID.to_numpy(dtype=object)[d_.argmin(1)]  # placeholder: keep every choice set
bf = stack(F19, [2019], HS, RC)
phr = sorted(set(tr["phr"].columns) | set(bf["phr"].columns))


def flat(s):
    Q = np.hstack([s["pat"], s["phr"].reindex(columns=phr, fill_value=0.0).to_numpy(float)])
    n, k, f = s["X_alt"].shape
    return np.hstack([s["X_alt"].reshape(n * k, f), np.repeat(Q, k, axis=0)])


ytr = np.zeros((len(tr["y"]), K)); ytr[np.arange(len(tr["y"])), tr["y"]] = 1
Xtr, Xbf = flat(tr), flat(bf)
u = np.mean([np.log(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                                    random_state=s).fit(Xtr, ytr.ravel()).predict_proba(Xbf)[:, 1] + 1e-12)
             for s in range(5)], axis=0).reshape(-1, K)
d, pci = bf["d"], bf["pci"]
dmin = d[:, 0]
dd = F19.loc[bf["rows"], "dd"].values
dnp = np.where(pci == 1, d, np.inf).min(1)
rows = []
for X in [10, 20, 50]:
    m = (dd > 0.01) & (dd <= X)
    ok = (pci == 1) & (d <= dmin[:, None] + X)
    ch = np.where(ok, u, -np.inf).argmax(1)
    i = np.where(m)[0]
    rows.append({"limit km": X, "2019 failures with 0<detour<=X": int(m.sum()),
                 "ranker picks nearest PCI centre %": round(100 * (d[i, ch[i]] == dnp[i]).mean(), 1),
                 "mean detour to nearest PCI km": round((dnp[i] - dmin[i]).mean(), 1),
                 "mean detour to ranker's PCI km": round((d[i, ch[i]] - dmin[i]).mean(), 1)})
R = pd.DataFrame(rows)
R.to_csv("results/composed_rule_2019.csv", index=False)
print(R.to_string(index=False))
