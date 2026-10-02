"""Paired significance tests for the hospital-ranking models on the 2019 test set.
Cluster bootstrap over patient ZIPs (2,000 replicates) for differences in Top-1, Top-5, MRR and NLL;
exact McNemar test for Top-1 (patient level). Run from 'derived' after stemi_ranker_ml.py."""
import numpy as np
import pandas as pd
from scipy import stats

B = 2000
rng = np.random.default_rng(2019)
d = np.load("results/ml_test_scores.npz", allow_pickle=True)
G = pd.Series(d["pat_zip"]).astype("category").cat.codes.values
ng = G.max() + 1
key = lambda n: n.replace(" ", "_").replace(":", "").replace(",", "")  # noqa: E731


def metrics(name):
    r = d[f"rank__{key(name)}"]
    m = {"Top-1": (r == 1).astype(float), "Top-5": (r <= 5).astype(float), "MRR": 1.0 / r}
    lp = f"logp__{key(name)}"
    if lp in d.files:
        m["NLL"] = -d[lp]
    return m


C = rng.multinomial(ng, np.full(ng, 1 / ng), size=B).astype(float)  # cluster resampling weights (B x clusters)
n_g = np.bincount(G, minlength=ng).astype(float)
PAIRS = [("gradient boosting (pointwise)", "logit"),
         ("MLP ranker + hospital effects", "logit"),
         ("MLP ranker", "logit"),
         ("gradient boosting (pointwise)", "MLP ranker + hospital effects"),
         ("logit", "rule: nearest PCI centre"),
         ("gradient boosting (pointwise)", "rule: nearest PCI centre"),
         ("logit + hospital effects", "logit"),
         ("logit, road time", "logit"),
         ("gradient boosting, road time", "gradient boosting (pointwise)")]
rows = []
for a, b in PAIRS:
    if f"rank__{key(a)}" not in d.files or f"rank__{key(b)}" not in d.files:
        continue
    ma, mb = metrics(a), metrics(b)
    for met in ["Top-1", "Top-5", "MRR", "NLL"]:
        if met not in ma or met not in mb:
            continue
        diff = ma[met] - mb[met]
        s_g = np.bincount(G, weights=diff, minlength=ng)
        est = (C @ s_g) / (C @ n_g)
        p = max(2 * min((est <= 0).mean(), (est >= 0).mean()), 1 / B)
        row = {"model A": a, "model B": b, "metric": met, "A": ma[met].mean(), "B": mb[met].mean(),
               "A - B": diff.mean(), "95% CI (cluster bootstrap)": tuple(np.round(np.percentile(est, [2.5, 97.5]), 4)),
               "p (bootstrap)": p}
        if met == "Top-1":
            ac, bc = ma[met].astype(bool), mb[met].astype(bool)
            n01, n10 = int((~ac & bc).sum()), int((ac & ~bc).sum())
            row["McNemar p"] = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            row["discordant A+/B- vs A-/B+"] = f"{n10}/{n01}"
        rows.append(row)
R = pd.DataFrame(rows)
for c in ["A", "B", "A - B"]:
    R[c] = R[c].round(4)
R["p (bootstrap)"] = R["p (bootstrap)"].map(lambda x: f"{x:.4f}" if x >= 1e-3 else "<0.001")
if "McNemar p" in R:
    R["McNemar p"] = R["McNemar p"].map(lambda x: "" if pd.isna(x) else (f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"))
R.to_csv("results/ml_significance.csv", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 40)
print(f"{len(G)} test patients in {ng} ZIP clusters, {B} bootstrap replicates (NLL: lower is better)\n")
print(R.fillna("").to_string(index=False))
