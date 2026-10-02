"""Figures 1-9 of the paper. Run from the working directory (see README). Output: figures/."""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, "scripts")
from stemi_common import load  # noqa: E402

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
W1, W2 = 6.85, 3.35  # full width / half width (inches)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


ci = lambda s: tuple(float(x) for x in re.findall(r"-?\d+\.?\d*", str(s).replace("np.float64", "")))  # noqa: E731

P, H, HS = load()
pcap = {(h, y): c for y, s_ in HS.items() for h, c in zip(s_.THCIC_ID, s_.pci_prev)}
P["admit_pci"] = [pcap.get((h, y), 0) for h, y in zip(P.THCIC_ID, P.year)]
S = P[~((P.transfer == 0) & (P.transfer_out == 1))].dropna(subset=["dd"]).copy()
S["first_nonpci"] = ((S.transfer == 1) | (S.admit_pci == 0)).astype(int)

# ---------------------------------------------------------------- Fig1 cohort flow
fig, ax = plt.subplots(figsize=(W1, 3.6))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
boxes = [((0.5, 0.92), "STEMI admissions, Texas inpatient PUDF 2016–2019\n(principal I21.0–I21.3 POA, emergency/urgent, adult, Texas ZIP5)\nn = 39,969"),
         ((0.5, 0.70), "ZIP code matched to a ZIP code tabulation area\nn = 39,657"),
         ((0.5, 0.48), "Analytic sample (one record per patient)\nn = 38,559"),
         ((0.17, 0.16), "Direct admission,\nPCI centre\nn = 33,238"),
         ((0.5, 0.16), "Direct admission,\nnon-PCI hospital\nn = 601"),
         ((0.83, 0.16), "Arrival by\ninterfacility transfer\nn = 4,720")]
widths = [0.62, 0.42, 0.42, 0.28, 0.28, 0.28]
for ((x, y), t), w in zip(boxes, widths):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.075), w, 0.15, boxstyle="round,pad=0.005", fc="0.95", ec="0.3", lw=0.8))
    ax.text(x, y, t, ha="center", va="center", fontsize=7.5)
for y0, y1 in [(0.845, 0.775), (0.625, 0.555)]:
    ax.annotate("", (0.5, y1), (0.5, y0), arrowprops=dict(arrowstyle="->", lw=0.8))
for x in [0.17, 0.5, 0.83]:
    ax.annotate("", (x, 0.235), (0.5, 0.405), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.73, 0.69, "Excluded: 312 without\nmatching ZCTA", fontsize=7)
ax.text(0.73, 0.45, "Removed: 1,098 records of\npatients later transferred out\n(counted at receiving hospital)", fontsize=7)
ax.text(0.5, 0.01, "Routing failure (non-PCI first hospital) = 601 + 4,720 = 5,321 (13.8%)", ha="center", fontsize=7.5, style="italic")
save(fig, "Fig1")

# ---------------------------------------------------------------- Fig2 map: hospitals and differential distance
s19 = HS[2019]
z = S[S.year == 2019].groupby("pat_zip").agg(lat=("zlat", "first"), lon=("zlon", "first"), dd=("dd", "first"))
fig, axs = plt.subplots(1, 2, figsize=(W1, 3.2))
ax = axs[0]
ax.scatter(z.lon, z.lat, s=1.5, c="0.85", lw=0)
ne = s19[s19.pci_prev == 0]
pc = s19[s19.pci_prev == 1]
ax.scatter(ne.lon, ne.lat, s=7, marker="o", facecolors="none", edgecolors="0.35", lw=0.6, label=f"Hospital without PCI (n={len(ne)})")
ax.scatter(pc.lon, pc.lat, s=9, marker="^", c="k", lw=0, label=f"PCI centre (n={len(pc)})")
ax.set_title("(a) Eligible hospitals, 2019", fontsize=8.5, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
ax = axs[1]
bins = [-0.01, 0.01, 5, 20, 1e4]
cols = ["0.88", "0.65", "0.4", "0.1"]
labs = ["Δ = 0", "0 < Δ ≤ 5 km", "5 < Δ ≤ 20 km", "Δ > 20 km"]
cat = pd.cut(z.dd, bins, labels=False)
for k in range(4):
    m = cat == k
    ax.scatter(z.lon[m], z.lat[m], s=3, c=cols[k], lw=0, label=labs[k])
ax.set_title("(b) Differential distance Δ by patient ZIP code", fontsize=8.5, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="lower left", markerscale=2.5)
for ax in axs:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect(1.15)
fig.tight_layout()
save(fig, "Fig2")

# ---------------------------------------------------------------- Fig5 first stage and reduced form
S["bin"] = pd.cut(S.dd, [-0.01, 0.01, 5, 10, 20, 40, 1e4], labels=["0", "0–5", "5–10", "10–20", "20–40", ">40"])
g = S.groupby("bin", observed=True).agg(fail=("first_nonpci", "mean"), pci0=("pci_day0", "mean"), n=("died", "size"))
fig, ax = plt.subplots(figsize=(W2 + 0.6, 2.6))
x = np.arange(len(g))
ax.bar(x, 100 * g.fail, color="0.7", label="Non-PCI first hospital")
ax.set_ylabel("Non-PCI first hospital, %")
ax.set_xticks(x, g.index)
ax.set_xlabel("Differential distance Δ, km")
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(x, 100 * g.pci0, "k-o", ms=3, label="PCI on day of admission")
ax2.set_ylabel("PCI on day of admission, %")
ax2.set_ylim(55, 80)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, frameon=False)
save(fig, "Fig5")

# ---------------------------------------------------------------- Fig3 ranking: Top-1 with CI, Top-k curves
d = np.load("results/ml_test_scores.npz", allow_pickle=True)
G = pd.Series(d["pat_zip"]).astype("category").cat.codes.values
ng = G.max() + 1
rng = np.random.default_rng(1)
Cw = rng.multinomial(ng, np.full(ng, 1 / ng), size=2000).astype(float)
n_g = np.bincount(G, minlength=ng)
names = [("rule: nearest hospital", "Nearest\nhospital"), ("rule: nearest PCI centre", "Nearest\nPCI centre"),
         ("logit", "Conditional\nlogit"), ("MLP ranker + hospital effects", "Neural\nranker"),
         ("gradient boosting (pointwise)", "Gradient\nboosting")]
key = lambda n: n.replace(" ", "_").replace(":", "").replace(",", "")  # noqa: E731
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
mean, lo, hi = [], [], []
for n, _ in names:
    t1 = (d[f"rank__{key(n)}"] == 1).astype(float)
    est = (Cw @ np.bincount(G, weights=t1, minlength=ng)) / (Cw @ n_g)
    mean.append(t1.mean()); lo.append(np.percentile(est, 2.5)); hi.append(np.percentile(est, 97.5))
mean, lo, hi = map(np.array, (mean, lo, hi))
ax = axs[0]
ax.bar(range(5), mean, yerr=[mean - lo, hi - mean], capsize=3, color=["0.85", "0.85", "0.6", "0.4", "0.2"])
ax.set_xticks(range(5), [l for _, l in names], fontsize=7)
ax.set_ylabel("Top-1 accuracy")
ax.set_ylim(0.3, 0.6)
for i, v in enumerate(mean):
    ax.text(i, hi[i] + 0.006, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_title("(a) Top-1 accuracy, 2019 test year", fontsize=8.5, loc="left")
ax = axs[1]
ks = np.arange(1, 11)
styles = ["k:", "k--", "0.5", "k-.", "k-"]
for (n, l), st in zip(names, styles):
    r = d[f"rank__{key(n)}"]
    ax.plot(ks, [(r <= k).mean() for k in ks], st, lw=1.1, label=l.replace("\n", " "))
ax.set_xlabel("k"); ax.set_ylabel("Top-k accuracy"); ax.set_xticks(ks)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
ax.set_title("(b) Top-k accuracy", fontsize=8.5, loc="left")
fig.tight_layout()
save(fig, "Fig3")

# ---------------------------------------------------------------- Fig6 forest plot of IV specifications
R = pd.read_csv("results/iv_robustness.csv")
lab = {"1 baseline": "Baseline (km, >5 and >20)", "2 + ZIP": "+ ZIP SES and rurality", "3 road-time": "Road-time instruments",
       "4 road-time": "Road time + SES and rurality (primary)", "5 baseline": "2017–2019 only",
       "6 baseline": "Without cardiac arrest", "7 alternative": "Cut-offs >10 and >30 km",
       "8 continuous": "Continuous log(1+Δ)", "9 without": "No shock/arrest controls", "10 excluding": "Excl. transfers with CABG"}
R["label"] = [next(v for k, v in lab.items() if s.startswith(k)) for s in R.specification]
fig, axs = plt.subplots(1, 2, figsize=(W1, 3.0), sharey=True)
for ax, outc, ttl in [(axs[0], "pci_day0", "(a) PCI on day of admission"), (axs[1], "died", "(b) In-hospital death")]:
    r = R[R.outcome == outc].reset_index(drop=True)
    y = np.arange(len(r))[::-1]
    lo_, hi_ = zip(*[ci(c)[:2] for c in r["95% CI"]])
    for yy, est, a, b, l in zip(y, r["2SLS (pp)"], lo_, hi_, r.label):
        bold = "primary" in l
        ax.plot([a, b], [yy, yy], "k-", lw=1.4 if bold else 0.9)
        ax.plot(est, yy, "s" if bold else "o", c="k", ms=4)
    ax.axvline(0, c="0.5", lw=0.6, ls="--")
    ax.set_yticks(y, r.label, fontsize=7)
    ax.set_xlabel("2SLS effect, percentage points")
    ax.set_title(ttl, fontsize=8.5, loc="left")
fig.tight_layout()
save(fig, "Fig6")

# ---------------------------------------------------------------- Fig7 leave-one-region-out
L = pd.read_csv("results/iv_leave_one_region_out.csv", dtype={"dropped PHR": str})
fig, ax = plt.subplots(figsize=(W2 + 0.6, 2.4))
lo_, hi_ = zip(*[ci(c)[:2] for c in L["95% CI"]])
x = np.arange(len(L))
ax.errorbar(x, L["2SLS (pp)"], yerr=[L["2SLS (pp)"] - np.array(lo_), np.array(hi_) - L["2SLS (pp)"]], fmt="o", c="k", ms=3, capsize=2, lw=0.8)
ax.axhline(-23.37, c="0.5", lw=0.7, ls="--", label="All regions")
ax.set_xticks(x, [p.lstrip("0") for p in L["dropped PHR"]])
ax.set_xlabel("Public health region left out")
ax.set_ylabel("Effect on same-day PCI, pp")
ax.legend(fontsize=7, frameon=False)
save(fig, "Fig7")

# ---------------------------------------------------------------- Fig8 detour trade-off
tau_b, tau_p, se_p = 0.2337, 0.1655, 0.0438
ny = S.year.nunique()
Xs = np.arange(1, 61)
fail, mdet = [], []
for X in Xs:
    gg = S[(S.dd > 0.01) & (S.dd <= X)]
    fail.append(gg.first_nonpci.sum() / ny); mdet.append(gg.dd.mean())
fail, mdet = np.array(fail), np.array(mdet)
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.5))
ax = axs[0]
ax.fill_between(Xs, (tau_p - 1.96 * se_p) * fail, (tau_p + 1.96 * se_p) * fail, color="0.85", lw=0, label="95% CI (primary)")
ax.plot(Xs, tau_p * fail, "k-", lw=1.2, label="Primary estimate (16.6 pp)")
ax.plot(Xs, tau_b * fail, "k--", lw=1, label="Baseline estimate (23.4 pp)")
ax.set_xlabel("Detour limit X, km"); ax.set_ylabel("Additional same-day PCI per year")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.set_title("(a) Expected gain", fontsize=8.5, loc="left")
ax = axs[1]
ax.plot(mdet, tau_p * fail, "k-", lw=1.2)
for X in [5, 10, 20, 30, 50]:
    i = X - 1
    ax.plot(mdet[i], tau_p * fail[i], "ko", ms=3)
    ax.annotate(f"X={X}", (mdet[i], tau_p * fail[i]), textcoords="offset points", xytext=(4, -9), fontsize=6.5)
ax.set_xlabel("Mean detour of covered patients, km"); ax.set_ylabel("Additional same-day PCI per year")
ax.set_title("(b) Gain versus detour", fontsize=8.5, loc="left")
fig.tight_layout()
save(fig, "Fig8")

# ---------------------------------------------------------------- Fig4 subgroups
U = pd.read_csv("results/ml_subgroups_test2019.csv", index_col=0)
U = U[~U.index.str.startswith("female") & ~U.index.str.contains("=Other")]
pretty = {"raceeth=Asian/PI": "Asian/PI", "raceeth=Black": "Black", "raceeth=Hispanic": "Hispanic", "raceeth=White": "White",
          "payer_g=Commercial": "Commercial", "payer_g=Medicaid": "Medicaid", "payer_g=Medicare": "Medicare",
          "payer_g=Self-pay": "Self-pay", "rural=False": "Urban ZIP", "rural=True": "Rural ZIP", "age75=0": "Age <75",
          "age75=1": "Age ≥75"}
U = U[U.index.isin(pretty)]
fig, ax = plt.subplots(figsize=(W1, 2.4))
x = np.arange(len(U))
ax.bar(x - 0.2, U["Top-1 logit"], 0.4, color="0.65", label="Conditional logit")
ax.bar(x + 0.2, U["Top-1 MLP ranker + hospital effects"], 0.4, color="0.25", label="Neural ranker")
ax.set_xticks(x, [pretty[i] for i in U.index], fontsize=7, rotation=30, ha="right")
ax.set_ylabel("Top-1 accuracy"); ax.set_ylim(0.35, 0.62)
ax.legend(fontsize=7, frameon=False, ncol=2)
save(fig, "Fig4")

# ---------------------------------------------------------------- Fig9 road time vs straight line
T = pd.read_csv("ext/osrm_zip_hosp.csv", dtype={"pat_zip": str, "THCIC_ID": str}).dropna()
Z = pd.read_csv("zcta2010_tx_internal_points.csv", dtype={"zcta5": str}).set_index("zcta5")
Hh = H.set_index("THCIC_ID")
T = T[T.pat_zip.isin(Z.index) & T.THCIC_ID.isin(Hh.index)].sample(20000, random_state=0)
from stemi_common import hav  # noqa: E402
km = hav(Z.loc[T.pat_zip, "lat"].values, Z.loc[T.pat_zip, "lon"].values, Hh.loc[T.THCIC_ID, "lat"].values, Hh.loc[T.THCIC_ID, "lon"].values)
fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.6))
ax.scatter(km, T.minutes, s=1, c="k", alpha=0.15, lw=0)
ax.set_xlabel("Straight-line distance, km"); ax.set_ylabel("Road travel time, min")
r = np.corrcoef(km, T.minutes)[0, 1]
ax.text(0.03, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=7.5)
save(fig, "Fig9")
print("figures written to", os.path.abspath(OUT))
