"""Build hospitals_tx_2016_2019.csv, the hospital input of the analysis scripts.

Combines the public geocoding table (hospital_geocodes.csv) with per-hospital aggregates from the PUDF
(hosp_year_2016_2019.tsv, produced by tx_hosp_year.awk). Run from the working directory.
  active_<y>       >= 50 discharges in year y
  pci_ami_<y>      PCIs performed in admissions with a principal diagnosis of acute myocardial infarction
  pci_capable_<y>  pci_ami_<y> >= 10
  has_ed           >= 100 admissions flagged as coming through the emergency department in some year 2017-2019
"""
import pandas as pd

G = pd.read_csv("hospital_geocodes.csv", dtype={"THCIC_ID": str})
hy = pd.read_csv("hosp_year_2016_2019.tsv", sep="\t", dtype={"THCIC_ID": str})
hy = hy[hy.year.between(2016, 2019) & ~hy.THCIC_ID.isin(["999999", "999998"])]  # masked hospital codes

piv = hy.pivot_table(index="THCIC_ID", columns="year", values=["discharges", "pci_ami"], aggfunc="sum", fill_value=0)
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
agg = hy.groupby("THCIC_ID").agg(ed_max=("ed_flag_admits", "max"), stemi_direct_2016_19=("stemi_direct", "sum"),
                                 stemi_transfer_in_2016_19=("stemi_transfer_in", "sum"))
T = G.merge(piv, left_on="THCIC_ID", right_index=True, how="left").merge(agg, left_on="THCIC_ID", right_index=True, how="left")
T["has_ed"] = T.ed_max.fillna(0) >= 100
for y in range(2016, 2020):
    T[f"pci_ami_{y}"] = T.get(f"pci_ami_{y}", 0).fillna(0).astype(int)
    T[f"pci_capable_{y}"] = T[f"pci_ami_{y}"] >= 10
    T[f"active_{y}"] = T.get(f"discharges_{y}", 0).fillna(0) >= 50
cols = list(G.columns) + ["has_ed"] + [f"active_{y}" for y in range(2016, 2020)] + \
    [f"pci_ami_{y}" for y in range(2016, 2020)] + [f"pci_capable_{y}" for y in range(2016, 2020)] + \
    ["stemi_direct_2016_19", "stemi_transfer_in_2016_19"]
T[cols].to_csv("hospitals_tx_2016_2019.csv", index=False)
print(f"hospitals_tx_2016_2019.csv: {len(T)} hospitals; PCI-capable by year:",
      {y: int(T[f'pci_capable_{y}'].sum()) for y in range(2016, 2020)})
