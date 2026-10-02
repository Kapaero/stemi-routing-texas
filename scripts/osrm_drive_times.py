"""Road travel times (OSRM public demo server) from patient ZIP internal points to the nearest hospitals.
Only public coordinates are sent (Census ZCTA internal points, hospital locations); <= 1 request per second.
Output: ext/osrm_zip_hosp.csv (pat_zip, THCIC_ID, minutes, road_km). Run from the 'derived' folder."""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

NN = 40           # hospitals per ZIP (nearest by straight line)
MAXC = 100        # coordinates per table request
SRC_PER_REQ = 20

P = pd.read_csv("stemi_patients_2016_2019.tsv", sep="\t", dtype=str, usecols=["pat_zip"])
Z = pd.read_csv("zcta2010_tx_internal_points.csv", dtype={"zcta5": str}).set_index("zcta5")
H = pd.read_csv("hospitals_tx_2016_2019.csv", dtype={"THCIC_ID": str})
H = H[H.lat.notna()].reset_index(drop=True)
zips = sorted(set(P.pat_zip) & set(Z.index))
zl = Z.loc[zips]


def hav(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


d = hav(zl.lat.values[:, None], zl.lon.values[:, None], H.lat.values[None], H.lon.values[None])
near = np.argsort(d, axis=1)[:, :NN]
# spatially coherent groups of ZIPs: sort by a coarse grid, then chunk
key = np.lexsort((np.round(zl.lon.values / 0.4), np.round(zl.lat.values / 0.4)))
groups = [key[i:i + SRC_PER_REQ] for i in range(0, len(key), SRC_PER_REQ)]


def table(src_pts, dst_pts):
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in src_pts + dst_pts)
    ns = len(src_pts)
    url = (f"https://router.project-osrm.org/table/v1/driving/{coords}?sources={';'.join(map(str, range(ns)))}"
           f"&destinations={';'.join(map(str, range(ns, ns + len(dst_pts))))}&annotations=duration,distance")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stemi-routing-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.loads(r.read().decode())
            if j.get("code") == "Ok":
                return j
        except Exception as e:  # noqa: BLE001
            print("  retry", attempt, type(e).__name__, str(e)[:80])
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("OSRM request failed")


rows, nreq, t0 = [], 0, time.time()
for g in groups:
    dst = sorted(set(near[g].ravel()))
    src_pts = [(zl.lat.values[i], zl.lon.values[i]) for i in g]
    step = MAXC - len(src_pts)
    for k in range(0, len(dst), step):
        chunk = dst[k:k + step]
        j = table(src_pts, [(H.lat.values[h], H.lon.values[h]) for h in chunk])
        nreq += 1
        for a, i in enumerate(g):
            want = set(near[i])
            for b, h in enumerate(chunk):
                if h in want:
                    du, di = j["durations"][a][b], j["distances"][a][b]
                    rows.append((zips[i], H.THCIC_ID[h], None if du is None else du / 60, None if di is None else di / 1000))
        time.sleep(1.05)
    if nreq % 20 == 0:
        print(f"  {nreq} requests, {len(rows)} pairs, {time.time() - t0:.0f}s", flush=True)

out = pd.DataFrame(rows, columns=["pat_zip", "THCIC_ID", "minutes", "road_km"])
out.to_csv("ext/osrm_zip_hosp.csv", index=False)
print(f"done: {nreq} requests, {len(out)} ZIP-hospital pairs, missing {out.minutes.isna().sum()}, {time.time() - t0:.0f}s")
