"""Geocode hospital street addresses with the US Census batch geocoder (public addresses only)."""
import csv
import io
import subprocess
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, dtype=str).fillna("")
todo = df[df.addr != ""]
buf = io.StringIO()
w = csv.writer(buf)
for _, r in todo.iterrows():
    w.writerow([r.THCIC_ID, r.addr, r.a_city, "TX", r.a_zip])
open("geo_in.csv", "w", newline="", encoding="utf-8").write(buf.getvalue())

res = subprocess.run(
    ["curl", "-s", "--max-time", "600", "-F", "addressFile=@geo_in.csv", "-F", "benchmark=Public_AR_Current",
     "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"],
    capture_output=True, text=True, encoding="utf-8")
rows = list(csv.reader(io.StringIO(res.stdout)))
geo = {}
for r in rows:
    if len(r) >= 3:
        lon = lat = ""
        if r[2] == "Match" and len(r) >= 6 and "," in r[5]:
            lon, lat = r[5].split(",")
        geo[r[0]] = {"geo_status": r[2], "geo_type": r[3] if len(r) > 3 else "", "geo_matched": r[4] if len(r) > 4 else "",
                     "lon": lon, "lat": lat}
g = pd.DataFrame.from_dict(geo, orient="index").reset_index().rename(columns={"index": "THCIC_ID"})
out = df.merge(g, on="THCIC_ID", how="left")
out.to_csv(dst, index=False)
print("sent", len(todo), "| returned", len(rows), "|", out.geo_status.value_counts(dropna=False).to_dict())
