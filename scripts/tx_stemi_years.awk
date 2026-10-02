# STEMI (POA=Y, emergency/urgent, adult) by discharge year and arrival mode; facility IDs per year
BEGIN { FS = "\t" }
$1 == "RECORD_ID" { next }
{
  sub(/\r$/, "")
  y = substr($2, 1, 4); h = $3; hosp[y, h] = 1
  if ($33 !~ /^I21[0-3]/) next
  if ($34 != "Y" || ($4 != "1" && $4 != "2")) next
  if ($22 !~ /^(0[6-9]|1[0-9]|2[0-1]|2[3-6])$/) next
  m = ($5 == "4" || $5 == "6") ? "transfer" : "direct"
  pci = 0; pd = 999
  for (i = 103; i <= 151; i += 2) { p = $i; if (p ~ /^(027[0-3][34]|02C[0-3][34])/) { pci = 1; d = $(i + 1) + 0; if (d < pd) pd = d } }
  n[y, m]++; ys[y] = 1
  if ($16 == "20") die[y, m]++
  if (pci && pd <= 0) p0[y, m]++
  if ($11 == "TX" && $12 ~ /^[0-9][0-9][0-9][0-9][0-9]$/ && h != "999999") ok[y, m]++
}
END {
  printf "year  STEMI  direct  transfer(%%)  died_direct  died_transfer  PCIday0_direct  PCIday0_transfer  usable(TX ZIP5)  hospitals_in_file\n"
  for (y in ys) { if (n[y, "direct"] < 100 || n[y, "transfer"] < 10) continue
    nh = 0; for (k in hosp) { split(k, a, SUBSEP); if (a[1] == y) nh++ }
    T = n[y, "direct"] + n[y, "transfer"]
    printf "%s  %5d  %6d  %5d (%.1f%%)  %10.1f%%  %12.1f%%  %13.1f%%  %15.1f%%  %6d  %4d\n", y, T, n[y, "direct"], n[y, "transfer"], 100 * n[y, "transfer"] / T, 100 * die[y, "direct"] / n[y, "direct"], 100 * die[y, "transfer"] / n[y, "transfer"], 100 * p0[y, "direct"] / n[y, "direct"], 100 * p0[y, "transfer"] / n[y, "transfer"], ok[y, "direct"] + ok[y, "transfer"], nh
  }
}
