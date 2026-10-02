# Texas PUDF base1 (tab) -> AMI cohort funnel + per-hospital aggregates (no individual rows kept)
BEGIN { FS = "\t" }
$1 == "RECORD_ID" { next }
{
  sub(/\r$/, "")
  q = $2; h = $3; dx = $33
  tot[q]++; totall++

  pci = 0; pciday = 999; cabg = 0; cath = 0
  for (i = 103; i <= 151; i += 2) {
    p = $i
    if (p == "") continue
    if (p ~ /^(027[0-3][34]|02C[0-3][34])/) { pci = 1; d = $(i + 1) + 0; if (d < pciday) pciday = d }
    else if (p ~ /^021[0-3]/) cabg = 1
    else if (p ~ /^(B21|4A023N)/) cath = 1
  }
  if (pci) pci_all[h]++

  if (substr(dx, 1, 3) != "I21") next
  ami[q]++
  t = (dx ~ /^I21[0-3]/) ? "STEMI" : ((dx ~ /^I214/) ? "NSTEMI" : "OTHER_I21")
  amitype[t]++
  if (pci) pci_ami[h]++
  if (pci && t == "STEMI") pci_stemi[h]++
  if (t == "OTHER_I21") next

  f0++; src[$5]++
  if ($34 != "Y") { poa_out[$34]++; next }
  f1++
  if ($4 != "1" && $4 != "2") next
  f2++
  if ($5 == "4") next
  f3++
  if ($22 !~ /^(0[6-9]|1[0-9]|2[0-1]|2[3-6])$/) next
  f4++
  if ($11 != "TX") next
  if ($12 ~ /^[0-9][0-9][0-9][0-9][0-9]$/) zipfmt["zip5"]++
  else if ($12 ~ /^[0-9][0-9][0-9]/) zipfmt["zip3_only"]++
  else zipfmt["blank_or_other"]++
  if ($12 !~ /^[0-9][0-9][0-9][0-9][0-9]$/) next
  f5++
  if (h == "999999") next
  f6++

  cohT[t]++; coh[h]++; zips[$12] = 1; stat[$16]++; ed[$167]++
  if (pci) cpci[t]++
  if (pci && pciday <= 1) cpci01[t]++
  if (cabg) ccabg[t]++
  if (pci || cath) ccath[t]++
  if ($16 == "20") cdied[t]++
  if ($16 == "50" || $16 == "51") chosp[t]++
  if ($16 == "02") ctrout[t]++
  if (t == "STEMI") {
    cohS[h]++
    if ($16 == "20") diedS[h]++
    if ($16 == "02") trS[h]++
    if (pci && pciday <= 1) ppciS[h]++
  }
}
END {
  printf "== records per quarter ==\n"
  for (q in tot) printf "  %s  all=%d  I21_principal=%d\n", q, tot[q], ami[q]
  printf "  total records 2019: %d\n", totall
  printf "== I21 principal by type ==\n"
  for (t in amitype) printf "  %-10s %d\n", t, amitype[t]
  printf "== SOURCE_OF_ADMISSION among STEMI+NSTEMI (before filters) ==\n"
  for (s in src) printf "  src=%-3s %d\n", s, src[s]
  printf "== funnel (STEMI+NSTEMI) ==\n"
  printf "  principal STEMI/NSTEMI      %d\n", f0
  printf "  + POA=Y                     %d\n", f1
  printf "  + emergency/urgent          %d\n", f2
  printf "  + not transfer-in (src!=4)  %d\n", f3
  printf "  + adult                     %d\n", f4
  printf "  + TX resident, full ZIP5    %d\n", f5
  printf "  + hospital not masked       %d\n", f6
  printf "  ZIP format among TX adults: "; for (z in zipfmt) printf "%s=%d ", z, zipfmt[z]; printf "\n"
  nz = 0; for (z in zips) nz++
  printf "  unique patient ZIP5 in cohort: %d\n", nz
  printf "== cohort outcomes ==\n"
  for (t in cohT) printf "  %-7s n=%d  died=%d (%.1f%%)  hospice=%d  transfer_out=%d (%.1f%%)  PCI=%d (%.1f%%)  PCI_day0-1=%d (%.1f%%)  CABG=%d  cath_or_PCI=%d (%.1f%%)\n", t, cohT[t], cdied[t], 100*cdied[t]/cohT[t], chosp[t], ctrout[t], 100*ctrout[t]/cohT[t], cpci[t], 100*cpci[t]/cohT[t], cpci01[t], 100*cpci01[t]/cohT[t], ccabg[t], ccath[t], 100*ccath[t]/cohT[t]
  printf "  ED flag: "; for (e in ed) printf "[%s]=%d ", e, ed[e]; printf "\n"
  printf "== hospitals ==\n"
  nh = 0; for (h in coh) nh++
  printf "  hospitals receiving cohort patients: %d\n", nh
  split("1 10 36", thr, " ")
  for (k = 1; k <= 3; k++) {
    T = thr[k]; ncap = 0; capS = 0; nonS = 0; trC = 0; trN = 0; dC = 0; dN = 0
    for (h in pci_ami) if (pci_ami[h] >= T) ncap++
    for (h in cohS) {
      if (pci_ami[h] + 0 >= T) { capS += cohS[h]; trC += trS[h]; dC += diedS[h] }
      else { nonS += cohS[h]; trN += trS[h]; dN += diedS[h] }
    }
    printf "  PCI-capable if >=%d AMI-PCIs/yr: %d hospitals | STEMI first-admitted there: %d (%.1f%%), transfer-out %.1f%%, died %.1f%% | at non-capable: %d, transfer-out %.1f%%, died %.1f%%\n", T, ncap, capS, 100*capS/(capS+nonS), (capS?100*trC/capS:0), (capS?100*dC/capS:0), nonS, (nonS?100*trN/nonS:0), (nonS?100*dN/nonS:0)
  }
  out = ENVIRON["HOSP_OUT"]
  if (out != "") {
    print "THCIC_ID\tcohort\tcohort_stemi\tpci_all\tpci_ami\tpci_stemi\tstemi_died\tstemi_trout\tstemi_pci01" > out
    for (h in pci_all) all[h] = 1
    for (h in coh) all[h] = 1
    for (h in all) printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", h, coh[h], cohS[h], pci_all[h], pci_ami[h], pci_stemi[h], diedS[h], trS[h], ppciS[h] > out
  }
}
