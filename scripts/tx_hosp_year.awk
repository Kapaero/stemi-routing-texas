# Per hospital x year aggregates (facility-level only): all discharges, ED-flagged admissions,
# STEMI direct / transfer-in admissions, PCI performed on AMI patients
BEGIN { FS = "\t" }
$1 == "RECORD_ID" { next }
{
  sub(/\r$/, "")
  y = substr($2, 1, 4); h = $3; k = h SUBSEP y
  n[k]++
  if ($167 == "Y") ed[k]++
  if ($4 == "1" || $4 == "2") emerg[k]++
  if (substr($33, 1, 3) != "I21") next
  pci = 0
  for (i = 103; i <= 151; i += 2) if ($i ~ /^(027[0-3][34]|02C[0-3][34])/) { pci = 1; break }
  if (pci) pciami[k]++
  if ($33 ~ /^I21[0-3]/ && $34 == "Y" && ($4 == "1" || $4 == "2")) {
    if ($5 == "4" || $5 == "6") stin[k]++; else stdir[k]++
  }
}
END {
  print "THCIC_ID\tyear\tdischarges\ted_flag_admits\temerg_urgent_admits\tstemi_direct\tstemi_transfer_in\tpci_ami"
  for (k in n) { split(k, a, SUBSEP); printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n", a[1], a[2], n[k], ed[k], emerg[k], stdir[k], stin[k], pciami[k] }
}
