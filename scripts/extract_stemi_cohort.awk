# Patient-level STEMI cohort (Texas inpatient PUDF base1, 2016-2019). Stays local under the THCIC DUA.
# Inclusion: principal I21.0-I21.3 with POA=Y, emergency/urgent admission, adult, Texas resident with 5-digit ZIP,
#            hospital not masked (999999), discharge year 2016-2019.
# Comorbidities: secondary diagnoses with POA=Y only (present before the routing decision).
BEGIN {
  FS = "\t"; OFS = "\t"
  print "year", "quarter", "THCIC_ID", "src", "adm_type", "ed_flag", "pat_zip", "pat_county", "phr", "pat_status",
        "sex", "race", "ethnicity", "weekday", "los", "age_grp", "payer", "stemi_site",
        "dm", "htn", "hf", "ckd", "prior_mi", "prior_pci", "prior_cabg", "af", "copd", "cvd", "pvd", "dementia",
        "obesity", "smoking", "cancer", "arrest", "shock",
        "pci", "pci_day", "cabg", "cath", "died", "hospice", "transfer_out"
}
$1 == "RECORD_ID" { next }
{
  sub(/\r$/, "")
  if ($33 !~ /^I21[0-3]/ || $34 != "Y" || ($4 != "1" && $4 != "2")) next
  if ($22 !~ /^(0[6-9]|1[0-9]|2[0-1]|2[3-6])$/) next
  if ($11 != "TX" || $12 !~ /^[0-9][0-9][0-9][0-9][0-9]$/ || $3 == "999999") next
  y = substr($2, 1, 4) + 0
  if (y < 2016 || y > 2019) next

  site = ($33 ~ /^I210/) ? "anterior" : (($33 ~ /^I211/) ? "inferior" : (($33 ~ /^I212/) ? "other" : "unspecified"))
  split("dm htn hf ckd prior_mi prior_pci prior_cabg af copd cvd pvd dementia obesity smoking cancer arrest shock", nm, " ")
  for (k in nm) f[nm[k]] = 0
  for (i = 35; i <= 81; i += 2) {
    if ($(i + 1) != "Y") continue
    d = $i
    if (d ~ /^E(0[89]|1[0-3])/) f["dm"] = 1
    if (d ~ /^I1[0-6]/) f["htn"] = 1
    if (d ~ /^I50/) f["hf"] = 1
    if (d ~ /^N18/) f["ckd"] = 1
    if (d ~ /^I252/) f["prior_mi"] = 1
    if (d ~ /^Z955/) f["prior_pci"] = 1
    if (d ~ /^Z951/) f["prior_cabg"] = 1
    if (d ~ /^I48/) f["af"] = 1
    if (d ~ /^J4[34]/) f["copd"] = 1
    if (d ~ /^(Z8673|I69)/) f["cvd"] = 1
    if (d ~ /^(I70[2-9]|I739)/) f["pvd"] = 1
    if (d ~ /^(F0[1-3]|G30)/) f["dementia"] = 1
    if (d ~ /^E66/) f["obesity"] = 1
    if (d ~ /^(F17|Z87891)/) f["smoking"] = 1
    if (d ~ /^C/) f["cancer"] = 1
    if (d ~ /^I46/) f["arrest"] = 1
    if (d ~ /^R570/) f["shock"] = 1
  }
  pci = 0; pd = ""; cabg = 0; cath = 0
  for (i = 103; i <= 151; i += 2) {
    p = $i
    if (p == "") continue
    if (p ~ /^(027[0-3][34]|02C[0-3][34])/) { pci = 1; dd = $(i + 1) + 0; if (pd == "" || dd < pd) pd = dd }
    else if (p ~ /^021[0-3]/) cabg = 1
    else if (p ~ /^(B21|4A023N)/) cath = 1
  }
  st = $16
  print y, substr($2, 6, 1), $3, $5, $4, $167, $12, $14, $15, st, $17, $18, $19, $20, $21 + 0, $22, $23, site,
        f["dm"], f["htn"], f["hf"], f["ckd"], f["prior_mi"], f["prior_pci"], f["prior_cabg"], f["af"], f["copd"],
        f["cvd"], f["pvd"], f["dementia"], f["obesity"], f["smoking"], f["cancer"], f["arrest"], f["shock"],
        pci, pd, cabg, (cath || pci), (st == "20"), (st == "50" || st == "51"), (st ~ /^(02|05|43|66)$/)
}
