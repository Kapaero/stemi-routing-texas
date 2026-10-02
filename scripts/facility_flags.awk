# Facility type flags (teaching, psychiatric, pediatric) from the PUDF Facility_type files, any quarter 2016-2019.
# Usage: unzip -p <zips> 'Facility_type*' | awk -f facility_flags.awk > facility_flags.tsv
BEGIN { FS = "\t" }
{
  sub(/\r$/, "")
  if ($1 == "THCIC_ID" || $1 == "") next
  t[$1] = t[$1] || ($3 != "")
  p[$1] = p[$1] || ($4 != "")
  c[$1] = c[$1] || ($10 != "")
}
END {
  print "THCIC_ID\tteaching\tpsych\tpeds"
  for (h in t) print h "\t" (t[h] ? 1 : 0) "\t" (p[h] ? 1 : 0) "\t" (c[h] ? 1 : 0)
}
