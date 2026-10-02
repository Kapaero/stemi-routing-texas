"""Final address table: stage-2 matches + acceptance rules + manual campus overrides."""
import pandas as pd

d = pd.read_csv("hosp_addresses_stage2.csv", dtype=str).fillna("")
d["s1"] = d.s1.astype(float)
d["stemi"] = d.stemi.astype(float)

# 1) an exact CMS POS name match is the CMS-registered physical location; NPI rivals are usually admin/old addresses
d.loc[(d.status == "review") & (d.src == "POS") & (d.s1 >= 0.95), "status"] = "auto"
d.loc[(d.status == "review") & (d.src == "NPI") & (d.s1 >= 0.95), "status"] = "auto"

# 2) manual campus-level overrides (multi-campus systems sharing one CCN, renamed or unmatched hospitals)
M = {
    "119000": ("11800 Astoria Blvd", "Houston", "77089", "Memorial Hermann Southeast campus"),
    "407000": ("7600 Beechnut St", "Houston", "77074", "Memorial Hermann Southwest campus"),
    "172000": ("1635 N Loop W", "Houston", "77008", "Memorial Hermann Greater Heights campus"),
    "975201": ("27800 Northwest Fwy", "Cypress", "77433", "Memorial Hermann Cypress campus"),
    "975138": ("16100 South Fwy", "Pearland", "77584", "Memorial Hermann Pearland campus"),
    "347000": ("6411 Fannin St", "Houston", "77030", "Memorial Hermann TMC"),
    "615000": ("9250 Pinecroft Dr", "The Woodlands", "77380", "Memorial Hermann The Woodlands"),
    "398001": ("600 Elizabeth St", "Corpus Christi", "78404", "CHRISTUS Spohn Shoreline"),
    "398002": ("5950 Saratoga Blvd", "Corpus Christi", "78414", "CHRISTUS Spohn South"),
    "398000": ("2606 Hospital Blvd", "Corpus Christi", "78405", "CHRISTUS Spohn Memorial"),
    "703002": ("3315 S Alameda St", "Corpus Christi", "78411", "CCMC Doctors Regional"),
    "703000": ("7101 S Padre Island Dr", "Corpus Christi", "78412", "CCMC Bay Area"),
    "703003": ("7002 Williams Dr", "Corpus Christi", "78412", "CCMC Heart Hospital"),
    "704004": ("13725 Northwest Blvd", "Corpus Christi", "78410", "CCMC Northwest"),
    "459000": ("1504 Taub Loop", "Houston", "77030", "Harris Health Ben Taub"),
    "384000": ("5656 Kelley St", "Houston", "77026", "Harris Health LBJ"),
    "549001": ("2831 E President George Bush Hwy", "Richardson", "75082", "Methodist Richardson Bush/Renner campus"),
    "000102": ("301 University Blvd", "Galveston", "77555", "UTMB John Sealy"),
    "975354": ("20380 Gulf Fwy", "Webster", "77598", "UTMB Clear Lake campus"),
    "975140": ("2240 Gulf Fwy S", "League City", "77573", "UTMB League City campus"),
    "126000": ("132 E Hospital Dr", "Angleton", "77515", "UTMB Angleton Danbury"),
    "168000": ("120 E Harris Ave", "San Angelo", "76903", "Shannon Medical Center"),
    "154003": ("6700 Interstate 10 W", "San Antonio", "78201", "Methodist Texsan"),
    "503001": ("7930 Floyd Curl Dr", "San Antonio", "78229", "St Luke's Baptist"),
    "339002": ("11212 State Highway 151", "San Antonio", "78251", "CHRISTUS Santa Rosa Westover Hills"),
    "035000": ("919 E 32nd St", "Austin", "78705", "St David's Medical Center"),
    "829000": ("3801 N Lamar Blvd", "Austin", "78756", "Heart Hospital of Austin"),
    "602000": ("901 W Ben White Blvd", "Austin", "78704", "St David's South Austin"),
    "335000": ("601 E 15th St", "Austin", "78701", "University Medical Center Brackenridge"),
    "975165": ("100 Medical Pkwy", "Lakeway", "78738", "BSW Lakeway"),
    "970200": ("100 Medical Pkwy", "Lakeway", "78738", "Lakeway Regional (same building, earlier ID)"),
    "453000": ("506 E San Antonio St", "Victoria", "77901", "DeTar Navarro"),
    "453001": ("101 Medical Dr", "Victoria", "77904", "DeTar North"),
    "095003": ("865 Deshong Dr", "Paris", "75460", "Paris Regional North campus"),
    "095002": ("820 Clarksville St", "Paris", "75460", "Paris Regional South campus"),
    "975321": ("21214 Northwest Fwy", "Cypress", "77429", "North Cypress (HCA)"),
    "843200": ("21214 Northwest Fwy", "Cypress", "77429", "North Cypress (earlier ID)"),
    "974390": ("11100 Shadow Creek Pkwy", "Pearland", "77584", "HCA Pearland"),
    "975139": ("5500 Frisco Square Blvd", "Frisco", "75034", "Medical City Frisco"),
    "001000": ("1600 Wallace Blvd", "Amarillo", "79106", "Baptist St Anthony's"),
    "730001": ("811 Wright St", "Arlington", "76012", "Texas Health Heart & Vascular"),
    "158000": ("4502 Medical Dr", "San Antonio", "78229", "University Hospital"),
    "047000": ("11801 South Fwy", "Burleson", "76028", "Texas Health Huguley"),
    "448001": ("6201 Harry Hines Blvd", "Dallas", "75390", "UTSW Clements"),
    "140002": ("1102 W Trenton Rd", "Edinburg", "78539", "Edinburg Regional"),
    "797101": ("5501 S McColl Rd", "Edinburg", "78539", "DHR campus (women's hospital)"),
    "975407": ("7777 Forest Ln", "Dallas", "75230", "Medical City Dallas campus"),
    "975385": ("5252 W University Dr", "McKinney", "75071", "BSW McKinney campus"),
    "537006": ("1901 SW H K Dodgen Loop", "Temple", "76502", "BSW McLane Children's"),
    "117002": ("18200 Katy Fwy", "Houston", "77094", "Texas Children's West Campus"),
    "975205": ("17580 Interstate 45 S", "The Woodlands", "77384", "Texas Children's The Woodlands"),
    "973100": ("1917 Ashland St", "Houston", "77008", "St Joseph Heights"),
    "101100": ("8230 N Loop 1604 W", "San Antonio", "78249", "Baptist ER Hausman"),
    "647000": ("16088 San Pedro Ave", "San Antonio", "78232", "Baptist ER Thousand Oaks"),
    "972810": ("25615 US Highway 281 N", "San Antonio", "78258", "Baptist ER Overlook"),
    "975179": ("7719 Interstate 35 S", "San Antonio", "78224", "Baptist ER Zarzamora"),
    "973000": ("10811 Town Center Dr", "San Antonio", "78251", "Baptist ER Westover Hills"),
    "973120": ("16977 Interstate 35 N", "Schertz", "78154", "Baptist ER Schertz"),
    "736000": ("301 Londonberry Dr", "Waco", "76712", "Providence DePaul Center"),
    "852000": ("4900 Mueller Blvd", "Austin", "78723", "Dell Children's"),
    "703005": ("6226 Saratoga Blvd", "Corpus Christi", "78414", "Bayview Behavioral"),
    "974800": ("1401 E Trinity Mills Rd", "Carrollton", "75006", "First Texas Hospital"),
    "973920": ("12500 South Fwy", "Burleson", "76028", "Baylor ER Burleson"),
    "000112": ("11937 US Highway 271", "Tyler", "75708", "UT Health Center Tyler"),
    "806500": ("800 E Dawson St", "Tyler", "75701", "Tyler ContinueCare (inside Mother Frances)"),
    "230001": ("22003 Southwest Fwy", "Richmond", "77469", "OakBend Williams Way campus"),
    "788003": ("1007 S William St", "Atlanta", "75551", "CHRISTUS St Michael Atlanta"),
    "367000": ("2323 Texas St", "Pecos", "79772", "Reeves County Hospital"),
    "559000": ("3201 S Water St", "Burnet", "78611", "Seton Highland Lakes"),
    "334002": ("1905 Highway 97 E", "Jourdanton", "78026", "South Texas Regional (earlier ID)"),
    "975221": ("1905 Highway 97 E", "Jourdanton", "78026", "Methodist Hospital South"),
}
for k, (a, c, z, note) in M.items():
    i = d.index[d.THCIC_ID == k]
    if len(i):
        d.loc[i, ["addr", "a_city", "a_zip", "src", "status"]] = [a, c, z, "MANUAL", "manual"]
        d.loc[i, "cname"] = note

# 3) leftovers without STEMI admissions (ED-only facilities): accept the best candidate as approximate
d.loc[(d.status == "review") & (d.stemi == 0) & (d.s1 >= 0.55), "status"] = "approx"

d.to_csv("hosp_addresses_final.csv", index=False)
print(d.status.value_counts().to_string())
print("STEMI admissions by status:", d.groupby("status").stemi.sum().to_dict())
left = d[d.status == "review"].sort_values("stemi", ascending=False)
print(left[["THCIC_ID", "name", "city", "stemi", "ed_max", "s1", "src", "cname", "addr"]].to_string(index=False))
