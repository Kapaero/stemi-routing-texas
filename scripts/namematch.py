"""Name normalisation and similarity used for hospital matching."""
import json
import re
from difflib import SequenceMatcher

import pandas as pd

ABBR = {
    "SAINT": "ST", "MED": "MEDICAL", "CTR": "CENTER", "CNTR": "CENTER", "CENTRE": "CENTER", "HOSP": "HOSPITAL",
    "REG": "REGIONAL", "RGNL": "REGIONAL", "MEM": "MEMORIAL", "HLTH": "HEALTH", "SYS": "SYSTEM",
    "UNIV": "UNIVERSITY", "COMM": "COMMUNITY", "CMNTY": "COMMUNITY", "HOSPS": "HOSPITALS", "FT": "FORT",
    "MT": "MOUNT", "BAPT": "BAPTIST", "METH": "METHODIST", "PRESBY": "PRESBYTERIAN", "REHAB": "REHABILITATION",
    "REHABILITATIVE": "REHABILITATION", "BEHAV": "BEHAVIORAL", "SPEC": "SPECIALTY", "CHILDRENS": "CHILDREN",
    "CHILDREN'S": "CHILDREN", "DAVIDS": "DAVID", "LUKES": "LUKE", "JOSEPHS": "JOSEPH", "MARKS": "MARK",
    "HCA": "", "THE": "", "OF": "", "AT": "", "INC": "", "LLC": "", "LP": "", "LTD": "", "DBA": "", "AND": "",
}
GENERIC = {"HOSPITAL", "MEDICAL", "CENTER", "HEALTH", "REGIONAL", "SYSTEM", "CAMPUS", "TEXAS", "MEMORIAL", "COMMUNITY"}


def norm_tokens(s):
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    out = []
    for t in s.split():
        t = ABBR.get(t, t)
        if t:
            out.append(t)
    return out


def norm_city(s):
    return "".join(norm_tokens(s)).replace("SAINT", "ST")


def score(a, b):
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    overlap = len(inter) / min(len(ta), len(tb))
    distinctive = len(inter - GENERIC) / max(1, min(len(ta - GENERIC), len(tb - GENERIC)))
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return round(0.4 * overlap + 0.4 * distinctive + 0.2 * seq, 3)

