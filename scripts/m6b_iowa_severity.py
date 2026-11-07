"""Iowa (ds004584) severity associations from the 5-site LODO predictions, same method as m6 T4 (ds007526):
PD only, partial Spearman (rank residualisation on age, df = n-3), BH over the 10 tests of this cohort.
UPDRS = the 'UPDRS score' column of ds004584 participants.tsv (part not specified by the dataset).
Output: results/5site/m6/T4b_iowa_severity.csv"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
os.environ["PD_SITES"] = "5"
sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, _ROOT  # noqa: E402
from m6_5site_analysis import MODELS, bh, partial_rank, seedavg  # noqa: E402

subj = pd.read_csv(_ROOT / "m0" / "subjects.csv").set_index("uid")
rows = []
for m in MODELS:
    P = seedavg(m, "ds004584")
    P = P[P.y == 1].join(subj[["UPDRS", "MOCA", "age"]], on="uid")
    for v in ["UPDRS", "MOCA"]:
        z = P[["p", v, "age"]].apply(pd.to_numeric, errors="coerce").dropna()
        r0, p0 = spearmanr(z.p, z[v])
        r, pv = partial_rank(z.p.values, z[v].values, z.age.values)
        rows.append(dict(model=m, variable=f"{v}|age", n=len(z), r=r, p=pv, r_unadjusted=r0, p_unadjusted=p0))
T = pd.DataFrame(rows); T["q_bh"] = bh(T.p)
T.to_csv(RESULTS / "m6" / "T4b_iowa_severity.csv", index=False)
print(T.round(3).to_string())
