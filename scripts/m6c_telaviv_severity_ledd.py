"""Tel Aviv (ds007526) severity associations adjusted for age AND LEDD (reviewer request), all 5 models.
Partial Spearman: ranks residualised on ranks of the covariates (OLS), t test df = n - 2 - k. BH over the 10 tests.
Output: results/5site/m6/T4c_telaviv_severity_age_ledd.csv"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, t as tdist
os.environ["PD_SITES"] = "5"
sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, _ROOT  # noqa: E402
from m6_5site_analysis import MODELS, bh, seedavg  # noqa: E402


def partial(a, b, C):
    R = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in C])
    res = lambda x: x - R @ np.linalg.lstsq(R, x, rcond=None)[0]  # noqa: E731
    r = pearsonr(res(rankdata(a)), res(rankdata(b)))[0]
    df = len(a) - 2 - len(C)
    return r, float(2 * tdist.sf(abs(r * np.sqrt(df / max(1 - r ** 2, 1e-12))), df))


rec = pd.read_csv(_ROOT / "m0" / "recordings_new.csv")
tv = rec[(rec.dataset == "ds007526") & (rec.label == "PD")][["uid", "age", "updrs3", "moca", "ledd"]]
rows = []
for m in MODELS:
    P = seedavg(m, "ds007526").merge(tv, on="uid")
    for v in ["updrs3", "moca"]:
        z = P.dropna(subset=[v, "age", "ledd"])
        r, p = partial(z.p.values, z[v].values, [z.age.values, z.ledd.values])
        rows.append(dict(model=m, variable=f"{v}|age+ledd", n=len(z), r=r, p=p))
T = pd.DataFrame(rows); T["q_bh"] = bh(T.p)
T.to_csv(RESULTS / "m6" / "T4c_telaviv_severity_age_ledd.csv", index=False)
print(T.round(3).to_string())
