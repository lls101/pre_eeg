"""M0: lock subject-level LODO outer folds + inner grouped CV, with leakage assertions.

Main analysis set: eyes-open rest, PD in ON medication, one recording per subject.
Output: results/m0/splits.json, results/m0/main_set.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "m0"
SEED = 20261106
N_INNER = 5


def main_set(rec: pd.DataFrame) -> pd.DataFrame:
    rest = rec[rec.task.str.lower().str.contains("rest")].copy()
    keep = rest[(rest.label == "HC") | (rest.med == "ON")]
    # exactly one recording per subject
    dup = keep.uid.value_counts()
    assert (dup == 1).all(), f"multiple main recordings: {dup[dup > 1].to_dict()}"
    return keep


def build_splits(ms: pd.DataFrame) -> dict:
    sites = sorted(ms.dataset.unique())
    folds = {}
    for test_site in sites:
        tr = ms[ms.dataset != test_site].reset_index(drop=True)
        te = ms[ms.dataset == test_site]
        strata = tr.dataset + "_" + tr.label
        sgkf = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
        inner = []
        for k, (_, vi) in enumerate(sgkf.split(tr, strata, groups=tr.uid)):
            inner.append(sorted(tr.uid.iloc[vi].tolist()))
        folds[test_site] = dict(train=sorted(tr.uid), test=sorted(te.uid), inner_val=inner)
    return folds


def check(folds: dict, ms: pd.DataFrame):
    lab = ms.set_index("uid").label
    site = ms.set_index("uid").dataset
    for ts, f in folds.items():
        tr, te = set(f["train"]), set(f["test"])
        assert not tr & te, f"{ts}: train/test overlap"
        assert set(site[list(te)]) == {ts}, f"{ts}: test not single site"
        assert ts not in set(site[list(tr)]), f"{ts}: test site in train"
        allv = [u for v in f["inner_val"] for u in v]
        assert len(allv) == len(set(allv)) and set(allv) == tr, f"{ts}: inner folds not a partition of train"
        for v in f["inner_val"]:
            labs = lab[v].value_counts()
            assert {"PD", "HC"} <= set(labs.index), f"{ts}: inner fold missing a class"
    # participant ids are only unique within dataset -> uid must carry dataset prefix
    assert all(u.split("_sub-")[0] in folds for u in ms.uid)
    print("all leakage assertions passed")


def main():
    rec = pd.read_csv(OUT / "recordings.csv")
    ms = main_set(rec)
    ms.to_csv(OUT / "main_set.csv", index=False)
    print(ms.groupby(["dataset", "label"]).size().unstack())
    folds = build_splits(ms)
    check(folds, ms)
    for ts, f in folds.items():
        lab = ms.set_index("uid").label
        print(f"test={ts}: train n={len(f['train'])} ({(lab[f['train']]=='PD').sum()} PD) | "
              f"test n={len(f['test'])} ({(lab[f['test']]=='PD').sum()} PD) | inner sizes "
              f"{[len(v) for v in f['inner_val']]}")
    (OUT / "splits.json").write_text(json.dumps(dict(seed=SEED, folds=folds), indent=1))


if __name__ == "__main__":
    main()
