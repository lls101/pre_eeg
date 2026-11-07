"""5-site LODO splits (pre-registered 2026-11-07 before any 5-site model run).

Same procedure, seed and inner 5-fold StratifiedGroupKFold as m0_splits.py; the 3-site splits.json is untouched.
Output: results/m0/splits_5site.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from m0_splits import SEED, build_splits, check  # noqa: E402
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "m0"


def main():
    ms = pd.read_csv(OUT / "main_set_5site.csv", keep_default_na=False)
    assert ms.uid.is_unique
    folds = build_splits(ms)
    check(folds, ms)
    lab = ms.set_index("uid").label
    for ts, f in folds.items():
        print(f"test={ts}: train n={len(f['train'])} ({(lab[f['train']] == 'PD').sum()} PD) | test n={len(f['test'])} "
              f"({(lab[f['test']] == 'PD').sum()} PD) | inner {[len(v) for v in f['inner_val']]}")
    old = json.loads((OUT / "splits.json").read_text())["folds"]
    print("3-site splits.json untouched:", sorted(old))
    (OUT / "splits_5site.json").write_text(json.dumps(dict(seed=SEED, folds=folds), indent=1))


if __name__ == "__main__":
    main()
