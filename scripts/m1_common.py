"""Shared loading + evaluation utilities for M1+ (subject-level metrics)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from paths import RESULTS as _ROOT  # noqa: E402

# PD_SITES=5 -> 5-site extension (pre_v1_5site cache, splits_5site.json); outputs go to results/5site/<milestone>
# so the 3-site outputs are never overwritten. Default (unset/3) = original 3-site study.
FIVE = os.environ.get("PD_SITES", "3") == "5"
CACHE = _ROOT / "cache" / ("pre_v1_5site" if FIVE else "pre_v1")
SPLITS = _ROOT / "m0" / ("splits_5site.json" if FIVE else "splits.json")
RESULTS = _ROOT / "5site" if FIVE else _ROOT
SFREQ = 200.0


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(CACHE / "meta.csv")
    m = m[m.ok].reset_index(drop=True)
    m["y"] = (m.label == "PD").astype(int)
    return m


def load_x(uid: str) -> np.ndarray:
    return np.load(CACHE / f"{uid}.npy")


def load_splits() -> dict:
    return json.loads(SPLITS.read_text())["folds"]


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def balanced_acc(y, p, thr) -> float:
    yhat = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y)
    tpr = (yhat[y == 1] == 1).mean()
    tnr = (yhat[y == 0] == 0).mean()
    return float((tpr + tnr) / 2)


def ece(y, p, n_bins=10) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(n_bins) if (idx == b).any()))


def boot_auc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    s = [roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))],
                       np.r_[p[rng.choice(i1, len(i1))], p[rng.choice(i0, len(i0))]]) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def evaluate(y, p, thr) -> dict:
    lo, hi = boot_auc_ci(y, p)
    return dict(n=len(y), n_pd=int(np.sum(y)), auroc=float(roc_auc_score(y, p)), auroc_lo=lo, auroc_hi=hi,
                ba_train_thr=balanced_acc(y, p, thr), ba_05=balanced_acc(y, p, 0.5), ece=ece(y, p), thr=thr)


def summarize(df: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Site-macro averages over test sites (and seeds if present)."""
    cols = ["auroc", "ba_train_thr", "ba_05", "ece"]
    per_site = df.groupby(list(by) + ["test_site"])[cols].mean()
    macro = per_site.groupby(list(by)).mean().add_prefix("macro_")
    return per_site.round(3), macro.round(3)
