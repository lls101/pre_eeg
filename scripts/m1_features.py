"""M1 R003 + R004: spectral / aperiodic features, site-prior null, logistic regression.

Features per subject (19 ch, Welch 4 s / 50% overlap on the 60 s segment):
  log absolute + relative band power (delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45),
  FOOOF (2-40 Hz, fixed aperiodic) exponent + offset, alpha peak CF + PW (7-14 Hz).
Protocols:
  LODO  : 3 outer folds from splits.json; C chosen by inner grouped CV AUROC; threshold = Youden on inner OOF.
  POOLED: 5-fold StratifiedGroupKFold over all subjects (sites mixed) -> replicates the pooled benchmark.
  site-prior null: score = PD prevalence of the subject's site in the training data (no EEG).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fooof import FOOOFGroup
from joblib import Parallel, delayed
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, SFREQ, balanced_acc, evaluate, load_meta, load_splits, load_x, summarize, youden_threshold  # noqa
from paths import CH19  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "m1"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CS = [0.001, 0.01, 0.1, 1.0]


def subject_features(uid: str) -> dict:
    x = load_x(uid)
    f, pxx = welch(x, fs=SFREQ, nperseg=int(4 * SFREQ), noverlap=int(2 * SFREQ), axis=-1)
    feats = {}
    tot = pxx[:, (f >= 1) & (f < 45)].sum(axis=1)
    for b, (lo, hi) in BANDS.items():
        bp = pxx[:, (f >= lo) & (f < hi)].sum(axis=1)
        for c, ch in enumerate(CH19):
            feats[f"logabs_{b}_{ch}"] = np.log10(bp[c])
            feats[f"rel_{b}_{ch}"] = bp[c] / tot[c]
    fg = FOOOFGroup(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fg.fit(f, pxx, freq_range=(2, 40))
    ap = fg.get_params("aperiodic_params")  # (n_ch, 2): offset, exponent
    peaks = fg.get_params("peak_params")    # rows: CF, PW, BW, ch_index
    for c, ch in enumerate(CH19):
        feats[f"ap_offset_{ch}"], feats[f"ap_exp_{ch}"] = ap[c, 0], ap[c, 1]
        pk = peaks[(peaks[:, -1] == c) & (peaks[:, 0] >= 7) & (peaks[:, 0] <= 14)] if peaks.size else np.empty((0, 4))
        if len(pk):
            best = pk[np.argmax(pk[:, 1])]
            feats[f"alpha_cf_{ch}"], feats[f"alpha_pw_{ch}"] = best[0], best[1]
        else:
            feats[f"alpha_cf_{ch}"], feats[f"alpha_pw_{ch}"] = np.nan, 0.0
    feats["fooof_r2_mean"] = float(np.mean(fg.get_params("r_squared")))
    return feats


def build_feature_table(meta: pd.DataFrame) -> pd.DataFrame:
    path = OUT / "features_v1.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = Parallel(n_jobs=12)(delayed(subject_features)(u) for u in meta.uid)
    F = pd.DataFrame(rows)
    F.insert(0, "uid", meta.uid.values)
    F.to_csv(path, index=False)
    return F


def impute(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def fit_lr(Xtr, ytr, gtr, strata, seed=0):
    """Pick C by inner grouped CV AUROC; return model refit on all train + OOF threshold."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best = (-1, None, None)
    for C in CS:
        oof = np.zeros(len(ytr))
        for tr, va in sgkf.split(Xtr, strata, groups=gtr):
            a, b = impute(Xtr[tr], Xtr[va])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            m.fit(a, ytr[tr])
            oof[va] = m.predict_proba(b)[:, 1]
        from sklearn.metrics import roc_auc_score
        s = roc_auc_score(ytr, oof)
        if s > best[0]:
            best = (s, C, oof)
    _, C, oof = best
    a, _ = impute(Xtr, Xtr)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(a, ytr)
    return m, C, youden_threshold(ytr, oof)


def site_prior_scores(train_meta, test_meta):
    prev = train_meta.groupby("dataset").y.mean()
    glob = train_meta.y.mean()
    return test_meta.dataset.map(prev).fillna(glob).to_numpy(), train_meta.dataset.map(prev).to_numpy()


def main():
    meta = load_meta()
    F = build_feature_table(meta)
    D = meta.merge(F, on="uid")
    fcols = [c for c in F.columns if c != "uid" and c != "fooof_r2_mean"]
    groups = {"all": fcols, "aperiodic": [c for c in fcols if c.startswith("ap_")],
              "bandpower": [c for c in fcols if c.startswith(("logabs_", "rel_"))]}
    print("FOOOF mean R^2 by site:", D.groupby("dataset").fooof_r2_mean.median().round(3).to_dict())
    res = []

    # ---- LODO
    for ts, fold in load_splits().items():
        tr = D[D.uid.isin(fold["train"])].reset_index(drop=True)
        te = D[D.uid.isin(fold["test"])].reset_index(drop=True)
        strata = (tr.dataset + "_" + tr.label).to_numpy()
        p_te, p_tr = site_prior_scores(tr, te)
        res.append(dict(protocol="LODO", model="site_prior_null", test_site=ts,
                        **evaluate(te.y, p_te, youden_threshold(tr.y, p_tr))))  # constant score -> AUROC exactly 0.5 (audit fix: no jitter)
        for gname, cols in groups.items():
            m, C, thr = fit_lr(tr[cols].to_numpy(float), tr.y.to_numpy(), tr.uid.to_numpy(), strata)
            _, Xte = impute(tr[cols].to_numpy(float), te[cols].to_numpy(float))
            p = m.predict_proba(Xte)[:, 1]
            res.append(dict(protocol="LODO", model=f"LR_{gname}", test_site=ts, C=C, **evaluate(te.y, p, thr)))
            pd.DataFrame(dict(uid=te.uid, y=te.y, p=p)).to_csv(OUT / f"pred_LODO_LR_{gname}_{ts}.csv", index=False)

    # ---- POOLED (sites mixed; replicates the standard benchmark)
    strata_all = (D.dataset + "_" + D.label).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261106)
    pooled = {k: np.zeros(len(D)) for k in ["site_prior_null"] + [f"LR_{g}" for g in groups]}
    thr_subj = {k: np.zeros(len(D)) for k in pooled}  # audit fix: each subject uses ITS OWN outer-fold threshold
    for tr_i, te_i in sgkf.split(D, strata_all, groups=D.uid):
        tr, te = D.iloc[tr_i].reset_index(drop=True), D.iloc[te_i]
        p_te, p_tr = site_prior_scores(tr, te)
        pooled["site_prior_null"][te_i] = p_te
        thr_subj["site_prior_null"][te_i] = youden_threshold(tr.y, p_tr)
        for gname, cols in groups.items():
            m, C, thr = fit_lr(tr[cols].to_numpy(float), tr.y.to_numpy(), tr.uid.to_numpy(),
                               (tr.dataset + "_" + tr.label).to_numpy())
            _, Xte = impute(tr[cols].to_numpy(float), te[cols].to_numpy(float))
            pooled[f"LR_{gname}"][te_i] = m.predict_proba(Xte)[:, 1]
            thr_subj[f"LR_{gname}"][te_i] = thr
    for k, p in pooled.items():
        r = evaluate(D.y, p, 0.5)
        r["ba_train_thr"] = balanced_acc(D.y, p - thr_subj[k], 0.0)
        r["thr"] = float(np.nan)
        res.append(dict(protocol="POOLED", model=k, test_site="pooled", **r))

    R = pd.DataFrame(res)
    R.to_csv(OUT / "results_features.csv", index=False)
    pd.set_option("display.width", 200)
    print(R[["protocol", "model", "test_site", "n", "n_pd", "auroc", "auroc_lo", "auroc_hi", "ba_train_thr", "ba_05", "ece"]].round(3).to_string())
    per_site, macro = summarize(R[R.protocol == "LODO"])
    print("\nLODO site-macro:\n", macro.to_string())


if __name__ == "__main__":
    main()
