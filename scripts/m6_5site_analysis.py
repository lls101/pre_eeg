"""Pre-registered 5-site analysis (refine-logs/M0M1_5SITE_REPORT.md), run with PD_SITES=5 outputs.

T1 main table  : mean-of-seed site-macro AUROC / BA (train-site threshold) per model, per site; plus macro over the
                 3 original sites and over the 2 new sites (descriptive split, same 5-site models).
T2 primary     : paired class-stratified subject bootstrap of seed-averaged predictions, each model minus EEGNet
                 ensemble, per site and site-macro (the C1 replication test).
T3 Turku       : (a) PD-ON vs HC and PD-OFF_break vs HC AUROC (descriptive, n = 7 / 13 vs 20);
                 (b) amplitude check: AUROC of raw RMS alone, Spearman of model score with RMS;
                 (c) subject-level exclusion (any > 200 uV window) -- uninformative, superseded by T6.
T5 (post hoc)  : ds007526 age: age-only AUROC, age-restricted / matched subsets, likelihood-ratio test.
T6 (pre-reg.)  : Turku window-level artefact exclusion for deep models, with a same-subject baseline.
T4 ds007526    : PD-only Spearman of score with UPDRS-III and MoCA, and age-adjusted partials (BH over the set).
                 Also reports the age difference HC vs PD (a possible age shortcut).
Outputs -> results/5site/m6/*.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr
from scipy.stats import t as tdist
from sklearn.metrics import roc_auc_score

os.environ["PD_SITES"] = "5"
sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, _ROOT, balanced_acc  # noqa: E402

OUT = RESULTS / "m6"
OUT.mkdir(parents=True, exist_ok=True)
SITES = ["ds002778", "ds003490", "ds004584", "ds007526", "turku"]
OLD, NEW = SITES[:3], SITES[3:]
REF = "EEGNet_ens5"
# model -> (subdir, file tag, seeds or None for deterministic LR)
MODELS = {"LR_all": ("m1", "LR_all", None), "LR_coh": ("m4", "LR_coh", None),
          "EEGNet_ens5": ("m1", "EEGNet_ens5", 3), "FT_LaBraM": ("m2", "FT_LaBraM", 3),
          "FT_CBraMod": ("m2", "FT_CBraMod", 3)}
RESULT_FILES = {"LR_all": ("m1/results_features.csv", "LR_all"), "LR_coh": ("m4/conn_lodo.csv", "LR_coh"),
                "EEGNet_ens5": ("m1/results_EEGNet_ens5.csv", None), "FT_LaBraM": ("m2/results_FT_LaBraM.csv", None),
                "FT_CBraMod": ("m2/results_FT_CBraMod.csv", None)}


def avail(m):
    return (RESULTS / RESULT_FILES[m][0]).exists()


def seedavg(m, s):
    d, tag, k = MODELS[m]
    if k is None:
        return pd.read_csv(RESULTS / d / f"pred_LODO_{tag}_{s}.csv")[["uid", "y", "p"]]
    return pd.concat([pd.read_csv(RESULTS / d / f"pred_LODO_{tag}_{s}_s{i}.csv") for i in range(k)]
                     ).groupby(["uid", "y"]).p.mean().reset_index()


def t1(models):
    rows = []
    for m in models:
        f, name = RESULT_FILES[m]
        R = pd.read_csv(RESULTS / f)
        R = R[(R.protocol == "LODO") & (R.model == (name or R.model))] if "protocol" in R else R
        if name:
            R = R[R.model == name]
        per = R.groupby("test_site")[["auroc", "ba_train_thr"]].mean()
        seed_sd = R.groupby("seed").auroc.mean().std() if "seed" in R and R.seed.nunique() > 1 else np.nan
        row = dict(model=m, macro5_auroc=per.auroc.mean(), macro5_ba=per.ba_train_thr.mean(), seed_sd_macro5=seed_sd,
                   macro_old3=per.auroc[OLD].mean(), macro_new2=per.auroc[NEW].mean())
        row.update({f"auroc_{s}": per.auroc[s] for s in SITES})
        row.update({f"ba_{s}": per.ba_train_thr[s] for s in SITES})
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T1_main_table.csv", index=False)
    return T


def strat_idx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]


def t2(models, n=4000):
    rng = np.random.default_rng(20261107)
    rows = []
    for m in [x for x in models if x != REF]:
        O = {s: seedavg(m, s).merge(seedavg(REF, s), on=["uid", "y"], suffixes=("_m", "_r")) for s in SITES}
        obs = {s: roc_auc_score(o.y, o.p_m) - roc_auc_score(o.y, o.p_r) for s, o in O.items()}
        B = {s: [] for s in SITES}
        for _ in range(n):
            for s, o in O.items():
                y = o.y.values; ix = strat_idx(y, rng)
                B[s].append(roc_auc_score(y[ix], o.p_m.values[ix]) - roc_auc_score(y[ix], o.p_r.values[ix]))
        B = {s: np.array(v) for s, v in B.items()}
        for s in SITES + ["macro5"]:
            b = np.mean([B[k] for k in SITES], 0) if s == "macro5" else B[s]
            d = np.mean(list(obs.values())) if s == "macro5" else obs[s]
            a_m = np.mean([roc_auc_score(o.y, o.p_m) for o in O.values()]) if s == "macro5" else roc_auc_score(O[s].y, O[s].p_m)
            rows.append(dict(model=m, vs=REF, site=s, auroc_model_seedavg=a_m, auroc_ref_seedavg=a_m - d, delta=d,
                             ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5),
                             ci_excludes_0=bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0),
                             # audit qualifier: Bonferroni over the 2 primary FM-vs-EEGNet comparisons (97.5% CI)
                             ci_lo_bonf2=np.percentile(b, 1.25), ci_hi_bonf2=np.percentile(b, 98.75)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T2_paired_bootstrap.csv", index=False)
    return T


def t3(models):
    rec = pd.read_csv(_ROOT / "m0" / "recordings_new.csv")
    qc = pd.read_csv(_ROOT / "cache" / "pre_v1_new" / "meta.csv")
    tk = rec[rec.dataset == "turku"][["uid", "label", "med"]].merge(qc[["uid", "rms_uv", "n_win_bad"]], on="uid")
    y = (tk.label == "PD").astype(int).values
    rows = [dict(model="RMS_only", subset="all", n_pd=int(y.sum()), auroc=roc_auc_score(y, tk.rms_uv))]
    for med in ["ON", "OFF_break"]:
        k = (tk.label == "HC") | (tk.med == med)
        rows.append(dict(model="RMS_only", subset=f"PD_{med}_vs_HC", n_pd=int((tk.med[k] == med).sum()),
                         auroc=roc_auc_score(y[k], tk.rms_uv[k])))
    for m in models:
        P = seedavg(m, "turku").merge(tk, on="uid")
        yy = P.y.values
        rows.append(dict(model=m, subset="all", n_pd=int(yy.sum()), auroc=roc_auc_score(yy, P.p),
                         rho_score_rms_all=spearmanr(P.p, P.rms_uv)[0],
                         rho_score_rms_PD=spearmanr(P.p[yy == 1], P.rms_uv[yy == 1])[0]))
        for med in ["ON", "OFF_break"]:
            k = ((P.label == "HC") | (P.med == med)).values
            rows.append(dict(model=m, subset=f"PD_{med}_vs_HC", n_pd=int((P.med[k] == med).sum()),
                             auroc=roc_auc_score(yy[k], P.p[k])))
        k = (P.n_win_bad == 0).values
        rows.append(dict(model=m, subset="no_bad_window_subjects", n_pd=int(yy[k].sum()), n_hc=int((1 - yy[k]).sum()),
                         auroc=roc_auc_score(yy[k], P.p[k])))
        # PD_ON vs PD_OFF score difference (descriptive)
        pd_ = P[P.label == "PD"]
        u = mannwhitneyu(pd_.p[pd_.med == "OFF_break"], pd_.p[pd_.med == "ON"])
        rows.append(dict(model=m, subset="PD_OFF_vs_PD_ON_score", n_pd=len(pd_),
                         auroc=u.statistic / ((pd_.med == "OFF_break").sum() * (pd_.med == "ON").sum()), p=u.pvalue))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T3_turku.csv", index=False)
    return T, tk


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def partial_rank(a, b, c):
    ra, rb, rc = (rankdata(v) for v in (a, b, c))
    res = lambda x: x - np.polyval(np.polyfit(rc, x, 1), rc)  # noqa: E731
    r = pearsonr(res(ra), res(rb))[0]
    df = len(a) - 3
    return r, float(2 * tdist.sf(abs(r * np.sqrt(df / max(1 - r ** 2, 1e-12))), df))


def t4(models):
    rec = pd.read_csv(_ROOT / "m0" / "recordings_new.csv")
    tv = rec[rec.dataset == "ds007526"][["uid", "label", "age", "updrs3", "moca"]]
    age = tv.groupby("label").age.describe()[["count", "mean", "std", "min", "max"]]
    rows = []
    for m in models:
        P = seedavg(m, "ds007526").merge(tv, on="uid")
        rows.append(dict(model=m, variable="age_all_subjects", n=len(P), r=spearmanr(P.p, P.age)[0],
                         p=spearmanr(P.p, P.age)[1], kind="spearman"))
        q = P[P.label == "PD"]
        for v in ["updrs3", "moca"]:
            z = q.dropna(subset=[v, "age"])
            r, pv = spearmanr(z.p, z[v])
            rows.append(dict(model=m, variable=v, n=len(z), r=r, p=pv, kind="spearman"))
            r, pv = partial_rank(z.p.values, z[v].values, z.age.values)
            rows.append(dict(model=m, variable=f"{v}|age", n=len(z), r=r, p=pv, kind="partial"))
    T = pd.DataFrame(rows)
    k = T.kind == "partial"
    T.loc[k, "q_bh"] = bh(T.loc[k, "p"])
    T.to_csv(OUT / "T4_ds007526_severity.csv", index=False)
    age.to_csv(OUT / "T4_ds007526_age_by_group.csv")
    return T, age


def t5(models):
    """POST-HOC (not pre-registered): age shortcut check for ds007526 (HC younger than PD).
    Age-only AUROC, and model AUROC in the age-overlap subset (PD restricted to HC age range) plus a
    1:k nearest-age matched subset (each HC matched to its 3 nearest-age PD without replacement)."""
    rec = pd.read_csv(_ROOT / "m0" / "recordings_new.csv")
    tv = rec[rec.dataset == "ds007526"][["uid", "label", "age"]]
    hc = tv[tv.label == "HC"]; pdx = tv[tv.label == "PD"].copy()
    lo, hi = hc.age.min(), hc.age.max()
    used, match = set(), []
    for a in hc.sort_values("age").age:
        c = pdx[~pdx.uid.isin(used)].assign(d=lambda d: (d.age - a).abs()).nsmallest(3, "d")
        used |= set(c.uid); match += list(c.uid)
    # audit follow-up: caliper matching (|age diff| <= 2 y), UP TO 3 PD per HC, greedy nearest without replacement
    used2, match2, hc_kept = set(), [], []
    for r in hc.sort_values("age").itertuples():
        c = pdx[~pdx.uid.isin(used2)].assign(d=lambda d: (d.age - r.age).abs())
        c = c[c.d <= 2.0].nsmallest(3, "d")
        if len(c):
            used2 |= set(c.uid); match2 += list(c.uid); hc_kept.append(r.uid)
    subsets = {"all": tv.uid, "age_overlap": tv[(tv.label == "HC") | tv.age.between(lo, hi)].uid,
               "age_matched_1to3": pd.concat([hc.uid, pd.Series(match)]),
               "age_caliper2y_1to3": pd.Series(hc_kept + match2)}
    rows = []
    for name, u in subsets.items():
        t = tv[tv.uid.isin(u)]; y = (t.label == "PD").astype(int)
        rows.append(dict(model="age_only", subset=name, n_hc=int((y == 0).sum()), n_pd=int(y.sum()),
                         age_hc=t.age[y == 0].mean(), age_pd=t.age[y == 1].mean(), auroc=roc_auc_score(y, t.age)))
        for m in models:
            P = seedavg(m, "ds007526"); P = P[P.uid.isin(u)]
            rows.append(dict(model=m, subset=name, n_hc=int((P.y == 0).sum()), n_pd=int(P.y.sum()),
                             auroc=roc_auc_score(P.y, P.p)))
    # audit follow-up: does the score add information beyond age? Likelihood-ratio test of
    # logit(PD) ~ age  vs  ~ age + logit(score), unpenalised, in the full ds007526 fold (descriptive, post hoc).
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    for m in models:
        P = seedavg(m, "ds007526").merge(tv, on="uid")
        z = np.log(np.clip(P.p, 1e-6, 1 - 1e-6) / (1 - np.clip(P.p, 1e-6, 1 - 1e-6)))
        X0 = ((P.age - P.age.mean()) / P.age.std()).to_numpy()[:, None]
        X1 = np.c_[X0, (z - z.mean()) / z.std()]
        ll = []
        for X in (X0, X1):
            f = LogisticRegression(penalty=None, max_iter=1000).fit(X, P.y)
            q = np.clip(f.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            ll.append(float(np.sum(P.y * np.log(q) + (1 - P.y) * np.log(1 - q))))
        lr = 2 * (ll[1] - ll[0])
        rows.append(dict(model=m, subset="LRtest_score_beyond_age", n_hc=int((P.y == 0).sum()), n_pd=int(P.y.sum()),
                         lr_stat=lr, p=float(chi2.sf(lr, 1))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T5_posthoc_ds007526_age.csv", index=False)
    return T


def t6(models):
    """PRE-REGISTERED Turku sensitivity, window level: re-score each Turku test subject using only the model
    windows (800 samples, stride 400, as in training) whose peak-to-peak <= 200 uV on every channel. Uses the
    per-window test logits saved by m1_eegnet / m2_finetune (deep models only; the LR features are not
    window-scored). Subjects with no clean window are dropped and counted."""
    from m1_common import load_x
    rows = []
    for m in [x for x in models if MODELS[x][2]]:
        d, tag, k = MODELS[m]
        ps, ys = [], None
        for i in range(k):
            f = RESULTS / d / f"testwin_LODO_{tag}_turku_s{i}.npz"
            if not f.exists():
                return pd.DataFrame()
            Z = np.load(f)
            uids, sid, lg, W, S = Z["uid"], Z["sid"], Z["logit"], int(Z["win"]), int(Z["stride"])
            p_sub, keep_n, tot_n = [], [], []
            for j, u in enumerate(uids):
                x = load_x(str(u)); w = lg[sid == j]
                ok = np.array([np.ptp(x[:, s * S:s * S + W], axis=1).max() <= 200.0 for s in range(len(w))])
                keep_n.append(int(ok.sum())); tot_n.append(len(w))
                p_sub.append(1 / (1 + np.exp(-w[ok].mean())) if ok.any() else np.nan)
            ps.append(np.array(p_sub))
        p = np.nanmean(ps, 0)
        P = pd.DataFrame(dict(uid=uids.astype(str), p=p, kept=keep_n, total=tot_n)).merge(
            seedavg(m, "turku").rename(columns={"p": "p_all"}), on="uid")
        v = P.p.notna()
        rows.append(dict(model=m, n_subjects_kept=int(v.sum()), n_pd=int(P.y[v].sum()), n_hc=int((1 - P.y[v]).sum()),
                         windows_kept_frac=float(P.kept.sum() / P.total.sum()),
                         windows_kept_frac_PD=float(P.kept[P.y == 1].sum() / P.total[P.y == 1].sum()),
                         windows_kept_frac_HC=float(P.kept[P.y == 0].sum() / P.total[P.y == 0].sum()),
                         auroc_all_windows_all40=roc_auc_score(P.y, P.p_all),
                         # re-audit fix: same-subject baseline, so the change isolates window exclusion
                         auroc_all_windows_same_subjects=roc_auc_score(P.y[v], P.p_all[v]),
                         auroc_clean_windows=roc_auc_score(P.y[v], P.p[v]),
                         delta_window_exclusion_same_subjects=roc_auc_score(P.y[v], P.p[v]) - roc_auc_score(P.y[v], P.p_all[v])))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "T6_turku_window_level_artefact.csv", index=False)
    return T


def main():
    models = [m for m in MODELS if avail(m)]
    print("available:", models)
    pd.set_option("display.width", 250)
    print("\n== T1\n", t1(models).round(3).to_string())
    if REF in models and len(models) > 1:
        print("\n== T2\n", t2(models).round(3).to_string())
    T3, tk = t3(models)
    print("\n== T3 Turku\n", T3.round(3).to_string())
    print("Turku RMS by med:", tk.groupby(["label", "med"]).rms_uv.median().round(1).to_dict())
    T4, age = t4(models)
    print("\n== T4 ds007526\n", T4.round(3).to_string(), "\n", age.round(1).to_string())
    print("\n== T5 post-hoc age check ds007526\n", t5(models).round(3).to_string())
    print("\n== T6 Turku window-level artefact exclusion\n", t6(models).round(3).to_string())


if __name__ == "__main__":
    main()
