"""M2: frozen FM embeddings -> (a) R006/R007 LODO linear probe, (b) R008 site/disease decodability diagnosis.

Encoders: LaBraM, CBraMod (pretrained) + random-init versions (control).
Input transforms (phase-preserving, per subject x channel on the 60 s segment, FOOOF fixed fit 2-40 Hz):
  none     : pre_v1 as is
  aperref  : aperiodic line moved to a common reference (per-channel median offset/exponent of ALL subjects;
             diagnosis only - M3 will fit the reference on training-site HC only)
  flat     : aperiodic slope removed entirely (Bindra-style flattening; level kept at the subject's mean)
  sham     : the aperref gain of a RANDOM OTHER subject (same magnitude of spectral change, uninformative)
Subject embedding = mean over 29 windows (4 s, stride 2 s), input scaled uV/100.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fm_models import ENCODERS  # noqa: E402
from m1_common import RESULTS, SFREQ, evaluate, load_meta, load_splits, load_x, summarize  # noqa: E402
from m1_features import fit_lr, impute  # noqa: E402
from paths import CH19  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "m2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
WIN, STRIDE = 800, 400
TRANSFORMS = ["none", "aperref", "flat", "sham"]


# ---------------------------------------------------------------- aperiodic transforms
def aperiodic_table(meta):
    F = pd.read_csv(RESULTS / "m1" / "features_v1.csv").set_index("uid").loc[meta.uid]
    off = F[[f"ap_offset_{c}" for c in CH19]].to_numpy()
    exp = F[[f"ap_exp_{c}" for c in CH19]].to_numpy()
    return off, exp  # (n_sub, 19) each, log10-power line: L(f) = off - exp*log10 f


def apply_gain(x, log_gain_fn):
    """x (19, T); log_gain_fn(freqs)->(19, n_f) log10 POWER gain; amplitude gain = 10**(lg/2)."""
    X = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(x.shape[1], 1 / SFREQ)
    fc = np.clip(f, 2.0, 40.0)
    X = X * 10 ** (log_gain_fn(np.log10(fc)) / 2)
    return np.fft.irfft(X, n=x.shape[1], axis=1).astype(np.float32)


def transformed(i, uid, kind, off, exp, ref_off, ref_exp, perm):
    x = load_x(uid)
    if kind == "none":
        return x
    if kind == "aperref":
        return apply_gain(x, lambda lf: (ref_off - ref_exp * lf[None, :].T).T - (off[i] - exp[i] * lf[None, :].T).T)
    if kind == "flat":
        lf0 = np.log10(np.linspace(2, 40, 200))
        mean_level = (off[i][:, None] - exp[i][:, None] * lf0[None]).mean(1)
        return apply_gain(x, lambda lf: mean_level[:, None] - (off[i][:, None] - exp[i][:, None] * lf[None]))
    if kind == "sham":
        j = perm[i]
        return apply_gain(x, lambda lf: (ref_off[:, None] - ref_exp[:, None] * lf[None])
                          - (off[j][:, None] - exp[j][:, None] * lf[None]))
    raise ValueError(kind)


# ---------------------------------------------------------------- embeddings
@torch.no_grad()
def embed_all(enc, meta, kind, off, exp, ref_off, ref_exp, perm):
    Z = np.zeros((len(meta), enc.dim), np.float32)
    for i, u in enumerate(meta.uid):
        x = transformed(i, u, kind, off, exp, ref_off, ref_exp, perm) / 100.0
        w = np.stack([x[:, s:s + WIN] for s in range(0, x.shape[1] - WIN + 1, STRIDE)])  # (29,19,800)
        w = torch.tensor(w).reshape(len(w), 19, WIN // 200, 200).to(DEV)
        Z[i] = enc.features(w).mean(0).cpu().numpy()
    return Z


def get_embeddings(meta):
    path = OUT / "embeddings.npz"
    if path.exists():
        return dict(np.load(path))
    off, exp = aperiodic_table(meta)
    ref_off, ref_exp = np.median(off, 0), np.median(exp, 0)
    perm = np.random.default_rng(0).permutation(len(meta))
    perm = np.where(perm == np.arange(len(meta)), np.roll(perm, 1), perm)
    store = {}
    for name, E in ENCODERS.items():
        for pre in (True, False):
            torch.manual_seed(0)
            enc = E(pretrained=pre).to(DEV).eval()
            tag = name if pre else f"{name}-rand"
            for kind in (TRANSFORMS if pre else ["none"]):
                store[f"{tag}|{kind}"] = embed_all(enc, meta, kind, off, exp, ref_off, ref_exp, perm)
                print("embedded", tag, kind, store[f"{tag}|{kind}"].shape)
            del enc; torch.cuda.empty_cache()
    np.savez(path, uid=meta.uid.to_numpy().astype(str), **store)
    return store


# ---------------------------------------------------------------- probes
def lr():
    # audit fix: fold-internal median imputation (handcrafted alpha-CF has NaNs); no-op for complete embeddings
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


def site_probe(Z, site, reps=5):
    s = []
    for r in range(reps):
        p = np.zeros((len(Z), len(np.unique(site))))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Z, site):
            m = lr().fit(Z[tr], site[tr]); p[te] = m.predict_proba(Z[te])
        s.append(roc_auc_score(site, p, multi_class="ovr", average="macro"))
    return float(np.mean(s))


def pd_within_probe(Z, y, site, reps=5):
    s = []
    strata = np.char.add(site.astype(str), y.astype(str))
    for r in range(reps):
        p = np.zeros(len(Z))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Z, strata):
            p[te] = lr().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        s.append(np.mean([roc_auc_score(y[site == k], p[site == k]) for k in np.unique(site)]))
    return float(np.mean(s))


def main():
    meta = load_meta()
    store = get_embeddings(meta)
    F = pd.read_csv(RESULTS / "m1" / "features_v1.csv").set_index("uid").loc[meta.uid]
    store["Handcrafted|none"] = F.drop(columns=["fooof_r2_mean"]).to_numpy(float)  # audit fix: NaN kept; imputed inside folds
    site = meta.dataset.to_numpy(); y = meta.y.to_numpy(); hc = y == 0

    # ---- R008 diagnosis
    rows = []
    for key, Z in store.items():
        if key == "uid":
            continue
        enc, kind = key.split("|")
        rows.append(dict(encoder=enc, transform=kind,
                         site_HC_only=site_probe(Z[hc], site[hc]),
                         site_all=site_probe(Z, site),
                         pd_within_site=pd_within_probe(Z, y, site)))
    D = pd.DataFrame(rows)
    D["site_minus_pd"] = D.site_HC_only - D.pd_within_site
    D.round(3).to_csv(OUT / "diagnosis.csv", index=False)
    print("\n== R008 decodability (AUROC; site = 3-class macro OvR)\n", D.round(3).to_string())

    # ---- R006/R007 LODO linear probe (transform none)
    res = []
    for key in [k for k in store if k.endswith("|none")]:
        Z = store[key]
        for ts, fold in load_splits().items():
            tri = meta.uid.isin(fold["train"]).to_numpy(); tei = meta.uid.isin(fold["test"]).to_numpy()
            strata = (meta.dataset[tri] + "_" + meta.label[tri]).to_numpy()
            m, C, thr = fit_lr(Z[tri], y[tri], meta.uid[tri].to_numpy(), strata)
            p = m.predict_proba(impute(Z[tri], Z[tei])[1])[:, 1]
            res.append(dict(protocol="LODO", model=f"probe_{key.split('|')[0]}", test_site=ts, C=C, **evaluate(y[tei], p, thr)))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "results_probe.csv", index=False)
    per_site, macro = summarize(R)
    print("\n== R006/R007 LODO linear probe\n", per_site.to_string(), "\n", macro.to_string())


if __name__ == "__main__":
    main()
