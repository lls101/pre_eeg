"""M4 idea 2: does lag-insensitive (zero-lag) vs lagged connectivity separate SITE from DISEASE?

Per subject (60 s, 19 ch, pre_v1): Hann-windowed 4 s segments, 50 % overlap -> cross-spectra S_ij(f, segment).
Metrics per band (delta, theta, alpha, beta, gamma) x 171 pairs:
  coh   : |<S_ij>|^2 / (<S_ii><S_jj>)                  (zero-lag + lagged)
  icoh  : |Im<S_ij>| / sqrt(<S_ii><S_jj>)              (lagged only)
  pli   : |<sign Im S_ij>|                             (lagged only)
  wpli  : |<Im S_ij>| / <|Im S_ij|>                    (lagged only)
  coh_csd: coh after surface Laplacian (CSD; reduces volume conduction / reference effects)
Evaluations: HC-only 3-class site probe, within-site PD probe, and fold-safe 3-site LODO LR.
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, SFREQ, evaluate, load_meta, load_splits, load_x, summarize  # noqa: E402
from m1_features import fit_lr, impute  # noqa: E402
from m2_embed_probe import pd_within_probe, site_probe  # noqa: E402
from paths import CH19  # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
OUT = RESULTS / "m4"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
I, J = np.array(list(combinations(range(19), 2))).T
SEG, HOP = 800, 400


def csd(x):
    info = mne.create_info(CH19, SFREQ, "eeg")
    raw = mne.io.RawArray(x * 1e-6, info)
    raw.set_montage("standard_1020")
    return mne.preprocessing.compute_current_source_density(raw).get_data().astype(np.float32) * 1e6


def conn(x):
    win = np.hanning(SEG)
    segs = np.stack([x[:, s:s + SEG] * win for s in range(0, x.shape[1] - SEG + 1, HOP)])  # (n_seg, 19, SEG)
    X = np.fft.rfft(segs, axis=-1)
    f = np.fft.rfftfreq(SEG, 1 / SFREQ)
    Sij = X[:, I, :] * np.conj(X[:, J, :])  # (n_seg, 171, n_f)
    Pii, Pjj = (np.abs(X[:, I, :]) ** 2).mean(0), (np.abs(X[:, J, :]) ** 2).mean(0)
    mS = Sij.mean(0)
    m = {"coh": np.abs(mS) ** 2 / (Pii * Pjj),
         "icoh": np.abs(mS.imag) / np.sqrt(Pii * Pjj),
         "pli": np.abs(np.sign(Sij.imag).mean(0)),
         "wpli": np.abs(Sij.imag.mean(0)) / (np.abs(Sij.imag).mean(0) + 1e-20)}
    return {k: np.stack([v[:, (f >= a) & (f < b)].mean(-1) for a, b in BANDS.values()], 1).ravel() for k, v in m.items()}


def subject(u):
    x = load_x(u)
    d = conn(x)
    d["coh_csd"] = conn(csd(x))["coh"]
    d["icoh_csd"] = conn(csd(x))["icoh"]
    return d


def lodo(meta, Z, name):
    y = meta.y.to_numpy(); out = []
    for ts, fold in load_splits().items():
        tri = meta.uid.isin(fold["train"]).to_numpy(); tei = meta.uid.isin(fold["test"]).to_numpy()
        strata = (meta.dataset[tri] + "_" + meta.label[tri]).to_numpy()
        m, C, thr = fit_lr(Z[tri], y[tri], meta.uid[tri].to_numpy(), strata)
        p = m.predict_proba(impute(Z[tri], Z[tei])[1])[:, 1]
        out.append(dict(protocol="LODO", model=name, test_site=ts, **evaluate(y[tei], p, thr)))
        pd.DataFrame(dict(uid=meta.uid[tei].values, y=y[tei], p=p)).to_csv(OUT / f"pred_LODO_{name}_{ts}.csv", index=False)
    return out


def main():
    meta = load_meta()
    path = OUT / "conn_features.npz"
    if path.exists():
        F = dict(np.load(path))
    else:
        rows = Parallel(n_jobs=12)(delayed(subject)(u) for u in meta.uid)
        F = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez(path, **F)
    site = meta.dataset.to_numpy(); y = meta.y.to_numpy(); hc = y == 0
    diag, res = [], []
    for k, Z in F.items():
        Z = np.nan_to_num(Z)
        diag.append(dict(metric=k, site_HC_only=site_probe(Z[hc], site[hc]), pd_within_site=pd_within_probe(Z, y, site)))
        res += lodo(meta, Z, f"LR_{k}")
        print("done", k, flush=True)
    D = pd.DataFrame(diag); D["pd_minus_site"] = D.pd_within_site - D.site_HC_only
    D.to_csv(OUT / "conn_diagnosis.csv", index=False)
    print("\n== connectivity diagnosis\n", D.round(3).to_string())
    R = pd.DataFrame(res); R.to_csv(OUT / "conn_lodo.csv", index=False)
    per_site, macro = summarize(R)
    print("\n== LODO\n", per_site.to_string(), "\n", macro.to_string())


if __name__ == "__main__":
    main()
