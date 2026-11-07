"""M1: shared preprocessing -> cache of fixed-length 19-ch eyes-open segments (pre_v1).

Per main-set recording:
  1. load raw, keep EEG channels; ds004584: restore online reference Pz as a zero channel
  2. average reference over all EEG channels, then pick the 19-ch 10-20 set
  3. notch 60 Hz (all sites US), band-pass 0.5-45 Hz (FIR, zero-phase)
  4. crop the eyes-open span; take the FIRST 60 s (equal length across sites)
  5. resample to 200 Hz -> float32 array (19, 12000) in microvolts
Window-level artefact flag (4 s, ptp > 200 uV) is recorded, not applied, so every
downstream model sees the same data and rejection can be ablated.
Output: results/cache/pre_v1/<uid>.npy, results/cache/pre_v1/meta.csv
"""
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from m0_inventory import norm_ch, read_header  # noqa: E402
from paths import CH19, RESULTS  # noqa: E402

mne.set_log_level("ERROR")
CACHE = RESULTS / "cache" / "pre_v1"
CACHE.mkdir(parents=True, exist_ok=True)
SEG_S, SFREQ, WIN_S, PTP_UV = 60.0, 200.0, 4.0, 200.0
NON_EEG = {"VEOG", "HEOG", "X", "Y", "Z", "Resp", "Status"}


def process(row, cache_dir=None) -> dict:
    """cache_dir must be passed explicitly when not writing to pre_v1: joblib workers re-import this
    module, so monkey-patching the module-level CACHE does NOT reach them (bug found 2026-11-07)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE
    out = dict(uid=row.uid, dataset=row.dataset, label=row.label)
    try:
        raw = read_header(Path(row.file))
        raw.rename_channels({c: norm_ch(c) for c in raw.ch_names})
        drop = [c for c in raw.ch_names if c in NON_EEG or c.startswith("EXG")]
        raw.drop_channels(drop)
        raw.set_channel_types({c: "eeg" for c in raw.ch_names})
        raw.load_data()
        if row.dataset == "ds004584" and "Pz" not in raw.ch_names:
            raw = mne.add_reference_channels(raw, "Pz")  # online reference -> zero channel
        raw.set_eeg_reference("average", projection=False)
        missing = [c for c in CH19 if c not in raw.ch_names]
        assert not missing, f"missing {missing}"
        raw.pick(CH19)
        raw.notch_filter(60.0, picks="eeg")
        raw.filter(0.5, 45.0, picks="eeg")
        # skip 2 s at recording onset (filter edge / settling) when the span starts at t=0
        t0 = float(row.eo_start) + (2.0 if float(row.eo_start) == 0.0 else 0.0)
        t1 = min(float(row.eo_end), t0 + SEG_S, raw.times[-1])
        raw.crop(t0, t1, include_tmax=False)
        raw.resample(SFREQ)
        x = (raw.get_data() * 1e6).astype(np.float32)  # V -> uV
        n = int(SEG_S * SFREQ)
        assert x.shape[1] >= n - 2, f"short segment {x.shape[1] / SFREQ:.1f}s"
        x = x[:, :n] if x.shape[1] >= n else np.pad(x, ((0, 0), (0, n - x.shape[1])), mode="edge")
        np.save(cache_dir / f"{row.uid}.npy", x)
        w = int(WIN_S * SFREQ)
        ptp = np.array([np.ptp(x[:, i:i + w], axis=1).max() for i in range(0, n - w + 1, w)])
        out.update(ok=True, n_samples=x.shape[1], rms_uv=float(np.sqrt((x ** 2).mean())),
                   median_ch_std_uv=float(np.median(x.std(axis=1))),
                   n_win=len(ptp), n_win_bad=int((ptp > PTP_UV).sum()), max_ptp_uv=float(ptp.max()))
    except Exception as e:
        out.update(ok=False, error=repr(e)[:300])
    return out


def main():
    ms = pd.read_csv(RESULTS / "m0" / "main_set.csv", keep_default_na=False)
    rows = Parallel(n_jobs=12)(delayed(process)(r) for r in ms.itertuples())
    meta = pd.DataFrame(rows)
    meta.to_csv(CACHE / "meta.csv", index=False)
    print(meta.ok.value_counts())
    if (~meta.ok).any():
        print(meta.loc[~meta.ok, ["uid", "error"]].to_string())
    g = meta[meta.ok].groupby(["dataset", "label"])
    print(g[["rms_uv", "median_ch_std_uv", "max_ptp_uv"]].median().round(1))
    print(g[["n_win_bad"]].agg(["mean", "max"]))


if __name__ == "__main__":
    main()
