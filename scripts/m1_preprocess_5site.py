"""5-site extension of M0/M1: inventory + pre_v1 preprocessing for ds007526 (Tel Aviv) and Turku (Railo, OSF pehj9),
then a combined cache results/cache/pre_v1_5site with the unchanged 230 pre_v1 arrays copied in.

The pipeline is identical to m1_preprocess.py (avg ref -> 19 ch -> notch -> 0.5-45 Hz FIR -> 60 s -> 200 Hz, uV),
with these documented per-site differences:
  ds007526: EGI. VREF (all-zero; ASSUMED to be the vertex reference, sidecar EEGReference = n/a) is renamed Cz; EOG1-4 (typed EEG) dropped. 50 Hz notch.
            Recordings contain EEGLAB 'boundary' / 'break cnt' discontinuities -> use the FIRST contiguous span that
            holds 2 s skip + 60 s; record which span was used.
  turku:    64 ch 10-10, HEOG/VEOG dropped. FCz is absent from the file (ASSUMED online reference; not documented) ->
            restored as a zero channel before average reference, as for Pz in ds004584. 50 Hz notch. Eyes-open file.
  The 0.5-45 Hz band-pass attenuates 50 Hz by only ~4.5 dB with MNE defaults, so the 50 Hz notch is required.
Labels: ds007526 participants.tsv `group`; turku `PD group` column of InfoAboutParticipants.xlsx (checked against ID
parity: odd = PD). Medication: ds007526 'usual' (LEDD only, no ON/OFF); turku 'ON' or 'OFF_break' from `medicationbreak`.
Outputs: results/m0/recordings_new.csv, results/m0/main_set_5site.csv, results/cache/pre_v1_new/*.npy + meta.csv,
         results/cache/pre_v1_5site/*.npy + meta.csv
"""
import os
import shutil
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
DS = Path(os.environ.get("PD_DS007526", "data/ds007526"))
TK = Path(r"F:\DataSets\OpenSource\turku_railo")
NEW = RESULTS / "cache" / "pre_v1_new"
FIVE = RESULTS / "cache" / "pre_v1_5site"
OLD = RESULTS / "cache" / "pre_v1"
SEG_S, SFREQ, WIN_S, PTP_UV, SKIP_S = 60.0, 200.0, 4.0, 200.0, 2.0
DROP = {"VEOG", "HEOG", "EOG1", "EOG2", "EOG3", "EOG4"}
DISC = {"boundary", "break cnt"}


def inventory() -> pd.DataFrame:
    rows = []
    p = pd.read_csv(DS / "participants.tsv", sep="\t")
    for r in p.itertuples():
        f = DS / r.participant_id / "eeg" / f"{r.participant_id}_task-rest_eeg.set"
        rows.append(dict(dataset="ds007526", participant_id=r.participant_id, uid=f"ds007526_{r.participant_id}",
                         task="rest", file=str(f), label=r.group, med="usual" if r.group == "PD" else "n/a",
                         age=r.age, sex=str(r.sex)[0].upper(), updrs3=r.updrs_part_iii, moca=r.moca, ledd=r.ledd))
    x = pd.read_excel(next(TK.glob("*.xlsx")), sheet_name="Sheet1")
    x = x[x["id"].notna() & x["PD group"].notna()]
    for r in x.itertuples(index=False):
        sid = int(r[0])
        lab = "PD" if int(r[1]) == 1 else "HC"
        assert (sid % 2 == 1) == (lab == "PD"), f"turku id parity mismatch {sid}"
        g = dict(zip(x.columns, r))
        med = "n/a" if lab == "HC" else ("OFF_break" if g["medicationbreak"] == 1 else "ON")
        rows.append(dict(dataset="turku", participant_id=f"sub-{sid:02d}", uid=f"turku_sub-{sid:02d}", task="rest",
                         file=str(TK / "data" / f"{sid:02d}_eyesOpen.set"), label=lab, med=med, age=g["age"],
                         sex=str(g["sex"])[0].upper(), updrs3=g["MDS-UPDRS"], mmse=g["MMSE"], ledd=g["LEDD"]))
    rec = pd.DataFrame(rows)
    rec["exists"] = rec.file.map(lambda s: Path(s).exists())
    return rec


def contiguous_start(raw) -> tuple:
    """First span [a, b) free of discontinuities that holds SKIP_S + SEG_S seconds; returns (t0, n_spans, a, b)."""
    a = raw.annotations
    # audit fix (5-site audit G2): every discontinuity after t=0 is a cut (was: only onset > 0.5 s, which let a
    # 'break cnt' at 0.44 s fall 1.56 s before the segment). Only onsets are used: some 'break cnt' durations span
    # almost the whole file, so durations are not treated as excluded intervals.
    cuts = sorted({float(o) for o, d in zip(a.onset, a.description) if d in DISC and o > 1e-6})
    edges = [0.0] + cuts + [raw.times[-1]]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo >= SKIP_S + SEG_S:
            return lo + SKIP_S, len(edges) - 1, lo, hi
    raise AssertionError(f"no contiguous {SKIP_S + SEG_S:.0f}s span; spans={np.diff(edges).round(1).tolist()}")


def process(row) -> dict:
    out = dict(uid=row.uid, dataset=row.dataset, label=row.label)
    try:
        raw = read_header(Path(row.file))
        raw.rename_channels({c: norm_ch(c) for c in raw.ch_names})
        if "VREF" in raw.ch_names:
            raw.rename_channels({"VREF": "Cz"})
        raw.drop_channels([c for c in raw.ch_names if c in DROP])
        raw.set_channel_types({c: "eeg" for c in raw.ch_names})
        raw.load_data()
        if row.dataset == "ds007526":
            assert np.abs(raw.get_data(picks=["Cz"])).max() == 0.0, "VREF not all-zero"
        if row.dataset == "turku" and "FCz" not in raw.ch_names:
            raw = mne.add_reference_channels(raw, "FCz")
        t0, n_spans, lo, hi = contiguous_start(raw)
        # audit fix (G2): filter only within the continuous span (MNE does not split filtering at these
        # annotations), so no filter response crosses a discontinuity. Same as the old sites, whose recordings are
        # continuous: filter the continuous segment, then skip 2 s from its start.
        raw.crop(lo, hi, include_tmax=(hi >= raw.times[-1]))
        raw.set_annotations(None)
        t0 -= lo
        raw.set_eeg_reference("average", projection=False)
        missing = [c for c in CH19 if c not in raw.ch_names]
        assert not missing, f"missing {missing}"
        raw.pick(CH19)
        raw.notch_filter(50.0, picks="eeg")
        raw.filter(0.5, 45.0, picks="eeg")
        raw.crop(t0, t0 + SEG_S, include_tmax=False)
        raw.resample(SFREQ)
        x = (raw.get_data() * 1e6).astype(np.float32)
        n = int(SEG_S * SFREQ)
        assert x.shape[1] >= n - 2, f"short segment {x.shape[1] / SFREQ:.1f}s"
        x = x[:, :n] if x.shape[1] >= n else np.pad(x, ((0, 0), (0, n - x.shape[1])), mode="edge")
        np.save(NEW / f"{row.uid}.npy", x)
        w = int(WIN_S * SFREQ)
        ptp = np.array([np.ptp(x[:, i:i + w], axis=1).max() for i in range(0, n - w + 1, w)])
        out.update(ok=True, n_samples=x.shape[1], seg_t0=t0 + lo, span_lo=lo, span_hi=hi, n_spans=n_spans, rms_uv=float(np.sqrt((x ** 2).mean())),
                   median_ch_std_uv=float(np.median(x.std(axis=1))), n_win=len(ptp),
                   n_win_bad=int((ptp > PTP_UV).sum()), max_ptp_uv=float(ptp.max()))
    except Exception as e:  # noqa: BLE001
        out.update(ok=False, error=repr(e)[:300])
    return out


def main():
    NEW.mkdir(parents=True, exist_ok=True)
    FIVE.mkdir(parents=True, exist_ok=True)
    rec = inventory()
    rec.to_csv(RESULTS / "m0" / "recordings_new.csv", index=False)
    print(rec.groupby(["dataset", "label", "med"]).size(), "\nmissing files:", rec.loc[~rec.exists, "uid"].tolist())
    rec = rec[rec.exists]
    meta = pd.DataFrame(Parallel(n_jobs=12)(delayed(process)(r) for r in rec.itertuples()))
    meta.to_csv(NEW / "meta.csv", index=False)
    print(meta.ok.value_counts().to_dict())
    if (~meta.ok).any():
        print(meta.loc[~meta.ok, ["uid", "error"]].to_string())
    g = meta[meta.ok].groupby(["dataset", "label"])
    print(g[["rms_uv", "median_ch_std_uv", "max_ptp_uv", "n_win_bad"]].median().round(1))
    print("segments not starting at 2 s:", meta.loc[meta.ok & (meta.seg_t0 != SKIP_S), ["uid", "seg_t0", "n_spans"]].to_string())

    # combined 5-site cache: old pre_v1 arrays copied unchanged + new arrays
    old = pd.read_csv(OLD / "meta.csv")
    for u in old.loc[old.ok, "uid"]:
        shutil.copy2(OLD / f"{u}.npy", FIVE / f"{u}.npy")
    for u in meta.loc[meta.ok, "uid"]:
        shutil.copy2(NEW / f"{u}.npy", FIVE / f"{u}.npy")
    pd.concat([old, meta], ignore_index=True).to_csv(FIVE / "meta.csv", index=False)

    ms_old = pd.read_csv(RESULTS / "m0" / "main_set.csv", keep_default_na=False)
    ms_new = rec[rec.uid.isin(meta.loc[meta.ok, "uid"])]
    ms = pd.concat([ms_old, ms_new[["dataset", "participant_id", "uid", "task", "file", "med", "label"]]],
                   ignore_index=True)
    ms.to_csv(RESULTS / "m0" / "main_set_5site.csv", index=False)
    print(ms.groupby(["dataset", "label"]).size().unstack())


if __name__ == "__main__":
    main()
