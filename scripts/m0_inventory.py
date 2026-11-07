"""M0: build a per-recording and per-subject inventory of the three PD rsEEG cohorts.

Reads headers only (no preload). Outputs:
  results/m0/recordings.csv  one row per EEG file
  results/m0/subjects.csv    one row per subject (label, demographics)
  results/m0/events_summary.csv  event values per recording (to locate eyes-open spans)
"""
import json
import re
import sys
from pathlib import Path

import mne
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import CH19, CH_ALIASES, DATASETS, RESULTS  # noqa: E402

mne.set_log_level("ERROR")
OUT = RESULTS / "m0"
OUT.mkdir(parents=True, exist_ok=True)
EXTS = (".bdf", ".set", ".vhdr", ".edf")


def norm_ch(name: str) -> str:
    n = name.strip()
    for k, v in CH_ALIASES.items():
        if n.upper() == k.upper():
            return v
    return n


def read_header(path: Path):
    ext = path.suffix.lower()
    if ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=False)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=False)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=False)
    else:
        raise ValueError(ext)
    return raw


def subjects_table(ds: str, root: Path) -> pd.DataFrame:
    p = pd.read_csv(root / "participants.tsv", sep="\t")
    if ds == "ds002778":
        p["label"] = p["participant_id"].str.contains("pd").map({True: "PD", False: "HC"})
        p = p.rename(columns={"gender": "sex"})
    elif ds == "ds003490":
        p["label"] = p["Group"].map({"PD": "PD", "CTL": "HC"})
    elif ds == "ds004584":
        p["label"] = p["GROUP"].map({"PD": "PD", "Control": "HC"})
        p = p.rename(columns={"AGE": "age", "GENDER": "sex"})
    p["sex"] = p["sex"].astype(str).str[0].str.upper()
    p["dataset"] = ds
    p["uid"] = ds + "_" + p["participant_id"]
    keep = ["uid", "dataset", "participant_id", "label", "age", "sex"]
    extra = [c for c in p.columns if c not in keep]
    return p[keep + extra]


def med_state(ds: str, sub: str, ses: str, subj: pd.DataFrame) -> str:
    if ds == "ds002778":
        return {"on": "ON", "off": "OFF", "hc": "n/a"}.get(ses, "?")
    if ds == "ds003490":
        row = subj.loc[subj.participant_id == sub].iloc[0]
        if row.label == "HC":
            return "n/a"
        m = re.search(r"(\d+)", ses or "")
        return row.get(f"sess{int(m.group(1))}_Med", "?") if m else "?"
    if ds == "ds004584":
        return "ON" if subj.loc[subj.participant_id == sub, "label"].iloc[0] == "PD" else "n/a"
    return "?"


def main():
    rec_rows, ev_rows, subj_all = [], [], []
    for ds, root in DATASETS.items():
        if not (root / "participants.tsv").exists():
            print(f"[skip] {ds}: not found at {root}")
            continue
        subj = subjects_table(ds, root)
        subj_all.append(subj)
        files = sorted(f for f in root.glob("sub-*/**/eeg/*_eeg.*") if f.suffix.lower() in EXTS)
        print(f"{ds}: {len(subj)} subjects, {len(files)} EEG files")
        for f in files:
            sub = re.search(r"(sub-[^_]+)", f.name).group(1)
            ses_m = re.search(r"_ses-([^_]+)", f.name)
            ses = ses_m.group(1) if ses_m else ""
            task = re.search(r"_task-([^_]+)", f.name).group(1)
            row = dict(dataset=ds, participant_id=sub, uid=f"{ds}_{sub}", session=ses, task=task,
                       file=str(f), ext=f.suffix.lower())
            try:
                raw = read_header(f)
                chs = [norm_ch(c) for c in raw.ch_names]
                row.update(sfreq=raw.info["sfreq"], n_ch=len(chs), dur_s=round(raw.times[-1], 1),
                           ch19_missing=";".join(c for c in CH19 if c not in chs),
                           ch_names=";".join(chs))
            except Exception as e:  # keep going, record failure
                row.update(error=repr(e)[:300])
            row["med"] = med_state(ds, sub, ses, subj)
            ev = f.with_name(f.name.rsplit("_eeg", 1)[0] + "_events.tsv")
            if ev.exists():
                e = pd.read_csv(ev, sep="\t")
                col = "value" if "value" in e.columns else ("trial_type" if "trial_type" in e.columns else e.columns[-1])
                vc = e[col].astype(str).value_counts()
                row["n_events"] = len(e)
                txt = e["trial_type"].astype(str) if "trial_type" in e.columns else e[col].astype(str)
                eo = e[txt.str.contains("Eyes Open", case=False)]
                if len(eo):  # ds003490: 1-s markers spanning the instructed eyes-open minute
                    on = eo.onset.sort_values().to_numpy()
                    brk = (pd.Series(on).diff() > 1.5).to_numpy().nonzero()[0]
                    first = on[: brk[0]] if len(brk) else on  # first contiguous block only
                    row.update(eo_start=float(first[0]), eo_end=float(first[-1]) + 1.0,
                               eo_blocks=len(brk) + 1)
                for v, n in vc.items():
                    ev_rows.append(dict(dataset=ds, uid=row["uid"], session=ses, task=task, value=v, count=n,
                                        first_onset=float(e.loc[e[col].astype(str) == v, "onset"].iloc[0])))
            rec_rows.append(row)

    rec = pd.DataFrame(rec_rows)
    # Whole-recording eyes-open assumption for ds002778 / ds004584 (TO VERIFY against source papers)
    whole = rec.dataset.isin(["ds002778", "ds004584"])
    rec.loc[whole, "eo_start"] = 0.0
    rec.loc[whole, "eo_end"] = rec.loc[whole, "dur_s"]
    rec["eo_dur"] = (rec.eo_end - rec.eo_start).round(1)
    rec["med"] = rec["med"].fillna("n/a")
    # Cap / montage variant fingerprint (a within-site shortcut if it tracks diagnosis)
    rec["montage_variant"] = rec.ch_names.fillna("").map(
        lambda s: ("Iz" if "Iz" in s.split(";") else "noIz") + ("+Resp" if "Resp" in s.split(";") else "")
        + ("+XYZ" if "X" in s.split(";") else ""))
    subj = pd.concat(subj_all, ignore_index=True)
    rec = rec.merge(subj[["uid", "label"]], on="uid", how="left")
    rec.to_csv(OUT / "recordings.csv", index=False)
    subj.to_csv(OUT / "subjects.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(OUT / "events_summary.csv", index=False)

    # Console summary
    print("\n== subjects per dataset x label")
    print(subj.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    print("\n== recordings per dataset x task x med")
    print(rec.groupby(["dataset", "task", "med"]).size())
    print("\n== sfreq / n_ch / duration")
    print(rec.groupby("dataset")[["sfreq", "n_ch", "dur_s", "eo_dur"]].describe().T.round(1))
    miss = rec[rec.get("ch19_missing", "").fillna("") != ""] if "ch19_missing" in rec else rec.iloc[0:0]
    print(f"\n== recordings missing any of 19 channels: {len(miss)}")
    if len(miss):
        print(miss[["uid", "session", "ch19_missing"]].to_string())
    if "error" in rec and rec["error"].notna().any():
        print("\n== read errors"); print(rec.loc[rec.error.notna(), ["uid", "file", "error"]].to_string())
    no_eeg = set(subj.uid) - set(rec.uid)
    print(f"\n== subjects without EEG file: {sorted(no_eeg)}")
    print("\n== age by dataset x label")
    print(subj.groupby(["dataset", "label"])["age"].agg(["mean", "std", "min", "max"]).round(1))
    print(subj.groupby(["dataset", "label"])["sex"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
