"""Download OSF node pehj9 (Railo, Turku PD resting-state EEG) recursively.

Usage: python dl_osf_turku.py OUT_DIR [--list]
Skips files whose local size already matches; writes osf_manifest.tsv (path, size, sha256 from OSF).
"""
import sys
from pathlib import Path

import requests

ROOT = "https://api.osf.io/v2/nodes/pehj9/files/osfstorage/"
S = requests.Session()
S.headers["User-Agent"] = "pd-eeg-research-downloader/1.0"


def walk(url, prefix=""):
    while url:
        j = S.get(url, timeout=60).json()
        for it in j["data"]:
            a = it["attributes"]
            if a["kind"] == "folder":
                yield from walk(it["relationships"]["files"]["links"]["related"]["href"], prefix + a["name"] + "/")
            else:
                yield prefix + a["name"], a["size"], it["links"]["download"], (a.get("extra") or {}).get("hashes", {}).get("sha256", "")
        url = j["links"].get("next")


def main():
    out = Path(sys.argv[1])
    files = list(walk(ROOT))
    if "--raw-only" in sys.argv:  # our pipeline preprocesses from raw; skip the PREP copies
        files = [f for f in files if "_PREP_" not in f[0]]
    print(f"{len(files)} files, {sum(f[1] or 0 for f in files) / 1e9:.2f} GB")
    if "--list" in sys.argv:
        for f in files[:40]:
            print(f[0], f[1])
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "osf_manifest.tsv", "w", encoding="utf-8") as m:
        m.write("path\tsize\tsha256\n")
        for rel, size, link, sha in files:
            m.write(f"{rel}\t{size}\t{sha}\n")
    for i, (rel, size, link, _) in enumerate(files):
        p = out / rel
        if p.exists() and p.stat().st_size == size:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with S.get(link, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(p)
        print(f"[{i + 1}/{len(files)}] {rel}", flush=True)
    print("DONE_TURKU")


if __name__ == "__main__":
    main()
