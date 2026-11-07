"""Dataset roots and project paths. Override with environment variables:
PD_PROJECT (output root), PD_DS002778, PD_DS003490, PD_DS004584 (raw BIDS roots)."""
import os
from pathlib import Path

PROJECT = Path(os.environ.get("PD_PROJECT", Path(__file__).resolve().parents[1]))
RESULTS = PROJECT / "results"
DATASETS = {k: Path(os.environ.get("PD_" + k.upper(), f"data/{k}")) for k in ("ds002778", "ds003490", "ds004584")}
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
        "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
CH_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
