# EEG foundation models vs an ensembled CNN for unseen-site PD detection (five-cohort LODO)

Code for the ICBEA 2027 paper. No data or model weights are redistributed.

## Data (public)

OpenNeuro ds002778, ds003490, ds004584, ds007526; OSF pehj9 (Turku, `scripts/dl_osf_turku.py`).
Set PD_DS002778, PD_DS003490, PD_DS004584, PD_DS007526, PD_TURKU to the raw roots.

## Third-party models (not included)

Clone into `vendor/` and download the official pretrained weights:

- LaBraM: https://github.com/935963004/LaBraM (commit c431221e6cfd23dbfa9950e0180682fb322b0548), `vendor/LaBraM/checkpoints/labram-base.pth`
- CBraMod: https://github.com/wjq-learning/CBraMod (commit b9e961003214326972c567eff390e75b0287e32a), `vendor/CBraMod/pretrained_weights/pretrained_weights.pth`

## Pipeline

1. `python scripts/m1_preprocess.py` (3 US cohorts) and `python scripts/m1_preprocess_5site.py` (Tel Aviv, Turku, combined cache)
2. `python scripts/m0_splits_5site.py` (or use `splits/splits_5site.json`)
3. `bash scripts/run_5site.sh cpu` and `bash scripts/run_5site.sh gpu` (PD_SITES=5)
4. `python scripts/m6_5site_analysis.py`, `m6b_iowa_severity.py`, `m6c_telaviv_severity_ledd.py`
