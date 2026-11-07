"""M1 R002 (overfit sanity) + R005 (EEGNet from scratch, 3-fold LODO x 3 seeds).

Windows: 4 s (800 samples @200 Hz), stride 2 s -> 29 windows / subject; fixed scaling x/20 uV.
Validation: inner fold 0 of the training sites (subject-disjoint) for early stopping + Youden threshold.
Subject score = mean window logit -> sigmoid.
Usage: python m1_eegnet.py [--overfit]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from m1_common import RESULTS, evaluate, load_meta, load_splits, load_x, summarize, youden_threshold  # noqa

OUT = RESULTS / "m1"
OUT.mkdir(parents=True, exist_ok=True)
WIN, STRIDE, SCALE = 800, 400, 20.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from paths import CH19  # noqa: E402
FRONTAL = ["Fp1", "Fp2", "F7", "F8"]
CH_IDX = list(range(19))  # overridden by --drop-frontal
NORM = {"zscore": False}  # --zscore: per-channel z-score per recording (x10 uV)


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018)."""

    def __init__(self, n_ch=19, n_t=WIN, F1=8, D=2, F2=16, k1=100, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n, 1)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1))).squeeze(1)


def windows(uids):
    X, sid = [], []
    for i, u in enumerate(uids):
        x = load_x(u)[CH_IDX]
        if NORM["zscore"]:
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6) * 10.0
        x = x / SCALE
        for s in range(0, x.shape[1] - WIN + 1, STRIDE):
            X.append(x[:, s:s + WIN]); sid.append(i)
    return torch.tensor(np.stack(X)), np.array(sid)


@torch.no_grad()
def window_logits(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 512].to(DEV)).cpu() for i in range(0, len(X), 512)]).numpy()


@torch.no_grad()
def subject_scores(model, X, sid, n_sub):
    model.eval()
    logits = torch.cat([model(X[i:i + 512].to(DEV)).cpu() for i in range(0, len(X), 512)]).numpy()
    return np.array([logits[sid == i].mean() for i in range(n_sub)])


def train(tr_uids, ytr, va_uids, yva, seed, max_epochs=60, patience=10, lr=1e-3, wd=1e-3, log=False):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, str_ = windows(tr_uids)
    Xva, sva = windows(va_uids) if va_uids is not None else (None, None)
    ywin = torch.tensor(ytr[str_], dtype=torch.float32)
    model = EEGNet(n_ch=len(CH_IDX)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad = -1, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr[b].to(DEV)), ywin[b].to(DEV))
            loss.backward(); opt.step()
        if va_uids is None:
            continue
        auc = roc_auc_score(yva, subject_scores(model, Xva, sva, len(va_uids)))
        if log:
            print(f"  ep{ep:02d} loss={loss.item():.3f} val_auc={auc:.3f}")
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, ep


def overfit_check():
    meta = load_meta()
    sub = meta.groupby("y").head(10)  # 10 PD + 10 HC
    u, y = sub.uid.to_numpy(), sub.y.to_numpy()
    model, _, _ = train(u, y, None, None, seed=0, max_epochs=150)
    model.eval()
    X, sid = windows(u)
    with torch.no_grad():
        p = torch.sigmoid(torch.cat([model(X[i:i + 512].to(DEV)).cpu() for i in range(0, len(X), 512)])).numpy()
    win_acc = ((p >= 0.5) == y[sid]).mean()
    print(f"[R002 overfit] 20 subjects, window train acc = {win_acc:.3f} (expect ~1.0)")
    return win_acc


def main_ensemble(seeds=(0, 1, 2), tag="EEGNet_ens5", permute=False):
    """R016: 5 models per fold (each inner fold as val), mean test logit; threshold = Youden on pooled OOF.
    Saves OOF + test LOGITS (needed by B4). permute=True: training labels permuted within site (R014)."""
    meta = load_meta().set_index("uid")
    y_true = meta.y.copy()
    if permute:
        rng = np.random.default_rng(123)
        meta["y"] = meta.groupby("dataset").y.transform(lambda v: rng.permutation(v.to_numpy()))
    res = []
    for ts, fold in load_splits().items():
        te = fold["test"]
        Xte, ste = windows(te)
        for seed in seeds:
            t0 = time.time()
            test_logits, oof_u, oof_s, win_l = [], [], [], []
            for k, va in enumerate(fold["inner_val"]):
                tr = [u for u in fold["train"] if u not in set(va)]
                model, _, _ = train(np.array(tr), meta.loc[tr, "y"].to_numpy(), np.array(va),
                                    meta.loc[va, "y"].to_numpy(), seed * 10 + k)
                Xva, sva = windows(va)
                oof_s.append(subject_scores(model, Xva, sva, len(va))); oof_u += list(va)
                test_logits.append(subject_scores(model, Xte, ste, len(te)))
                win_l.append(window_logits(model, Xte))  # per-window test logits (window-level QC sensitivity)
            np.savez(OUT / f"testwin_LODO_{tag}_{ts}_s{seed}.npz", uid=np.array(te), sid=ste,
                     logit=np.mean(win_l, axis=0), win=WIN, stride=STRIDE)
            oof_p = 1 / (1 + np.exp(-np.concatenate(oof_s)))
            thr = youden_threshold(meta.loc[oof_u, "y"].to_numpy(), oof_p)
            p = 1 / (1 + np.exp(-np.mean(test_logits, axis=0)))
            r = evaluate(y_true.loc[te].to_numpy(), p, thr)
            pd.DataFrame(dict(uid=oof_u, y=y_true.loc[oof_u].to_numpy(), logit=np.concatenate(oof_s))).to_csv(
                OUT / f"oof_LODO_{tag}_{ts}_s{seed}.csv", index=False)
            pd.DataFrame(dict(uid=te, y=y_true.loc[te].to_numpy(), logit=np.mean(test_logits, axis=0))).to_csv(
                OUT / f"testlogit_LODO_{tag}_{ts}_s{seed}.csv", index=False)
            res.append(dict(protocol="LODO", model=tag, test_site=ts, seed=seed,
                            oof_auc=float(roc_auc_score(meta.loc[oof_u, "y"], oof_p)), **r))
            pd.DataFrame(dict(uid=te, y=y_true.loc[te].to_numpy(), p=p)).to_csv(
                OUT / f"pred_LODO_{tag}_{ts}_s{seed}.csv", index=False)
            print(f"test={ts} seed={seed} oof_auc={res[-1]['oof_auc']:.3f} test_auc={r['auroc']:.3f} "
                  f"ba={r['ba_train_thr']:.3f} ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(res)
    R.to_csv(OUT / f"results_{tag}.csv", index=False)
    per_site, macro = summarize(R)
    print(per_site.to_string()); print(macro.to_string())
    print("seed SD of site-macro AUROC:", R.groupby("seed").auroc.mean().std().round(3))


def main(seeds=(0, 1, 2)):
    meta = load_meta().set_index("uid")
    res = []
    for ts, fold in load_splits().items():
        va = fold["inner_val"][0]
        tr = [u for u in fold["train"] if u not in set(va)]
        te = fold["test"]
        for seed in seeds:
            t0 = time.time()
            model, vauc, ep = train(np.array(tr), meta.loc[tr, "y"].to_numpy(), np.array(va), meta.loc[va, "y"].to_numpy(), seed)
            Xva, sva = windows(va)
            thr = youden_threshold(meta.loc[va, "y"], 1 / (1 + np.exp(-subject_scores(model, Xva, sva, len(va)))))
            Xte, ste = windows(te)
            p = 1 / (1 + np.exp(-subject_scores(model, Xte, ste, len(te))))
            r = evaluate(meta.loc[te, "y"].to_numpy(), p, thr)
            res.append(dict(protocol="LODO", model="EEGNet", test_site=ts, seed=seed, val_auc=vauc, epochs=ep + 1, **r))
            pd.DataFrame(dict(uid=te, y=meta.loc[te, "y"].to_numpy(), p=p)).to_csv(
                OUT / f"pred_LODO_EEGNet_{ts}_s{seed}.csv", index=False)
            print(f"test={ts} seed={seed} val_auc={vauc:.3f} test_auc={r['auroc']:.3f} "
                  f"ba={r['ba_train_thr']:.3f} ep={ep + 1} ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(res)
    R.to_csv(OUT / "results_eegnet.csv", index=False)
    per_site, macro = summarize(R)
    print(per_site.to_string()); print(macro.to_string())
    print("seed SD of site-macro AUROC:", R.groupby("seed").auroc.mean().std().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--drop-frontal", action="store_true")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    tag = "EEGNet_ens5"
    if a.drop_frontal:
        CH_IDX[:] = [i for i, c in enumerate(CH19) if c not in FRONTAL]; tag += "_noFrontal"
    if a.permute:
        tag += "_perm"
    if a.zscore:
        NORM["zscore"] = True; tag += "_zscore"
    if a.overfit:
        overfit_check()
    elif a.ensemble:
        main_ensemble(seeds=tuple(a.seeds), tag=tag, permute=a.permute)
    else:
        main()
