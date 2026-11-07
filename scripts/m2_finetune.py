"""M2 R006/R007 full fine-tuning: FM encoder + linear head, 3-fold LODO, inner 5-fold ensemble (same as R016).

Windows 4 s (4 patches x 200) stride 2 s, input uV/100. AdamW: backbone lr 3e-5, head lr 1e-3, wd 0.05.
Early stopping on inner-val subject AUROC (patience 4, max 15 epochs). Threshold = Youden on pooled OOF.
Usage: python m2_finetune.py LaBraM|CBraMod [seed ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from fm_models import ENCODERS  # noqa: E402
from m1_common import RESULTS, evaluate, load_meta, load_splits, load_x, summarize, youden_threshold  # noqa: E402

OUT = RESULTS / "m2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
WIN, STRIDE = 800, 400
NORM = {"zscore": False}
SUF = lambda: "_zscore" if NORM["zscore"] else ""  # noqa: E731


class Clf(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc, self.head = enc, nn.Linear(enc.dim, 1)

    def forward(self, x):
        return self.head(self.enc.features(x)).squeeze(1)


def windows(uids):
    X, sid = [], []
    for i, u in enumerate(uids):
        x = load_x(u)
        if NORM["zscore"]:
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6) * 10.0
        x = x / 100.0
        for s in range(0, x.shape[1] - WIN + 1, STRIDE):
            X.append(x[:, s:s + WIN]); sid.append(i)
    return torch.tensor(np.stack(X)).reshape(-1, 19, WIN // 200, 200), np.array(sid)


@torch.no_grad()
def win_logits(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 256].to(DEV)).float().cpu() for i in range(0, len(X), 256)]).numpy()


def scores(model, X, sid, n, lg=None):
    lg = win_logits(model, X) if lg is None else lg
    return np.array([lg[sid == i].mean() for i in range(n)])


def train(name, tr, ytr, va, yva, seed, max_epochs=15, patience=4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Clf(ENCODERS[name](pretrained=True)).to(DEV)
    Xtr, s_tr = windows(tr); Xva, s_va = windows(va)
    yw = torch.tensor(ytr[s_tr], dtype=torch.float32)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 3e-5},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), device=DEV))
    scaler = torch.amp.GradScaler()
    best, state, bad = -1, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(Xtr[b].to(DEV)), yw[b].to(DEV))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        auc = roc_auc_score(yva, scores(model, Xva, s_va, len(va)))
        if auc > best:
            best, bad, state = auc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, ep + 1


def main(name, seeds):
    meta = load_meta().set_index("uid")
    res = []
    for ts, fold in load_splits().items():
        te = fold["test"]; Xte, s_te = windows(te)
        for seed in seeds:
            t0 = time.time()
            tl, oof_u, oof_s, eps, wl = [], [], [], [], []
            for k, va in enumerate(fold["inner_val"]):
                tr = [u for u in fold["train"] if u not in set(va)]
                model, ep = train(name, np.array(tr), meta.loc[tr, "y"].to_numpy(), np.array(va),
                                  meta.loc[va, "y"].to_numpy(), seed * 10 + k)
                Xva, s_va = windows(va)
                oof_s.append(scores(model, Xva, s_va, len(va))); oof_u += list(va); eps.append(ep)
                lg = win_logits(model, Xte); wl.append(lg)  # per-window test logits (window-level QC sensitivity)
                tl.append(scores(model, Xte, s_te, len(te), lg))
                del model; torch.cuda.empty_cache()
            np.savez(OUT / f"testwin_LODO_FT_{name}{SUF()}_{ts}_s{seed}.npz", uid=np.array(te), sid=s_te,
                     logit=np.mean(wl, 0), win=WIN, stride=STRIDE)
            oof_p = 1 / (1 + np.exp(-np.concatenate(oof_s)))
            thr = youden_threshold(meta.loc[oof_u, "y"].to_numpy(), oof_p)
            p = 1 / (1 + np.exp(-np.mean(tl, 0)))
            r = evaluate(meta.loc[te, "y"].to_numpy(), p, thr)
            res.append(dict(protocol="LODO", model=f"FT_{name}" + ("_zscore" if NORM["zscore"] else ""), test_site=ts, seed=seed, epochs=str(eps),
                            oof_auc=float(roc_auc_score(meta.loc[oof_u, "y"], oof_p)), **r))
            pd.DataFrame(dict(uid=te, y=meta.loc[te, "y"].to_numpy(), p=p)).to_csv(
                OUT / f"pred_LODO_FT_{name}{SUF()}_{ts}_s{seed}.csv", index=False)
            pd.DataFrame(dict(uid=oof_u, y=meta.loc[oof_u, "y"].to_numpy(), logit=np.concatenate(oof_s))).to_csv(
                OUT / f"oof_LODO_FT_{name}{SUF()}_{ts}_s{seed}.csv", index=False)
            pd.DataFrame(dict(uid=te, y=meta.loc[te, "y"].to_numpy(), logit=np.mean(tl, 0),
                              **{f"logit_m{k}": tl[k] for k in range(len(tl))})).to_csv(
                OUT / f"testlogit_LODO_FT_{name}{SUF()}_{ts}_s{seed}.csv", index=False)
            print(f"[{name}] test={ts} seed={seed} oof={res[-1]['oof_auc']:.3f} auc={r['auroc']:.3f} "
                  f"ba={r['ba_train_thr']:.3f} eps={eps} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(res).to_csv(OUT / f"results_FT_{name}{SUF()}.csv", index=False)
    per_site, macro = summarize(pd.DataFrame(res))
    print(per_site.to_string()); print(macro.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--zscore" in args:
        NORM["zscore"] = True; args.remove("--zscore")
    main(args[0], [int(s) for s in args[1:]] or [0])
