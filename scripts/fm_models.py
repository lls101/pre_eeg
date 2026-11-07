"""Pretrained EEG foundation-model wrappers (LaBraM-base, CBraMod) with a common API.

Input to .features(): float tensor (B, 19, n_patch, 200) already scaled to 0.1 mV (uV / 100),
channels in paths.CH19 order. Output: (B, 200) pooled embedding.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from paths import CH19, PROJECT  # noqa: E402

VENDOR = PROJECT / "vendor"
# copied from LaBraM/utils.py (standard_1020) to avoid importing its heavy deps
_LB_1020 = ['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
            'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
            'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
            'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
            'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
            'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
            'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]


class LaBraMEncoder(nn.Module):
    name = "LaBraM"

    def __init__(self, pretrained=True):
        super().__init__()
        sys.path.insert(0, str(VENDOR / "LaBraM"))
        from modeling_finetune import labram_base_patch200_200  # noqa
        self.net = labram_base_patch200_200(num_classes=0, drop_path_rate=0.1, use_mean_pooling=True,
                                            init_values=0.1, qkv_bias=False)
        self.register_buffer("input_chans", torch.tensor([0] + [_LB_1020.index(c.upper()) + 1 for c in CH19]),
                             persistent=False)
        self.load_report = "random-init"
        if pretrained:
            import argparse
            import numpy as np
            safe = [argparse.Namespace, np.core.multiarray.scalar, np.dtype] + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
            with torch.serialization.safe_globals(safe):  # still weights_only: only numpy scalar metadata allow-listed
                ck = torch.load(VENDOR / "LaBraM" / "checkpoints" / "labram-base.pth", map_location="cpu", weights_only=True)
            ck = ck.get("model", ck)
            sd = {k[8:]: v for k, v in ck.items() if k.startswith("student.")}
            own = self.net.state_dict()
            sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            self.load_report = f"loaded {len(sd)} tensors; missing={missing}"
        self.dim = 200

    def features(self, x):
        return self.net.forward_features(x, input_chans=self.input_chans)


class CBraModEncoder(nn.Module):
    name = "CBraMod"

    def __init__(self, pretrained=True):
        super().__init__()
        sys.path.insert(0, str(VENDOR / "CBraMod"))
        from models.cbramod import CBraMod  # noqa
        self.net = CBraMod()
        self.load_report = "random-init"
        if pretrained:
            sd = torch.load(VENDOR / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
                            map_location="cpu", weights_only=True)
            missing, unexpected = self.net.load_state_dict(sd, strict=True), None
            self.load_report = f"loaded strict ({len(sd)} tensors)"
        self.net.proj_out = nn.Identity()
        self.dim = 200

    def features(self, x):
        return self.net(x).mean(dim=(1, 2))  # (B, ch, patch, 200) -> (B, 200)


ENCODERS = {"LaBraM": LaBraMEncoder, "CBraMod": CBraModEncoder}


if __name__ == "__main__":  # smoke test
    for n, E in ENCODERS.items():
        m = E().eval()
        with torch.no_grad():
            y = m.features(torch.randn(2, 19, 4, 200))
        print(n, m.load_report[:200], tuple(y.shape), sum(p.numel() for p in m.parameters()) / 1e6, "M params")
