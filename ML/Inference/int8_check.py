"""
Simulated 8-bit version of the final BirdTCN, scored on the clean 3 s test (final model: float 0.901, int8 0.903).

What it simulates (the plan for the chip, see Inference README):
  - batch norm folded into the conv weights and biases
  - weights rounded to 8 bits, one scale per output channel
  - every feature map rounded to 8 bits, with its range measured on 6,000 random train clips
This is a Python simulation, not the C code. Check again with golden vectors after export.

Run (from the repo root):
  python ML/Inference/int8_check.py                      # weights in ML/Model/weights
  python ML/Inference/int8_check.py --weights <run dir>  # any run folder with seed*.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ML" / "Training"))
import train as T                                   # noqa: E402
from model import BirdTCN                           # noqa: E402

CALIB_CLIPS = 6000


def fold(seq):
    """conv + batch norm -> one conv with bias."""
    conv, bn = seq[0], seq[1]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    new = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                    conv.groups, bias=True).to(conv.weight.device)
    new.weight.data = conv.weight.data * s.view(-1, 1, 1)
    new.bias.data = (bn.bias - bn.running_mean * s).data
    return new


def q_weights(w):
    """8-bit weights, one scale per output channel."""
    a = w.abs().flatten(1).max(1)[0].clamp_min(1e-9).view(-1, *[1] * (w.dim() - 1))
    return torch.round(w / a * 127).clamp(-127, 127) * a / 127


def q_map(x, lo, hi):
    """8-bit feature map over the range [lo, hi]."""
    s = (hi - lo) / 255
    return torch.round((x.clamp(lo, hi) - lo) / s) * s + lo


class Int8BirdTCN:
    def __init__(self, m):
        self.first = fold(m.first)
        self.dw = [fold(x) for x in m.dw]
        self.pw = [fold(x) for x in m.pw]
        for c in [self.first, *self.dw, *self.pw]:
            c.weight.data = q_weights(c.weight.data)
        self.attn, self.fc = m.attn, m.fc
        self.aw, self.fw = q_weights(m.attn.weight.data), q_weights(m.fc.weight.data)
        self.ranges = {}

    def run(self, x, calibrate=False):
        def q(k, v):
            if calibrate:
                r = self.ranges.setdefault(k, [float("inf"), -float("inf")])
                r[0], r[1] = min(r[0], float(v.min())), max(r[1], float(v.max()))
                return v
            return q_map(v, *self.ranges[k])
        x = q("in", x.squeeze(1).transpose(1, 2))
        x = q("first", F.relu(self.first(x)))
        for i, (d, p) in enumerate(zip(self.dw, self.pw)):
            x = q(f"d{i}", F.relu(d(x)))
            x = q(f"p{i}", F.relu(p(x)))
        a = q("attn", F.conv1d(x, self.aw, self.attn.bias))
        w = torch.softmax(a, 2) if calibrate else q_map(torch.softmax(a, 2), 0.0, 1.0)
        v = q("pool", (x * w).sum(2))
        return q("out", F.linear(v, self.fw, self.fc.bias))

    def probs(self, x, calibrate=False):
        with torch.no_grad():
            return torch.cat([F.softmax(self.run(x[i:i + 1024], calibrate), 1) for i in range(0, len(x), 1024)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=REPO / "ML" / "Model" / "weights")
    ap.add_argument("--features", type=Path, default=T.FEATURES)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, names, _, _ = T.load(args.features, dev)
    n = len(names)
    c = data["test"]["clean"]
    res = []
    for f in sorted(args.weights.glob("seed*.pt")):
        m = BirdTCN(n_classes=n).to(dev)
        m.load_state_dict(torch.load(f, weights_only=False)["state_dict"])
        m.eval()
        torch.manual_seed(0)
        q = Int8BirdTCN(m)
        q.probs(data["train"]["x"][torch.randperm(len(data["train"]["x"]), device=dev)[:CALIB_CLIPS]], calibrate=True)
        _, cf = T.scores(T.probs(m, data["test"]["x"][c["idx"]]), c["y"], c["group"], n)
        _, cq = T.scores(q.probs(data["test"]["x"][c["idx"]]), c["y"], c["group"], n)
        res.append((T.per_class_acc(cf).mean(), T.per_class_acc(cq).mean()))
        print(f"{f.name}: clean 3 s test  float {res[-1][0]:.3f}  int8 {res[-1][1]:.3f}")
    r = np.array(res)
    print(f"mean: float {r[:, 0].mean():.3f} +/- {r[:, 0].std():.3f} | int8 {r[:, 1].mean():.3f} +/- {r[:, 1].std():.3f}")


if __name__ == "__main__":
    main()
