"""
Bird classifier models. Input: one MFCC grid per 1 s clip, shape (batch, 1, N_FRAMES, N_COEFFS).
Output: N_CLASSES raw scores (logits), one per class (7 birds + "no bird").

BirdTCN (CURRENT BEST, 11-06): time-only depthwise separable CNN.
  The 13 MFCC coefficients are the input CHANNELS and every filter slides along time only.
  Why: MFCC coefficients are not neighbours in frequency (they come out of a DCT), so sliding a 2D filter across
  them (DSCNN below) spends RAM on a pattern that has no physical meaning. Time-only maps are also tiny:
  64 channels x 49 frames = 3.1 KB int8, so the model can be wider and still use less RAM.
  Layers:
    1. conv over time, kernel K0, stride 2: 13 -> CHANNELS maps            (98 frames -> 49 frames)
    2. 4 x [depthwise conv over time, kernel K, dilation 1, 2, 4, 8 + pointwise 1x1], each + batch norm + ReLU
    3. attention pooling: a learned weight per frame (softmax over time), so frames with a call count more
    4. dropout + fully connected -> N_CLASSES scores
  The input must be "per-clip mean removed" (train.py does it; the chip must too): from every coefficient subtract
  its own mean over the 98 frames of the clip. This removes the steady colouring of the microphone and the site.

DSCNN (old best, 11-01, test 0.805): 2D depthwise separable CNN after Hello Edge (ARM, 2017). Kept so old runs
  can be reproduced (train.py --arch ds2d).

The feature recipe belongs to June (Firmware/Features). N_FRAMES and N_COEFFS follow her MFCC
(30 ms frames, 10 ms hop, 13 coefficients -> 98 x 13 per 1 s clip). If her recipe changes, change them here.

Check shapes, weights, MACs and RAM:
  python ML/Model/model.py
"""
import torch
from torch import nn

N_FRAMES = 98       # frames per 1 s clip (June's MFCC: 30 ms window, 10 ms hop)
N_COEFFS = 13       # MFCC coefficients per frame (c0..c12)
N_CLASSES = 8       # 7 birds + no bird
DROPOUT = 0.2

# BirdTCN (current best)
TCN_CHANNELS = 64           # maps in every layer; 64 x 49 int8 = 3.1 KB per map
TCN_FIRST_KERNEL = 9        # frames seen by the first conv (90 ms)
TCN_KERNEL = 9              # frames per depthwise conv
TCN_DILATIONS = (1, 2, 4, 8)    # with stride 2 the last block spans about 1 s of song

# DSCNN (old)
CHANNELS = 24
N_BLOCKS = 4
FIRST_KERNEL = (3, 3)   # (frames, coeffs)
FIRST_STRIDE = (2, 2)
DILATIONS = (1, 2, 4, 8)
POOL = "attn"           # "avg" or "attn"


def conv1d_bn_relu(c_in, c_out, kernel, stride=1, groups=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=dilation * (kernel // 2), groups=groups,
                  dilation=dilation, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class BirdTCN(nn.Module):
    def __init__(self, n_in=N_COEFFS, n_classes=N_CLASSES, channels=TCN_CHANNELS, first_kernel=TCN_FIRST_KERNEL,
                 kernel=TCN_KERNEL, dilations=TCN_DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.first = conv1d_bn_relu(n_in, channels, first_kernel, stride=2)
        self.dw = nn.ModuleList([conv1d_bn_relu(channels, channels, kernel, groups=channels, dilation=d)
                                 for d in dilations])                                   # depthwise
        self.pw = nn.ModuleList([conv1d_bn_relu(channels, channels, 1) for _ in dilations])   # pointwise
        self.attn = nn.Conv1d(channels, 1, 1)                                           # one score per frame
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, n_classes)

    def features(self, x):
        x = x.squeeze(1).transpose(1, 2)        # (batch, 1, frames, coeffs) -> (batch, coeffs, frames)
        x = self.first(x)
        for dw, pw in zip(self.dw, self.pw):
            x = pw(dw(x))
        return x

    def pool(self, x):
        w = torch.softmax(self.attn(x), dim=2)  # (batch, 1, frames), sums to 1 over time
        return (x * w).sum(dim=2)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x))))


def conv_bn_relu(c_in, c_out, kernel, stride=1, groups=1, dilation=(1, 1)):
    kernel = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
    pad = (dilation[0] * (kernel[0] // 2), dilation[1] * (kernel[1] // 2))
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad, groups=groups, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class DSCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES, channels=CHANNELS, n_blocks=N_BLOCKS, dropout=DROPOUT,
                 first_kernel=FIRST_KERNEL, first_stride=FIRST_STRIDE, dilations=DILATIONS, pool=POOL):
        super().__init__()
        assert len(dilations) == n_blocks and pool in ("avg", "attn")
        self.first = conv_bn_relu(1, channels, first_kernel, stride=first_stride)
        blocks = []
        for d in dilations:
            blocks.append(conv_bn_relu(channels, channels, 3, groups=channels, dilation=(d, 1)))   # depthwise
            blocks.append(conv_bn_relu(channels, channels, 1))                                    # pointwise
        self.blocks = nn.Sequential(*blocks)
        self.pool_mode = pool
        self.attn = nn.Conv1d(channels, 1, 1) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, n_classes)

    def pool(self, x):
        if self.pool_mode == "avg":
            return x.mean(dim=(2, 3))
        x = x.mean(dim=3)
        w = torch.softmax(self.attn(x), dim=2)
        return (x * w).sum(dim=2)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.blocks(self.first(x)))))


def chip_cost(model, n_frames=N_FRAMES, n_coeffs=N_COEFFS):
    """(weights, MACs per 1 s clip, peak int8 RAM in bytes). Peak RAM = the largest input map + output map of any
    conv layer, since the chip must hold both while it computes that layer."""
    macs, peak = [0], [0]

    def hook(m, i, o):
        peak[0] = max(peak[0], i[0].numel() + o.numel())
        if isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features
        else:
            k = 1
            for v in m.kernel_size:
                k *= v
            out_pos = o[0, 0].numel()
            macs[0] += k * (m.in_channels // m.groups) * out_pos * m.out_channels

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, n_frames, n_coeffs))
    for h in hooks:
        h.remove()
    return sum(p.numel() for p in model.parameters()), macs[0], peak[0]


if __name__ == "__main__":
    for name, model in (("BirdTCN (current best)", BirdTCN()), ("DSCNN (old)", DSCNN())):
        print(f"== {name}")
        x = torch.zeros(2, 1, N_FRAMES, N_COEFFS)
        if isinstance(model, BirdTCN):
            x = x.squeeze(1).transpose(1, 2)
            print("input  (coeffs x frames)", tuple(x.shape[1:]))
            x = model.first(x)
            print("conv1                  ", tuple(x.shape[1:]))
            for i, (dw, pw) in enumerate(zip(model.dw, model.pw)):
                x = pw(dw(x))
                print(f"block {i + 1}                ", tuple(x.shape[1:]))
        w, macs, ram = chip_cost(model)
        print(f"weights {w:,} (~{w / 1024:.1f} KB int8), MACs {macs / 1e6:.2f} M per 1 s clip, "
              f"peak RAM {ram / 1024:.1f} KB int8 (nRF52805: 24 KB RAM, 192 KB flash)")
