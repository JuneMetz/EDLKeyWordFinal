"""
Bird classifier models. Input: one feature grid per 1 s clip, shape (batch, 1, N_FRAMES, N_COEFFS).
Output: N_CLASSES raw scores (logits), one per class (7 birds + "no bird").

BirdTCN: the model that runs on the chip (final model: test 0.901, see ML/Training/Training README.md).
  A time-only depthwise separable CNN. The 13 coefficients are the input CHANNELS and every filter slides along
  time only. MFCC coefficients are not neighbours in frequency (they come out of a DCT), so a filter that slides
  across them would spend RAM on a pattern with no physical meaning. Time-only maps are also small:
  64 channels x 49 frames = 3.1 KB in 8-bit.
  Layers:
    1. conv over time, kernel 9, stride 2: 13 -> 64 maps                    (98 frames -> 49 frames)
    2. 4 x [depthwise conv over time, kernel 9, dilation 1, 2, 4, 8 + pointwise 1x1], each + batch norm + ReLU
    3. attention pooling: a learned weight per frame (softmax over time), so frames with a call count more
    4. dropout + fully connected -> 8 scores
  27,913 weights (about 28 KB of flash in 8-bit), 1.29 M multiply-adds per 1 s clip, peak RAM 6.1 KB in 8-bit.
  The input must have each clip's own mean removed: from every coefficient subtract its mean over the 98 frames
  of the clip (train.py does it; the chip must too).

TeacherTCN: a much larger model (about 3.9 M weights, 256 channels, 6 residual blocks) used ONLY on the laptop
  during training. BirdTCN learns from the teacher's softened answers (knowledge distillation). The teacher never
  goes on the chip.

The feature recipe belongs to June (Firmware/Features). N_FRAMES and N_COEFFS follow her MFCC
(30 ms frames, 10 ms hop, 13 coefficients -> 98 x 13 per 1 s clip). If her recipe changes, change them here.

Check shapes, weights, multiply-adds and RAM:
  python ML/Model/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F

N_FRAMES = 98       # frames per 1 s clip (June's MFCC: 30 ms window, 10 ms hop)
N_COEFFS = 13       # coefficients per frame (c0..c12)
N_CLASSES = 8       # 7 birds + no bird
DROPOUT = 0.2

TCN_CHANNELS = 64           # maps in every layer; 64 x 49 int8 = 3.1 KB per map
TCN_FIRST_KERNEL = 9        # frames seen by the first conv (90 ms)
TCN_KERNEL = 9              # frames per depthwise conv
TCN_DILATIONS = (1, 2, 4, 8)    # with stride 2 the last block spans about 1 s of song

TEACHER_CHANNELS = 256
TEACHER_DILATIONS = (1, 2, 4, 8, 16, 32)


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


class TeacherBlock(nn.Module):
    """Residual block: full conv over time (kernel k, dilation d) + BN + ReLU, pointwise conv + BN, dropout, skip."""
    def __init__(self, c, k, d, drop):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = self.bn2(self.pw(h))
        return F.relu(x + self.drop(h))


class TeacherTCN(nn.Module):
    """Training-only teacher. Same input and output as BirdTCN, no chip limits."""
    def __init__(self, n_in=N_COEFFS, n_classes=N_CLASSES, channels=TEACHER_CHANNELS, kernel=TCN_KERNEL,
                 dilations=TEACHER_DILATIONS, drop=0.2):
        super().__init__()
        self.first = nn.Sequential(nn.Conv1d(n_in, channels, kernel, stride=2, padding=kernel // 2, bias=False),
                                   nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[TeacherBlock(channels, kernel, d, drop) for d in dilations])
        self.attn = nn.Conv1d(channels, 1, 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.blocks(self.first(x.squeeze(1).transpose(1, 2)))
        w = torch.softmax(self.attn(x), dim=2)
        return self.fc(self.drop((x * w).sum(dim=2)))


def chip_cost(model, n_frames=N_FRAMES, n_coeffs=N_COEFFS):
    """(weights, multiply-adds per 1 s clip, peak int8 RAM in bytes). Peak RAM = the largest input map + output map
    of any conv layer, since the chip must hold both while it computes that layer."""
    macs, peak = [0], [0]

    def hook(m, i, o):
        peak[0] = max(peak[0], i[0].numel() + o.numel())
        if isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features
        else:
            macs[0] += m.kernel_size[0] * (m.in_channels // m.groups) * o[0, 0].numel() * m.out_channels

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, n_frames, n_coeffs))
    for h in hooks:
        h.remove()
    return sum(p.numel() for p in model.parameters()), macs[0], peak[0]


if __name__ == "__main__":
    model = BirdTCN()
    print("== BirdTCN (runs on the chip)")
    x = torch.zeros(2, 1, N_FRAMES, N_COEFFS).squeeze(1).transpose(1, 2)
    print("input  (coeffs x frames)", tuple(x.shape[1:]))
    x = model.first(x)
    print("conv1                  ", tuple(x.shape[1:]))
    for i, (dw, pw) in enumerate(zip(model.dw, model.pw)):
        x = pw(dw(x))
        print(f"block {i + 1}                ", tuple(x.shape[1:]))
    w, macs, ram = chip_cost(model)
    print(f"weights {w:,} (~{w / 1024:.1f} KB int8), multiply-adds {macs / 1e6:.2f} M per 1 s clip, "
          f"peak RAM {ram / 1024:.1f} KB int8 (nRF52805: 24 KB RAM, 192 KB flash)")
    t = TeacherTCN()
    print(f"== TeacherTCN (laptop only, for training): {sum(p.numel() for p in t.parameters()):,} weights")
