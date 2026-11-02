"""
Small DS-CNN (depthwise separable CNN) for the bird classifier, after Hello Edge (ARM, 2017).

Input:  one MFCC grid per 1 s clip, shape (batch, 1, N_FRAMES, N_COEFFS)
Output: N_CLASSES raw scores (logits), one per class (7 birds + "no bird")

Layers:
  1. normal conv FIRST_KERNEL (time x coeff), stride FIRST_STRIDE -> CHANNELS maps
  2. N_BLOCKS x depthwise separable block:
       depthwise conv 3x3 (one filter per channel, groups=CHANNELS), dilated in time by DILATIONS[i]
       pointwise conv 1x1 (mixes the channels)
     each conv is followed by batch norm + ReLU
  3. pooling -> CHANNELS numbers. POOL "avg": mean over every frame and coefficient.
     POOL "attn": mean over coefficients, then a learned weight per frame (softmax over time), so the few
     frames that hold a call count more than the silence around them.
  4. dropout + fully connected       -> N_CLASSES scores

The feature recipe belongs to June (Firmware/Features). The shape constants below follow her MFCC
(30 ms frames, 10 ms hop, 13 coefficients -> 98 x 13 per 1 s clip). If her recipe changes, change them here.

Check the shapes and size:
  python ML/Model/model.py
"""
import torch
from torch import nn

N_FRAMES = 98       # frames per 1 s clip (June's MFCC: 30 ms window, 10 ms hop)
N_COEFFS = 13       # MFCC coefficients per frame (c0..c12)
N_CLASSES = 8       # 7 birds + no bird
CHANNELS = 32       # feature maps in every layer (sets RAM on the chip, see activation_bytes)
N_BLOCKS = 4        # depthwise separable blocks
DROPOUT = 0.2
FIRST_KERNEL = (3, 3)   # (frames, coeffs) of the first conv; Hello Edge uses (10, 4) = 100 ms of time
FIRST_STRIDE = (2, 2)   # (frames, coeffs); a wider input (e.g. 40 log-mel bands) can use (2, 4) to save RAM
DILATIONS = (1, 1, 1, 1)    # time dilation per block; (1, 2, 4, 8) sees ~0.6 s of song instead of ~0.2 s
POOL = "avg"            # "avg" or "attn"


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
        self.attn = nn.Conv1d(channels, 1, 1) if pool == "attn" else None    # one score per frame
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, n_classes)

    def pool(self, x):
        if self.pool_mode == "avg":
            return x.mean(dim=(2, 3))
        x = x.mean(dim=3)                                   # (batch, channels, frames)
        w = torch.softmax(self.attn(x), dim=2)              # (batch, 1, frames), sums to 1 over time
        return (x * w).sum(dim=2)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        return self.fc(self.drop(self.pool(x)))


def activation_bytes(model, n_frames=N_FRAMES, n_coeffs=N_COEFFS, bytes_per_value=1):
    """Largest feature map in the network, in bytes (int8 = 1 byte). The chip needs about 2 of these at once."""
    sizes = []
    hooks = [m.register_forward_hook(lambda m, i, o: sizes.append(o.numel()))
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, n_frames, n_coeffs))
    for h in hooks:
        h.remove()
    return max(sizes) * bytes_per_value


if __name__ == "__main__":
    model = DSCNN()
    x = torch.zeros(2, 1, N_FRAMES, N_COEFFS)
    print("input ", tuple(x.shape))
    x = model.first(x)
    print("conv1 ", tuple(x.shape))
    for i, layer in enumerate(model.blocks):
        x = layer(x)
        print(f"block {i // 2 + 1} {'depthwise' if i % 2 == 0 else 'pointwise'}", tuple(x.shape))
    x = model.pool(x).flatten(1)
    print("pool  ", tuple(x.shape))
    print("output", tuple(model.fc(x).shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (int8 weights ~{n_params / 1024:.1f} KB)")
    act = activation_bytes(model)
    print(f"largest feature map: {act:,} bytes int8, ~{2 * act / 1024:.1f} KB for two buffers (nRF52805 RAM: 24 KB)")
