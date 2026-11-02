###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Model: small DS-CNN (`model.py`)

A depthwise-separable CNN after ARM's "Hello Edge" keyword-spotting design, sized for the nRF52805 (24 KB RAM).

- **Input:** one MFCC grid per 1 s clip, shape `(batch, 1, 98, 13)` (June's MFCC: 98 frames x 13 coefficients,
  see `Firmware/Features/`).
- **Output:** 8 raw scores (logits): 7 birds + no bird. The device averages the softmax over 3 s before deciding.

| Layer | Output shape | Weights |
|---|---|---|
| Input | 1 x 98 x 13 | 0 |
| Conv 3x3, stride 2, + batch norm + ReLU | 24 x 49 x 7 | 264 |
| 4 x [depthwise 3x3 (dilation 1, 2, 4, 8 in time) + pointwise 1x1], each + batch norm + ReLU | 24 x 49 x 7 | 3,552 |
| Attention pooling over time (one learned weight per frame) | 24 | 25 |
| Dropout 0.2 + fully connected | 8 | 200 |
| **Total** | | **4,041** |

Numbers for the **current best model** (24 channels, attention pooling, dilated). Weights include batch-norm scale
and shift, which can be folded into the conv weights before export.

## Why these choices
- **Depthwise + pointwise** convs: at 24 channels a layer needs 792 weights instead of 5,184 for a normal 3x3 conv.
- **Dilation in time** lets the 4 blocks hear about 0.6 s of song instead of 0.2 s; it was the biggest model-side gain
  (+0.05 macro accuracy) at no extra weights or RAM.
- **Attention pooling** lets the frames that hold a call count more than the silence around them (+0.01).
- **24 channels**: the two largest feature maps take 16.1 KB as int8, leaving room in 24 KB RAM for audio and MFCC.
  32 channels needs 21.4 KB and 48 channels 32.2 KB; neither was clearly better.

## Settings
Constants at the top of `model.py` (all can also be passed to `DSCNN(...)`):
`N_FRAMES`, `N_COEFFS` (follow June's MFCC), `N_CLASSES`, `CHANNELS`, `N_BLOCKS`, `DROPOUT`, `FIRST_KERNEL`,
`FIRST_STRIDE`, `DILATIONS`, `POOL` (`"avg"` or `"attn"`). The file defaults are the plain baseline
(32 channels, no dilation, average pooling); the best model is built by `train.py` flags
(see `ML/Training/Training README.md`).

## Check shapes, size and RAM
```
python ML/Model/model.py
```
prints each layer's output shape, the number of weights and the int8 RAM for the two largest feature maps.
