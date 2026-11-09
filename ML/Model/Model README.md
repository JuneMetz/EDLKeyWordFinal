###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Model: BirdTCN, a time-only separable CNN (`model.py`)

- **Input:** one MFCC grid per 1 s clip, shape `(batch, 1, 98, 13)` (June's MFCC: 98 frames x 13 coefficients,
  see `Firmware/Features/`), normalized with the train std and with **each clip's own mean removed** (see below).
- **Output:** 8 raw scores (logits): 7 birds + no bird. The device averages the softmax over 3 s before deciding.

## Layers (current best)
The 13 coefficients are the input channels. Every filter slides along time only.

| Layer | Output (channels x frames) | Weights |
|---|---|---|
| Input | 13 x 98 | 0 |
| Conv over time, kernel 9, stride 2, + batch norm + ReLU | 64 x 49 | 7,616 |
| 4 x [depthwise conv over time, kernel 9, dilation 1, 2, 4, 8 + pointwise 1x1], each + batch norm + ReLU | 64 x 49 | 19,712 |
| Attention pooling over time (one learned weight per frame) | 64 | 65 |
| Dropout 0.2 + fully connected | 8 | 520 |
| **Total** | | **27,913** |

Chip cost: about 28 KB of the 192 KB flash as int8, 1.29 M multiply-adds per 1 s clip, **peak RAM 6.1 KB** int8
(the largest input map + output map of one layer). Weights include batch-norm scale and shift, which are folded into
the conv weights before export.

## Why these choices
- **Time-only filters.** MFCC coefficients come out of a DCT, so coefficient 4 is not a "neighbour" of coefficient 5
  the way two frequency bins are. The old 2D model slid a 3x3 filter across them anyway and its maps (24 x 49 x 7)
  took 16.1 KB. Treating the 13 coefficients as channels gives maps of 64 x 49 = 3.1 KB, so the model can be
  wider and still use less than half the RAM. This alone moved validation from 0.807 to 0.845.
- **Each clip's own mean removed** (per-clip cepstral mean normalization). A microphone or a site colours the
  spectrum in a steady way, which shows up as a constant offset on each coefficient for the whole clip. Subtracting
  the clip's mean removes it, so the model cannot learn "this recording's microphone" instead of the bird. +0.012 on
  validation. **The chip must do the same:** after June's MFCC for a 1 s clip, subtract each coefficient's mean
  over the 98 frames, then divide by the train std saved with the weights.
- **Kernel 9 + dilation 1, 2, 4, 8.** Each block looks further back in time, so the last block covers about 1 s of
  song. Kernel 9 was +0.005 over kernel 5.
- **64 channels.** 96, 128 and even 256 channels (290,825 weights) were no better on validation, so size is no longer
  the limit; the number of recordings is.

## Old model (kept for reproducing old runs)
`DSCNN`: 2D depthwise separable CNN after ARM's Hello Edge, 24 channels, attention pooling, dilation 1, 2, 4, 8;
4,041 weights, 16.1 KB peak RAM, test 0.805. Train it with `train.py --arch ds2d ...` (see the Training README).

## Settings
Constants at the top of `model.py`: `N_FRAMES`, `N_COEFFS` (follow June's MFCC), `N_CLASSES`, `DROPOUT`,
`TCN_CHANNELS`, `TCN_FIRST_KERNEL`, `TCN_KERNEL`, `TCN_DILATIONS`, and the old DS-CNN ones.

## Check shapes, weights, MACs and RAM
```
python ML/Model/model.py
```
