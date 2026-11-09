###### Authors
- Jaimin Kadiwal
- Phuris Soontrapornchai

# Model: BirdTCN, a time-only separable CNN (`model.py`)

- **Input:** one MFCC grid per 1 s clip, shape `(batch, 1, 98, 13)` (June's MFCC: 98 frames x 13 coefficients,
  see `Firmware/Features/`), normalized with the train std and with **each clip's own mean removed** (see below).
- **Output:** 8 raw scores (logits): 7 birds + no bird. The device averages the softmax over 3 s before deciding.

## How it works, in plain words
Picture a team of 64 "listeners" reading a table that describes 1 second of sound.

1. **The sheet (input).** 1 s of audio becomes a table: 98 columns (moments, 10 ms apart) x 13 rows (how the
   sound looks at that moment: loud or quiet, bass or treble, ...). Each row's average over the clip is subtracted,
   so the microphone's colouring disappears and only the changes are left.
2. **First look: what is happening right now?** Each listener has a stencil 9 columns wide (about 0.1 s, a design
   choice: 9 scored better than 5) covering all 13 rows. Laid on the table, it multiplies and adds up everything
   under it and writes one number: "how much does this look like my sound?" (e.g. a rising whistle, a buzzy noise).
   The stencil jumps 2 columns at a time, so 98 columns give 49 numbers per listener: a notebook of 49 numbers.
3. **Four blocks: what is the pattern over time?** Each block has two steps:
   - *Look back* (depthwise): each listener re-reads only their own notebook, at points further apart in each block
     (1, 2, 4, 8 moments apart; this spacing is called dilation). Block 1 notices a single note, block 4 the rhythm
     of the whole second.
   - *Team meeting* (pointwise): at each moment, the 64 listeners combine what they found
     (e.g. "rising whistle" + "repeated 3 times" = "looks like a Robin phrase").
   After each step a batch norm (an automatic volume knob per listener, merged into the weights on the chip) and a
   ReLU (negative becomes 0: the listener either fires or stays quiet).
4. **Attention: which moments matter?** A second is mostly silence with a call somewhere. The model gives each
   moment an importance score and takes a weighted average, so the moments with a call count most. Each notebook
   shrinks to one number: 64 numbers describe the whole second.
5. **Vote: which bird?** Each of the 8 classes (7 birds + no bird) weighs the 64 numbers into a score; softmax turns
   the scores into percentages that add up to 100 %.
6. **Decision.** The device does this for 3 seconds in a row, averages the percentages and logs the winner.

**How it learned this:** nobody told the listeners what to look for. Training showed them thousands of labelled
clips, and after each wrong guess their stencils were nudged a little toward the right answer.

**Why it fits the chip:** the old model's listeners wrote 7 rows of notes per moment (a 3 x 3 window sliding along
time and across the 13 rows); BirdTCN's listeners read all 13 rows at once and write 1 number per moment. Notebooks
7 times thinner leave room for 64 listeners instead of 24 and still use 6.1 KB of RAM instead of 16.1 KB. Reading all
13 rows together also makes more sense, because the rows are 13 different descriptions, not neighbours.

**In one sentence:** listeners spot short sounds, the blocks connect them into song patterns, attention focuses on
the moments with calls, and a vote over 3 seconds picks the bird.

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
